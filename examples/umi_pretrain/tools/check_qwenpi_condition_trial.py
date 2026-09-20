"""One additional switch against decoder-LN reference; unchanged 200-step plan.

The step-50 screen tests learned noise response and generation, not just nonzero
activations. Failure requests a normal trainer pause; full checkpoint is retained.
Thresholds are a bounded engineering screen, not universal learning criteria.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import torch
import yaml

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from starVLA.training.trainer_utils.umi_checkpoint import write_json,sha256_file
from check_qwenpi_learnability import FixedEvaluation
from diagnose_qwenpi_condition import diagnose_condition
from diagnose_qwenpi_action_decoder import diagnose


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    plan=yaml.safe_load((args.reference_run/'plan.yaml').read_text())
    action=plan['framework']['action_model']
    if action.get('decoder_input_norm')!='layer_norm':
        raise ValueError('Use the completed decoder-LN reference, not the original run')
    if sum(s['updates'] for s in plan['stages'])!=200:
        raise ValueError('Keep the original 200-update schedule')
    action.setdefault('diffusion_model_cfg',{})['cross_condition_norm']='layer_norm'
    (root/'plan.yaml').write_text(yaml.safe_dump(plan,sort_keys=False))
    write_json(root/'trial_identity.json',dict(reference_run=str(args.reference_run),
        only_changed_configuration='framework.action_model.diffusion_model_cfg.cross_condition_norm',
        fresh_initialization=True,scheduled_updates=200,screen_at_update=50,
        script_sha256=sha256_file(Path(__file__)),
        screen_thresholds=dict(noise_directional_gain=.05,noise_cosine=.1,
            fixed_fm_relative_to_initial=.8,generated_mse_relative_to_initial=.8,
            endpoint_mse_relative_to_bias=.9,state_delta_rms=1e-5,state_repeat_multiplier=10)))
    from starVLA.training import train_umi_pretrain as training
    evaluator=FixedEvaluation(plan,root); holder={}; diagnostics=[]
    original_prepare=training.UMIAccelerator.prepare
    original_evaluate=training.evaluate
    original_commit=training.UMITrainingState.commit_update

    def record(step):
        report=[]
        for index in (0,16,31):
            report.append(diagnose_condition(holder['model'],evaluator.samples[index],index,
                policies=('normal',),times=(0.,),details=index==0))
        diagnostics.append(dict(step=step,samples=report))
        write_json(root/'condition_timeline.json',diagnostics)
        decoder=diagnose(holder['model'],evaluator.samples[0],0,include_sampling=False,modes=('eval',))
        # Keep precise gradients and residual statistics without the 1024-channel lists.
        for fm in decoder['fm'].values():
            for activation in fm['activations']:
                activation.pop('per_channel_positive_fraction',None)
        write_json(root/f'decoder_{step:06d}.json',decoder)
        return report

    def prepare(accelerator,*values,**kwargs):
        result=original_prepare(accelerator,*values,**kwargs)
        model=accelerator.unwrap_model(result[0]); holder['model']=model;holder['step']=0
        probes={}
        for name,p in model.named_parameters():
            if name in ('action_model.action_decoder.layer1.weight','action_model.action_decoder.layer2.weight',
                        'action_model.state_encoder.layer1.weight','action_model.future_tokens.weight'):
                probes[name]=p.detach().flatten()[:64].cpu().tolist()
        write_json(root/'initial_parameter_probes.json',probes)
        old=json.loads((args.reference_run/'initial_parameter_probes.json').read_text())
        if probes!=old: raise ValueError('Initial parameter probes differ from the reference')
        evaluator.measure(model,0); record(0)
        return result

    def commit(state):
        result=original_commit(state); holder['step']=state.global_update_step
        return result

    def evaluate(accelerator,model,*unused):
        step=holder['step']
        result=evaluator.measure(accelerator.unwrap_model(model),step)
        if step in (50,200):
            report=record(step)
            if step==50:
                rows=[x['rows'][0] for x in report]
                noise=[r['interventions']['noisy_action']['noise_endpoint_reference'] for r in rows]
                checks=dict(
                    noise_directional_gain=float(np.mean([n['directional_gain'] for n in noise]))>.05,
                    noise_cosine=float(np.mean([n['cosine'] for n in noise]))>.1,
                    fixed_fm_decreased=result['fixed_fm_loss']<.8*evaluator.history[0]['fixed_fm_loss'],
                    generation_improved=result['normalized_action_mse']<.8*evaluator.history[0]['normalized_action_mse'],
                    endpoint_beats_bias=bool(np.mean([r['fixed_velocity_mse'] for r in rows])<.9*np.mean([r['bias_velocity_mse'] for r in rows])),
                    state_above_repeat=all(max(r['interventions'][f'state_{c}']['velocity']['rms'] for c in (0,8,15))>
                        max(1e-5,10*r['repeat_velocity']['rms']) for r in rows))
                passed=all(checks.values())
                write_json(root/'screen_50.json',dict(continue_to_200=passed,checks=checks,
                    noise_reference=noise,evaluation=result,scope='engineering continuation screen, not final acceptance'))
                print('SCREEN_50 '+json.dumps(dict(passed=passed,checks=checks)),flush=True)
                if not passed:
                    # Existing SIGTERM handler pauses at the committed boundary and saves normally.
                    os.kill(os.getpid(),signal.SIGTERM)
        return result

    training.UMIAccelerator.prepare=prepare
    training.UMITrainingState.commit_update=commit
    training.evaluate=evaluate
    try:
        training.run(argparse.Namespace(plan=root/'plan.yaml',output_dir=root/'train',resume=None,
            stop_after_update=None,cpu=False))
    finally:
        training.UMIAccelerator.prepare=original_prepare
        training.UMITrainingState.commit_update=original_commit
        training.evaluate=original_evaluate
    first,last=evaluator.history[0],evaluator.history[-1]
    checks=dict(fixed_fm_decreased=last['fixed_fm_loss']<first['fixed_fm_loss'],
        generated_mse_decreased=last['normalized_action_mse']<first['normalized_action_mse'],
        position_beats_hold=last['prediction']['position_rmse']<last['hold_current']['position_rmse'],
        gripper_beats_hold=last['prediction']['gripper_rmse']<last['hold_current']['gripper_rmse'],
        rotation_beats_hold=last['prediction']['rotation_mean_degrees']<last['hold_current']['rotation_mean_degrees'])
    write_json(root/'summary.json',dict(status='completed' if holder['step']==200 else 'paused_at_screen',
        actual_updates=holder['step'],scheduled_updates=200,checks=checks,criteria_met=all(checks.values()),
        first=first,last=last,scope='32 overlapping training/evaluation windows; no generalization claim'))


if __name__=='__main__': main()
