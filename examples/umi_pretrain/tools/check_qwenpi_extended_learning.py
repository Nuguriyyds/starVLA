"""Analytic sampling check, then one bounded 2000-update dual-LN learning run.

Reuses the production trainer and fixed 32 windows. No step-50 performance gate,
architecture changes, optimizer changes, or copied Euler integrator.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import yaml

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from starVLA.training.trainer_utils.umi_checkpoint import preserve_rng,write_json,sha256_file
from check_qwenpi_learnability import FixedEvaluation
from diagnose_qwenpi_condition import diagnose_condition

EVALUATION_STEPS=(0,50,200,500,1000,2000)


def oracle_check(model,evaluator):
    """Inject truth-based velocity into real decoder output; retain real Euler."""
    from starVLA.training.trainer_utils.umi_training_data import make_loader
    from starVLA.training.train_umi_pretrain import seed_start
    sample=evaluator.samples[0]
    loader=make_loader(evaluator.plan,evaluator.plan['evaluation'],evaluator.root/'train',evaluation=True,record=False)
    try:
        raw_target=np.asarray(loader.dataset.raw_dataset[0]['action'],dtype=np.float32)[None]
    finally: loader.dataset.close()
    target_np=np.asarray(sample['action'],dtype=np.float32)[None]
    if not np.allclose(evaluator.normalizer.normalize_action(raw_target),target_np,rtol=1e-5,atol=1e-6):
        raise ValueError('Oracle raw and normalized targets do not correspond')
    head=model.action_model
    if head.num_inference_timesteps!=4 or head.num_timestep_buckets%4:
        raise ValueError('This oracle check expects the unchanged four-step sampler')
    target=torch.as_tensor(target_np,device=head.device,dtype=head.dtype)
    capture={}; times=[]; states=[]; hooks=[]; previous=model.training
    def actual_input(module,args):
        capture['x']=args[0].detach().clone()
        capture['t']=float(args[1][0])/head.num_timestep_buckets
    def oracle_output(module,args,output):
        x,t=capture['x'],capture['t']
        if not 0<=t<1: raise ValueError('Oracle received invalid generation time')
        times.append(t);states.append(x.cpu().numpy())
        replacement=torch.zeros_like(output)
        replacement[:,-head.action_horizon:]=(target-x)/(1-t)
        return replacement
    try:
        with preserve_rng(),torch.no_grad():
            model.eval();seed_start(19876)
            hooks.append(head.action_encoder.register_forward_pre_hook(actual_input))
            hooks.append(head.action_decoder.register_forward_hook(oracle_output))
            inputs={k:v for k,v in sample.items() if k!='action'}
            generated=model.predict_action(examples=[inputs])['normalized_actions']
    finally:
        for h in hooks: h.remove()
        model.train(previous)
    raw_generated=evaluator.normalizer.inverse_action(generated)
    normalized_error=float(np.max(np.abs(generated-target_np)))
    raw_error=float(np.max(np.abs(raw_generated-raw_target)))
    path_errors=[float(np.max(np.abs(x-((1-t)*states[0]+t*target_np)))) for t,x in zip(times,states)]
    passed=(times==[0.,.25,.5,.75] and np.isfinite(generated).all()
            and normalized_error<1e-5 and raw_error<1e-5 and max(path_errors)<1e-5)
    report=dict(passed=bool(passed),times=times,normalized_max_error=normalized_error,
        raw_max_error=raw_error,intermediate_path_max_errors=path_errors,
        real_euler_and_action_slice=True,raw_target_read_independently=True,
        truth_injected_for_chain_check_only=True,parameter_updates=False)
    write_json(evaluator.root/'oracle_check.json',report)
    print('ORACLE '+json.dumps(report),flush=True)
    if not passed: raise RuntimeError('Analytic velocity failed actual sampling/inverse chain')


class ExtendedEvaluation(FixedEvaluation):
    def measure(self,model,step):
        """Same fixed FM noise as before; bias baseline observes that actual draw."""
        bias_losses=[]; pending={}
        head=model.action_model
        original_randn=torch.randn
        def labels(module,args,kwargs):
            if 'target' in pending: raise RuntimeError('Previous FM draw was not captured')
            pending['target']=args[1] if len(args)>1 else kwargs['actions']
        def tracked_randn(*args,**kwargs):
            noise=original_randn(*args,**kwargs)
            if 'target' in pending:
                target=pending.pop('target')
                if noise.shape!=target.shape: raise ValueError('Unexpected FM noise shape')
                loss=((head.action_decoder.layer2.bias.detach()-(target-noise))**2).mean()
                bias_losses.append(float(loss))
            return noise
        hook=head.register_forward_pre_hook(labels,with_kwargs=True)
        try:
            with patch('torch.randn',tracked_randn):
                result=super().measure(model,step)
        finally: hook.remove()
        if pending or len(bias_losses)!=len(self.samples):
            raise RuntimeError('Expected exactly one FM draw per evaluation sample')
        result['same_noise_bias_only_loss']=float(np.mean(bias_losses))
        if not np.isfinite(result['same_noise_bias_only_loss']):
            raise ValueError('Nonfinite same-noise bias baseline')
        noise=[]
        for index in (0,16,31):
            row=diagnose_condition(model,self.samples[index],index,
                policies=('normal',),times=(0.,),details=False)['rows'][0]
            noise.append(dict(window_index=index,repeat_rms=row['repeat_velocity']['rms'],
                **row['interventions']['noisy_action']['noise_endpoint_reference']))
        result['noise_endpoint_response']=noise
        result['inference_steps']=head.num_inference_timesteps
        write_json(self.root/'fixed_evaluations.json',self.history)
        print('EXTENDED_EVAL '+json.dumps(result),flush=True)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('This bounded run is single-process only')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    plan=yaml.safe_load((args.reference_run/'plan.yaml').read_text())
    action=plan['framework']['action_model']
    if action.get('decoder_input_norm')!='layer_norm' or action['diffusion_model_cfg'].get('cross_condition_norm')!='layer_norm':
        raise ValueError('Reference must be the current dual-LN configuration')
    if len(plan['stages'])!=1 or plan['training']['warmup_updates']!=10:
        raise ValueError('Reference training recipe changed')
    plan['stages'][0]['updates']=2000
    # The existing trainer calls this tool at multiples of 50. Only named milestones
    # do work; no production-loop change is needed for the irregular schedule.
    plan['training']['eval_every']=50
    plan['checkpoint']=dict(integrity='basic',every_updates=1000)
    (root/'plan.yaml').write_text(yaml.safe_dump(plan,sort_keys=False))
    write_json(root/'trial_identity.json',dict(reference_run=str(args.reference_run),
        fresh_initialization=True,scheduled_updates=2000,evaluation_steps=list(EVALUATION_STEPS),
        checkpoint_steps=[1000,2000],soft_early_stopping=False,
        changed_training_controls=['total_updates: 200 -> 2000','linear decay follows 2000-update horizon',
            'remove step-50 soft gate','sparse evaluation and checkpoint cadence'],
        unchanged=['dual-LN model','continuous state','32-window view','normalization','seed',
            'full-model training','peak learning rates','10-update warmup','batch 1 / accumulation 2','4-step sampling'],
        script_sha256=sha256_file(Path(__file__))))
    from starVLA.training import train_umi_pretrain as training
    evaluator=ExtendedEvaluation(plan,root);holder={'step':0}
    prepare_old=training.UMIAccelerator.prepare
    evaluate_old=training.evaluate
    commit_old=training.UMITrainingState.commit_update
    def prepare(accelerator,*values,**kwargs):
        result=prepare_old(accelerator,*values,**kwargs)
        model=accelerator.unwrap_model(result[0]);holder['model']=model
        probes={name:p.detach().flatten()[:64].cpu().tolist() for name,p in model.named_parameters()
            if name in ('action_model.action_decoder.layer1.weight','action_model.action_decoder.layer2.weight',
                        'action_model.state_encoder.layer1.weight','action_model.future_tokens.weight')}
        if probes!=json.loads((args.reference_run/'initial_parameter_probes.json').read_text()):
            raise ValueError('Initial parameter probes do not match the short reference')
        write_json(root/'initial_parameter_probes.json',probes)
        evaluator.load_samples()
        if len(evaluator.samples)!=32: raise ValueError('Expected the same 32 fixed windows')
        oracle_check(model,evaluator)
        evaluator.measure(model,0)
        return result
    def commit(state):
        result=commit_old(state);holder['step']=state.global_update_step
        return result
    def evaluate(accelerator,model,*unused):
        step=holder['step']
        if step not in EVALUATION_STEPS:
            return dict(skipped=True,reason='outside predefined milestones',step=step)
        return evaluator.measure(accelerator.unwrap_model(model),step)
    training.UMIAccelerator.prepare=prepare
    training.UMITrainingState.commit_update=commit
    training.evaluate=evaluate
    try:
        training.run(argparse.Namespace(plan=root/'plan.yaml',output_dir=root/'train',resume=None,
            stop_after_update=None,cpu=False))
    finally:
        training.UMIAccelerator.prepare=prepare_old
        training.UMITrainingState.commit_update=commit_old
        training.evaluate=evaluate_old
    first,last=evaluator.history[0],evaluator.history[-1]
    checks={name:last['prediction'][metric]<last['hold_current'][metric] for name,metric in
        [('position_beats_hold','position_rmse'),('gripper_beats_hold','gripper_rmse'),
         ('rotation_beats_hold','rotation_mean_degrees')]}
    summary=dict(status='completed' if holder['step']==2000 else 'paused',actual_updates=holder['step'],
        checks=checks,first=first,last=last,scope='fixed training windows only; not generalization')
    write_json(root/'summary.json',summary)
    # A single optional inference-only diagnostic; do not compensate for a constant field.
    meaningful_noise=float(np.mean([x['directional_gain'] for x in last['noise_endpoint_response']]))>.1
    if holder['step']==2000 and last['fixed_fm_loss']<.5*first['fixed_fm_loss'] and meaningful_noise and not all(checks.values()):
        model=holder['model'];old_steps=model.action_model.num_inference_timesteps
        extra_root=root/'sampling_16';extra_root.mkdir()
        extra=FixedEvaluation(plan,extra_root);extra.samples=evaluator.samples;extra.normalizer=evaluator.normalizer
        try:
            model.action_model.num_inference_timesteps=16
            extra_result=extra.measure(model,2000)
        finally: model.action_model.num_inference_timesteps=old_steps
        summary['same_weights_sampling_16']=extra_result
        write_json(root/'summary.json',summary)


if __name__=='__main__': main()
