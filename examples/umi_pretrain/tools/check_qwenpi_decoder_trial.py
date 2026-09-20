"""One-variable decoder trial; same 32 windows and unchanged 200-update schedule.

reproduce: stop at 50, no checkpoint, no optimizer/architecture changes.
layer_norm: add only decoder input LayerNorm, run 200 if the path survives 50.
Diagnostic probes restore RNG/mode and run only at committed update boundaries.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from starVLA.training.trainer_utils.umi_checkpoint import write_json, sha256_file
from check_qwenpi_learnability import FixedEvaluation
from diagnose_qwenpi_action_decoder import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=('reproduce', 'layer_norm'), required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = yaml.safe_load((args.reference_run / 'plan.yaml').read_text())
    if sum(s['updates'] for s in plan['stages']) != 200:
        raise ValueError('Reference schedule must remain exactly 200 updates')
    if args.variant == 'layer_norm':
        plan['framework']['action_model']['decoder_input_norm'] = 'layer_norm'
    # Keep stage/eval indices, sampler, LR, seed, precision, losses and save policy.
    (root / 'plan.yaml').write_text(yaml.safe_dump(plan, sort_keys=False))
    write_json(root / 'trial_identity.json', dict(reference_run=str(args.reference_run), variant=args.variant,
        scheduled_updates=200, stop_after=50 if args.variant == 'reproduce' else 200,
        diagnostic_steps=[0, 1, 2, 5, 10, 20, 50, 200],
        script_sha256=sha256_file(Path(__file__)),
        diagnostic_sha256=sha256_file(Path(__file__).with_name('diagnose_qwenpi_action_decoder.py')),
        checkpoints_suppressed=args.variant == 'reproduce',
        initial_parameter_comparison='deterministic shared initialization; small probes recorded, not full-state equality'))
    from starVLA.training import train_umi_pretrain as training
    evaluator = FixedEvaluation(plan, root)
    holder, timeline = {}, []
    original_prepare = training.UMIAccelerator.prepare
    original_evaluate = training.evaluate
    original_commit = training.UMITrainingState.commit_update
    original_save = training.UMICheckpoints.save

    def record(step):
        model = holder['model']
        item = diagnose(model, evaluator.samples[0], 0, include_sampling=False)
        item['step'] = step
        timeline.append(item)
        write_json(root / 'decoder_timeline.json', timeline)
        print('DECODER '+json.dumps(dict(step=step, fm={mode:dict(loss=x['loss'], bias_loss=x['bias_only_loss'],
            h_rms=x['activations'][0]['decoder_input']['rms'],
            raw_rms=x['activations'][0]['raw_dit_output']['rms'],
            relu_fraction=x['activations'][0]['relu_output']['nonzero_fraction'],
            input_term_rms=x['activations'][0]['velocity_minus_bias']['rms']) for mode,x in item['fm'].items()})), flush=True)
        if args.variant == 'layer_norm' and step == 50:
            for mode, data in item['fm'].items():
                if (data['activations'][0]['velocity_minus_bias']['rms'] < 1e-6 or
                    data['gradients']['decoder_input_action_gradient']['status'] != 'nonzero'):
                    raise RuntimeError(f'Candidate still degenerate in {mode}; stop before extending to 200')

    def prepare(accelerator, *values, **kwargs):
        result = original_prepare(accelerator, *values, **kwargs)
        model = accelerator.unwrap_model(result[0])
        holder['model'] = model
        probes = {}
        for name, p in model.named_parameters():
            if name in ('action_model.action_decoder.layer1.weight', 'action_model.action_decoder.layer2.weight',
                        'action_model.state_encoder.layer1.weight', 'action_model.future_tokens.weight'):
                probes[name] = p.detach().flatten()[:64].cpu().tolist()
        write_json(root / 'initial_parameter_probes.json', probes)
        evaluator.measure(model, 0)
        record(0)
        return result

    def commit(state):
        result = original_commit(state)
        holder['step'] = state.global_update_step
        if state.global_update_step in (1, 2, 5, 10, 20, 50, 200):
            record(state.global_update_step)
        return result

    def evaluate(accelerator, model, *unused):
        return evaluator.measure(accelerator.unwrap_model(model), holder['step'])

    def no_save(checkpoints, *unused, **kwargs):
        # Only the disposable reproduction uses this. No pretend complete checkpoint.
        print('REPRODUCTION: checkpoint intentionally omitted; diagnostic run is not resumable', flush=True)

    training.UMIAccelerator.prepare = prepare
    training.UMITrainingState.commit_update = commit
    training.evaluate = evaluate
    if args.variant == 'reproduce':
        training.UMICheckpoints.save = no_save
    try:
        training.run(argparse.Namespace(plan=root/'plan.yaml', output_dir=root/'train', resume=None,
            stop_after_update=50 if args.variant == 'reproduce' else None, cpu=False))
    finally:
        training.UMIAccelerator.prepare = original_prepare
        training.UMITrainingState.commit_update = original_commit
        training.evaluate = original_evaluate
        training.UMICheckpoints.save = original_save
    first, last = evaluator.history[0], evaluator.history[-1]
    checks = dict(fixed_fm_loss_decreased=last['fixed_fm_loss'] < first['fixed_fm_loss'],
        generated_action_mse_decreased=last['normalized_action_mse'] < first['normalized_action_mse'],
        position_beats_hold=last['prediction']['position_rmse'] < last['hold_current']['position_rmse'],
        gripper_beats_hold=last['prediction']['gripper_rmse'] < last['hold_current']['gripper_rmse'],
        rotation_beats_hold=last['prediction']['rotation_mean_degrees'] < last['hold_current']['rotation_mean_degrees'])
    write_json(root/'summary.json', dict(status='completed', variant=args.variant, checks=checks,
        criteria_met=all(checks.values()), first=first, last=last, scheduled_updates=200,
        actual_updates=holder['step'], unchanged_reference_view=True))
    if args.variant == 'layer_norm':
        # Actual end-state noise/velocities and state response, without another weight load.
        terminal = diagnose(holder['model'], evaluator.samples[0], 0)
        write_json(root/'terminal_diagnostic.json', terminal)
    print('TRIAL '+json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
