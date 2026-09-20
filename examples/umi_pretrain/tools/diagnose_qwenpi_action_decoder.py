"""Observe action-token activations, actual sampling noise, and unclipped gradients.

Loads model weights only. Never creates an optimizer or updates parameters.
The observer is also used at selected steps of the bounded reproduction trial.
"""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from starVLA.training.trainer_utils.umi_checkpoint import preserve_rng, write_json


def stats(tensor):
    if tensor is None:
        return dict(status='none')
    x = tensor.detach().float()
    finite = bool(torch.isfinite(x).all())
    return dict(status='nonfinite' if not finite else 'zero' if not bool(x.any()) else 'nonzero',
                shape=list(x.shape), rms=float(x.square().mean().sqrt()),
                mean=float(x.mean()), std=float(x.std(unbiased=False)),
                minimum=float(x.min()), maximum=float(x.max()),
                max_abs=float(x.abs().max()), nonzero_fraction=float((x != 0).float().mean()))


class DecoderProbe:
    """Read-only hooks. Statistics exclude state/future tokens throughout."""
    def __init__(self, model):
        self.model = model
        self.head = model.action_model
        self.horizon = self.head.action_horizon
        self.handles = []
        self.reset()
        self.handles.append(self.head.model.register_forward_hook(self.dit_output))
        self.handles.append(self.head.action_decoder.layer1.register_forward_pre_hook(self.decoder_input))
        self.handles.append(self.head.action_decoder.layer1.register_forward_hook(self.preactivation))
        self.handles.append(self.head.action_decoder.layer2.register_forward_pre_hook(self.relu_output))
        self.handles.append(self.head.action_decoder.register_forward_hook(self.velocity))
        self.handles.append(self.head.action_encoder.register_forward_pre_hook(self.noisy_input))
        for index, block in enumerate(self.head.model.transformer_blocks):
            self.handles.append(block.register_forward_hook(self.block_hook(index)))

    def reset(self):
        self.records, self.blocks, self.noises = [], {}, []
        self.current, self.last_velocity, self.hidden = {}, None, None

    def action(self, x):
        return x[:, -self.horizon:]

    def block_hook(self, index):
        def hook(module, args, output):
            self.blocks[str(index)] = stats(self.action(output))['rms']
        return hook

    def dit_output(self, module, args, output):
        self.current['raw_dit_output'] = stats(self.action(output))
        self.current['block_output_rms'] = dict(self.blocks)

    def decoder_input(self, module, args):
        self.hidden = args[0]
        if self.hidden.requires_grad:
            self.hidden.retain_grad()
        self.current['decoder_input'] = stats(self.action(self.hidden))

    def preactivation(self, module, args, output):
        z = self.action(output)
        self.current['preactivation'] = stats(z)
        positive = (z > 0).float()
        self.current['preactivation']['positive_fraction'] = float(positive.mean())
        self.current['preactivation']['per_channel_positive_fraction'] = positive.mean((0, 1)).cpu().tolist()

    def relu_output(self, module, args):
        self.current['relu_output'] = stats(self.action(args[0]))

    def noisy_input(self, module, args):
        self.current['noisy_action_rms'] = stats(args[0])['rms']
        self.current['discrete_time'] = args[1].detach().cpu().tolist()

    def velocity(self, module, args, output):
        v = self.action(output)
        self.last_velocity = v.detach().clone()
        self.current['velocity'] = stats(v)
        self.current['velocity_minus_bias'] = stats(v - self.head.action_decoder.layer2.bias)
        self.records.append(self.current)
        self.current = {}

    @contextmanager
    def actual_noise(self):
        original = torch.randn
        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            if result.ndim == 3 and tuple(result.shape[-2:]) == (self.horizon, self.head.action_dim):
                self.noises.append(result.detach().clone())
            return result
        # Observe and return the original result; do not resample or replace RNG.
        with patch('torch.randn', observe):
            yield

    def gradient_report(self):
        names = dict(self.model.named_parameters())
        selected = [name for name in names if name.startswith('action_model.action_decoder.')]
        selected += [name for name in names if name.startswith('action_model.state_encoder.')]
        last = len(self.head.model.transformer_blocks)
        for layer in (last - 2, last - 1):
            prefix = f'action_model.model.transformer_blocks.{layer}.'
            selected += [name for name in names if name.startswith(prefix) and
                         (name.endswith('attn1.to_q.weight') or name.endswith('ff.net.2.weight'))]
        for layer in (0, last - 2):
            matches = [name for name in names if name.startswith('qwen_vl_interface.') and
                       f'.layers.{layer}.' in name and name.endswith('self_attn.v_proj.weight')]
            selected += matches
        report = {name: stats(names[name].grad) for name in selected}
        report['decoder_input_action_gradient'] = stats(
            self.action(self.hidden.grad) if self.hidden is not None and self.hidden.grad is not None else None)
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.reset()


def diagnose(model, sample, sample_index=0, include_sampling=True, modes=('eval', 'train')):
    """No optimizer updates; restore RNG/mode and release all diagnostic grads."""
    from starVLA.training.train_umi_pretrain import seed_start
    was_training = model.training
    probe = DecoderProbe(model)
    result = dict(sample_index=sample_index, action_positions_only=True,
                  parameter_updates=False, fm={})
    bias = model.action_model.action_decoder.layer2.bias
    result['decoder_layer2_weight'] = stats(model.action_model.action_decoder.layer2.weight)
    result['decoder_layer2_bias'] = bias.detach().cpu().tolist()
    try:
        with preserve_rng():
            for mode in modes:
                model.train(mode == 'train')
                model.zero_grad(set_to_none=True)
                probe.reset()
                seed_start(9876 + sample_index)
                with torch.enable_grad(), probe.actual_noise():
                    loss = model(examples=[sample])['action_loss']
                    if len(probe.noises) != 1:
                        raise ValueError('Expected one actual training noise draw')
                    noise = probe.noises[0]
                    target = torch.as_tensor(sample['action'], device=noise.device, dtype=noise.dtype)[None]
                    target = target.expand_as(noise) - noise
                    model_recomputed = ((probe.last_velocity - target) ** 2).mean()
                    bias_loss = ((bias.detach() - target) ** 2).mean()
                    if not torch.allclose(loss.detach(), model_recomputed, rtol=1e-5, atol=1e-6):
                        raise ValueError('Captured noise/velocity do not reconstruct actual FM loss')
                    loss.backward()
                result['fm'][mode] = dict(loss=float(loss.detach()), bias_only_loss=float(bias_loss),
                    activations=probe.records, gradients=probe.gradient_report())
                del loss, target, noise, model_recomputed, bias_loss
            model.zero_grad(set_to_none=True)
            probe.reset()
            if include_sampling:
                model.eval()
                seed_start(19876 + sample_index)
                inputs = {k:v for k,v in sample.items() if k != 'action'}
                with probe.actual_noise():
                    prediction = model.predict_action(examples=[inputs])['normalized_actions']
                if len(probe.noises) != 1:
                    raise ValueError('Expected one actual sampling noise draw')
                noise = probe.noises[0].cpu().numpy()
                result['sampling'] = dict(steps=probe.records, actual_initial_noise_captured=True,
                    noise_plus_bias_max_error=float(np.max(np.abs(prediction-noise-bias.detach().cpu().numpy()))))
                # Fixed actual noise/time; intervention is one normalized state coordinate.
                changed = dict(sample, state=np.array(sample['state'], copy=True))
                changed['state'][0, 0] += 0.5
                velocities, noises, times = [], [], []
                for item in (sample, changed):
                    probe.reset()
                    seed_start(9876 + sample_index)
                    with torch.no_grad(), probe.actual_noise():
                        model(examples=[item])
                    velocities.append(probe.last_velocity.cpu())
                    noises.append(probe.noises[0].cpu())
                    times.append(probe.records[0]['discrete_time'])
                assert torch.equal(noises[0], noises[1]) and times[0] == times[1]
                result['state_intervention'] = dict(normalized_coordinate=0, delta=0.5,
                    same_noise_and_time=True, velocity_max_change=float((velocities[1]-velocities[0]).abs().max()))
    finally:
        model.zero_grad(set_to_none=True)
        probe.close()
        model.train(was_training)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 16, 31])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new diagnostic output; never overwrite evidence')
    from starVLA.training.train_umi_pretrain import build_model, seed_start
    from check_qwenpi_learnability import FixedEvaluation
    plan = yaml.safe_load((args.run_dir / 'plan.yaml').read_text())
    seed_start(plan['training']['seed'])
    model = build_model(plan)
    checkpoint = args.run_dir / f'train/checkpoints/update_{args.step:08d}/pytorch_model.bin'
    weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    gc.collect()
    model.to('cuda')
    evaluator = FixedEvaluation(plan, args.run_dir)
    evaluator.load_samples()
    report = dict(checkpoint=str(checkpoint), optimizer_loaded=False, parameter_updates=False, samples=[])
    for i in args.indices:
        report['samples'].append(diagnose(model, evaluator.samples[i], i))
        write_json(args.output, report)
        brief = report['samples'][-1]
        print(json.dumps(dict(index=i, fm={mode:dict(loss=x['loss'], bias_loss=x['bias_only_loss'],
            relu=x['activations'][0]['relu_output']['nonzero_fraction']) for mode,x in brief['fm'].items()},
            sampling_error=brief['sampling']['noise_plus_bias_max_error'],
            state=brief['state_intervention'])), flush=True)


if __name__ == '__main__':
    main()
