"""Bounded no-update condition intervention, using the actual sampling path.

Only the first velocity is evaluated: hooks replace the real action encoder input
and both timestep inputs, then stop after the decoder. No copied DiT equations.
Features are cached once per window. Hooks, RNG and mode are restored on exit.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from starVLA.training.trainer_utils.umi_checkpoint import preserve_rng, write_json
from diagnose_qwenpi_action_decoder import stats


class FirstVelocityReady(Exception):
    pass


def rms(x):
    return float(x.float().square().mean().sqrt())


def difference(x, baseline):
    delta = x.float() - baseline.float()
    return dict(rms=rms(delta), max_abs=float(delta.abs().max()),
                relative_rms=rms(delta)/(rms(baseline)+1e-12))


def probe_velocity(head, features, mask, noisy, state, time, policy='normal', details=True):
    """Return the actual first decoder velocity and compact residual probes."""
    hooks, tensors, report = [], {}, {}
    if policy not in ('normal','layer_norm','zero_cross'):
        raise ValueError(f'Unknown diagnostic policy: {policy}')
    if not head.model.config.use_canonical_forward:
        raise ValueError('This diagnostic requires the current canonical state path')
    horizon = head.action_horizon
    steps = torch.full((noisy.shape[0],), int(time*head.num_timestep_buckets),
                       device=noisy.device, dtype=torch.long)
    def action_input(module, args):
        tensors['actual_noisy_input'] = noisy.detach().cpu().clone()
        return noisy, steps
    def dit_input(module, args, kwargs):
        kwargs = dict(kwargs, timestep=steps)
        return args, kwargs
    def finish(module, args, output):
        tensors['velocity'] = output[:, -horizon:].detach().float().cpu().clone()
        raise FirstVelocityReady()
    hooks.append(head.action_encoder.register_forward_pre_hook(action_input))
    hooks.append(head.model.register_forward_pre_hook(dit_input, with_kwargs=True))
    hooks.append(head.action_decoder.register_forward_hook(finish))
    selected = ({1, 2, 3, len(head.model.transformer_blocks)-1}
                & set(range(len(head.model.transformer_blocks)))) if details else set()
    def capture(name, x):
        x = x[:, -horizon:].detach().float()
        tensors[name] = x.cpu().clone()
        report[name] = stats(x)
    def valid_stats(x):
        x = x.detach().float()
        if mask is not None:
            x = x[mask.bool()]
        else:
            x = x.reshape(-1, x.shape[-1])
        result = stats(x)
        result['token_centered_std_mean'] = float(x.std(dim=-1, unbiased=False).mean())
        result['valid_tokens'] = x.shape[0]
        return result
    for index, block in enumerate(head.model.transformer_blocks):
        is_cross = not (head.model.config.interleave_self_attention and index % 2 == 1)
        if is_cross and policy == 'layer_norm':
            def normalize(module, args, kwargs):
                c = kwargs['encoder_hidden_states']
                return args, dict(kwargs, encoder_hidden_states=F.layer_norm(c, (c.shape[-1],), eps=1e-5))
            hooks.append(block.attn1.register_forward_pre_hook(normalize, with_kwargs=True))
        residual_branch = block.final_dropout if block.final_dropout is not None else block.attn1
        if is_cross and policy == 'zero_cross':
            hooks.append(residual_branch.register_forward_hook(lambda m,a,o: torch.zeros_like(o)))
        if index not in selected:
            continue
        prefix = f'block_{index}'
        def block_input(module, args, kwargs, prefix=prefix):
            capture(prefix+'/input', kwargs.get('hidden_states', args[0] if args else None))
        hooks.append(block.register_forward_pre_hook(block_input, with_kwargs=True))
        hooks.append(residual_branch.register_forward_hook(
            lambda m,a,o,p=prefix: capture(p+'/attention_residual', o)))
        hooks.append(block.norm3.register_forward_pre_hook(
            lambda m,a,p=prefix: capture(p+'/after_attention', a[0])))
        hooks.append(block.ff.register_forward_hook(lambda m,a,o,p=prefix: capture(p+'/ffn_residual', o)))
        hooks.append(block.register_forward_hook(lambda m,a,o,p=prefix: capture(p+'/output', o)))
        if is_cross:
            def condition(module, args, kwargs, prefix=prefix):
                report[prefix+'/condition_actual'] = valid_stats(kwargs['encoder_hidden_states'])
            hooks.append(block.attn1.register_forward_pre_hook(condition, with_kwargs=True))
            for name in ('to_k', 'to_v'):
                hooks.append(getattr(block.attn1, name).register_forward_hook(
                    lambda m,a,o,p=prefix,n=name: report.__setitem__(p+'/'+n, valid_stats(o))))
            report[prefix+'/condition_raw'] = valid_stats(features[index])
    try:
        with torch.no_grad():
            try:
                head.predict_action(features, state, encoder_attention_mask=mask)
            except FirstVelocityReady:
                pass
        if 'velocity' not in tensors:
            raise RuntimeError('Actual decoder path was not reached')
        assert torch.equal(tensors.pop('actual_noisy_input'), noisy.detach().cpu())
        for index in selected:
            p = f'block_{index}'
            report[p+'/injection_ratio'] = report[p+'/attention_residual']['rms'] / (report[p+'/input']['rms']+1e-12)
        report['velocity'] = stats(tensors['velocity'])
        return tensors, report
    finally:
        for hook in hooks:
            hook.remove()


def diagnose_condition(model, sample, index=0, policies=('normal','layer_norm','zero_cross'), times=(0.,.25,.75), details=True):
    from starVLA.training.train_umi_pretrain import seed_start
    previous = model.training
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            features, mask = model._encode_vl_hidden_states([sample['image']], [sample['lang']])
            if mask is not None: mask = mask.bool()
            device, dtype = features[0].device, features[0].dtype
            target = torch.as_tensor(np.asarray(sample['action'])[None], device=device, dtype=dtype)
            state = torch.as_tensor(np.asarray(sample['state'])[None], device=device, dtype=dtype)
            seed_start(29876+index)
            noise1, noise2 = torch.randn_like(target), torch.randn_like(target)
            rows=[]
            for policy in policies:
                for time in times:
                    noisy = (1-time)*noise1 + time*target
                    alternate = (1-time)*noise2 + time*target
                    base, measures = probe_velocity(model.action_model,features,mask,noisy,state,time,policy,details)
                    repeat, _ = probe_velocity(model.action_model,features,mask,noisy,state,time,policy,False)
                    row=dict(policy=policy,time=time,statistics=measures,
                             repeat_velocity=difference(repeat['velocity'],base['velocity']),interventions={})
                    velocity_target=(target-noise1).cpu()
                    row['fixed_velocity_mse']=float((base['velocity']-velocity_target).square().mean())
                    bias=model.action_model.action_decoder.layer2.bias.detach().float().cpu()
                    row['bias_velocity_mse']=float((bias-velocity_target).square().mean())
                    interventions=[('noisy_action',alternate,state)]
                    for coord in (0,8,15):
                        modified=state.clone(); modified[...,coord]+=0.5
                        interventions.append((f'state_{coord}',noisy,modified))
                    for name, action_input, state_input in interventions:
                        changed,_=probe_velocity(model.action_model,features,mask,action_input,state_input,time,policy,details)
                        delta={key:difference(value,base[key]) for key,value in changed.items()}
                        if name=='noisy_action' and time==0:
                            actual=(changed['velocity']-base['velocity']).flatten()
                            ideal=-(noise2-noise1).cpu().flatten()
                            delta['noise_endpoint_reference']=dict(
                                cosine=float(F.cosine_similarity(actual[None],ideal[None],eps=1e-12)),
                                rms_gain=rms(actual)/rms(ideal),
                                directional_gain=float(actual.dot(ideal)/ideal.square().sum()),
                                relative_error=rms(actual-ideal)/rms(ideal))
                        row['interventions'][name]=delta
                    rows.append(row)
            return dict(window_index=index,features_cached=True,parameter_updates=False,
                        actual_sampling_path=True,times=list(times),rows=rows)
    finally:
        model.train(previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    from starVLA.training.train_umi_pretrain import build_model,seed_start
    from check_qwenpi_learnability import FixedEvaluation
    plan=yaml.safe_load((args.reference_run/'plan.yaml').read_text())
    seed_start(plan['training']['seed'])
    model=build_model(plan).to('cuda')
    evaluator=FixedEvaluation(plan,args.reference_run); evaluator.load_samples()
    for phase in ('initial','decoder_ln_200'):
        if phase!='initial':
            checkpoint=args.reference_run/'train/checkpoints/update_00000200/pytorch_model.bin'
            weights=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
            model.load_state_dict(weights,strict=True)
            del weights
            gc.collect()
        for index in (0,16,31):
            report=diagnose_condition(model,evaluator.samples[index],index)
            write_json(args.output_dir/f'{phase}_{index:02d}.json',report)
            print(json.dumps(dict(phase=phase,index=index,noise_endpoint=[dict(policy=r['policy'],
                **r['interventions']['noisy_action']['noise_endpoint_reference']) for r in report['rows'] if r['time']==0])),flush=True)


if __name__=='__main__':
    main()
