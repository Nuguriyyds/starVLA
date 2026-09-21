"""Materialize a formal plan only after the total update budget is supplied.

Without --total-updates, write an explicitly non-runnable template. This does
not infer epochs from dataset hours or inherit the pilot's 10,000 updates.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from starVLA.training.trainer_utils.umi_training_state import validate_plan
from starVLA.training.trainer_utils.umi_checkpoint import sha256_file


def stage_budgets(total, weights):
    if total < len(weights) or any(w <= 0 for w in weights):
        raise ValueError('Budget must permit at least one update per nonempty stage')
    # Exact integer largest-remainder apportionment; total is never rounded away.
    denominator = sum(weights)
    budgets = [total*w//denominator for w in weights]
    order = sorted(range(len(weights)), key=lambda i:(-(total*weights[i] % denominator),i))
    for i in order[:total-sum(budgets)]:
        budgets[i] += 1
    if min(budgets) < 1:
        raise ValueError('Budget too small for proportional stage allocation')
    return budgets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-dir',type=Path,required=True)
    p.add_argument('--reference-plan',type=Path,required=True,help='Only reuse framework/data options; no weights or optimizer are loaded')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--total-updates',type=int)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--warmup-updates',type=int)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--eval-every',type=int,default=2000)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Plan output already exists; use a new filename to preserve frozen runs')
    root=args.preparation_dir.resolve()
    if not (root/'COMPLETED.json').is_file():
        raise ValueError('Formal data preparation is incomplete')
    report=json.loads((root/'formal_manifest.json').read_text())
    reference=yaml.safe_load(args.reference_plan.read_text())
    framework=deepcopy(reference['framework'])
    if args.model_dir:
        framework['qwenvl']['base_vlm']=str(args.model_dir.resolve())
    if (framework['action_model'].get('decoder_input_norm') != 'layer_norm'
        or framework['action_model']['diffusion_model_cfg'].get('cross_condition_norm') != 'layer_norm'):
        raise ValueError('Reference must be the fixed dual-LN model recipe')
    names=[f'stage_{i:02d}' for i in range(1,6)]
    weights=[report['views'][n]['windows'] for n in names]
    budgets=stage_budgets(args.total_updates,weights) if args.total_updates is not None else [None]*5
    global_batch=64*args.batch_size*args.accumulation
    data=deepcopy(reference['data'])
    data.update(normalization_statistics=str(root/'statistics.json'),
                normalization_experiment_contract=str(root/'normalization_experiment.json'),
                shuffle=True, shuffle_block_size=512)
    train=dict(seed=42,batch_size=args.batch_size,gradient_accumulation_steps=args.accumulation,
               global_batch_size=global_batch,num_workers=args.workers,prefetch_factor=2,
               mixed_precision='no',lr=reference['training']['lr'],vlm_lr=reference['training']['vlm_lr'],
               weight_decay=reference['training']['weight_decay'],max_grad_norm=reference['training']['max_grad_norm'],
               warmup_updates=args.warmup_updates if args.warmup_updates is not None else
                   (min(args.total_updates-1,max(1,args.total_updates//100)) if args.total_updates is not None else None),
               eval_every=args.eval_every,trace_samples=False)
    plan=dict(version='umi-training-plan-v1',purpose='formal',model_kind='qwenpi',framework=framework,data=data,
              training=train,checkpoint=dict(integrity='basic',every_updates=args.save_every),
              stages=[dict(name=n,index_dir=report['views'][n]['path'],updates=u) for n,u in zip(names,budgets)],
              evaluation=dict(name='fixed_development_validation',index_dir=report['views']['validation_monitor']['path'],
                              labels=str(root/'manifests/validation_monitor.json'),windows=report['views']['validation_monitor']['windows'],
                              labels_sha256=sha256_file(root/'manifests/validation_monitor.json'),
                              seed=19876,metrics='pilot_action_errors',at_end=True),
              performance=dict(enabled=False),
              deployment=dict(nodes=4,processes_per_node=16,world_size=64,backend='plain_ddp',
                              platform='Alibaba Cloud; actual addresses/ranks/mounts supplied at launch',
                              known_issue='PPU communication cleanup crash unresolved; platform handling pending'),
              formal=dict(manifest=str(root/'formal_manifest.json'),manifest_sha256=sha256_file(root/'formal_manifest.json'),
                          validation_pool=report['views']['validation']['path'],
                          initialization='Original Qwen perception weights plus newly initialized action head; no pilot resume',
                          budget_status='fixed' if args.total_updates is not None else 'REQUIRED: total update budget',
                          total_updates=args.total_updates,
                          total_window_exposures=args.total_updates*global_batch if args.total_updates else None,
                          stage_budget_policy='proportional to valid windows, integer largest remainder',
                          scheduler='one global linear warmup and linear decay; no stage reset',
                          sampling='natural window weights; bounded block shuffle changes order only'))
    if args.total_updates is not None:
        validate_plan(plan)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    header=('# Formal pretraining plan\n' if args.total_updates is not None else
            '# TEMPLATE ONLY: fill total budget using render_umi_formal_plan.py; null updates cannot run.\n')
    args.output.write_text(header+yaml.safe_dump(plan,sort_keys=False,allow_unicode=True),encoding='utf-8')
    print(json.dumps(dict(path=str(args.output),budget_status=plan['formal']['budget_status'],
                         global_batch_size=global_batch,stage_updates=budgets),ensure_ascii=False))


if __name__=='__main__':
    main()
