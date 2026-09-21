"""Existing pilot action metrics, used as periodic formal-training monitoring.

The fixed validation view is divided among ranks. No thresholds or early-stop
decisions are introduced. This evaluates offline errors, not robot success.
"""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from .umi_checkpoint import preserve_rng, require_all, sha256_file, write_json


def errors(pred, truth):
    # Same definition as check_qwenpi_learnability.errors / PilotEvaluation.
    delta = pred-truth
    positions = [0,1,2,8,9,10]
    grips = [7,15]
    angles, invalid = [], 0
    for start in (3,11):
        a,b = pred[...,start:start+4], truth[...,start:start+4]
        an,bn = np.linalg.norm(a,axis=-1), np.linalg.norm(b,axis=-1)
        valid = (an > 1e-8) & (bn > 1e-8)
        invalid += int((~valid).sum())
        dot = np.sum(a[valid]*b[valid],axis=-1)/(an[valid]*bn[valid])
        angles.extend(np.degrees(2*np.arccos(np.clip(np.abs(dot),0,1))).tolist())
    return dict(position_rmse=float(np.sqrt(np.mean(delta[...,positions]**2))),
                gripper_rmse=float(np.sqrt(np.mean(delta[...,grips]**2))),
                rotation_mean_degrees=float(np.mean(angles)) if angles else None,
                invalid_quaternions=invalid)


def evaluate_actions(accelerator, model, plan, run_dir, step):
    from .umi_training_data import make_loader
    start = time.monotonic()
    loader, failure, local = None, None, []
    was_training = model.training
    with preserve_rng():
        try:
            loader = make_loader(plan, plan['evaluation'], run_dir, evaluation=True, record=False)
            dataset = loader.dataset
            labels_path = Path(plan['evaluation']['labels'])
            if sha256_file(labels_path) != plan['evaluation']['labels_sha256']:
                raise ValueError('Fixed monitoring labels changed')
            labels = json.loads(labels_path.read_text())
            tasks = {int(r['episode_index']):r['task_class'] for r in labels}
            if len(dataset) != int(plan['evaluation']['windows']):
                raise ValueError('Fixed monitoring view size changed')
            model.eval()
            unwrapped = accelerator.unwrap_model(model)
            with torch.no_grad():
                for index in range(accelerator.process_index,len(dataset),accelerator.num_processes):
                    sample = dataset[index]
                    seed = int(plan['evaluation']['seed']) + index
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    inputs = {k:v for k,v in sample.items() if k!='action'}
                    predicted = np.asarray(unwrapped.predict_action(examples=[inputs])['normalized_actions'][0],dtype=np.float32)
                    target = np.asarray(sample['action'],dtype=np.float32)
                    if predicted.shape != target.shape or not np.isfinite(predicted).all():
                        raise ValueError(f'Invalid prediction at monitoring index {index}')
                    raw_pred = dataset.normalizer.inverse_action(predicted).astype(np.float64)
                    raw_true = dataset.normalizer.inverse_action(target).astype(np.float64)
                    state = dataset.normalizer.inverse_state(np.asarray(sample['state']))
                    hold = np.repeat(state,target.shape[0],axis=0).astype(np.float64)
                    ep = int(sample['umi_metadata']['episode_index'])
                    local.append((index,ep,tasks[ep],raw_pred,raw_true,hold))
        except Exception as error:
            failure = str(error)
        finally:
            model.train(was_training)
            if loader is not None:
                loader.dataset.close()
        # Every rank reaches the same error check before any gather.
        require_all(accelerator, failure is None, f'Action monitoring failed: {failure}')
        if dist.is_initialized():
            gathered = [None]*accelerator.num_processes
            dist.all_gather_object(gathered, local)
            rows = [r for part in gathered for r in part]
        else:
            rows = local
    rows.sort(key=lambda r:r[0])
    if [r[0] for r in rows] != list(range(plan['evaluation']['windows'])):
        raise ValueError('Monitoring ranks did not cover each fixed window exactly once')
    pred, truth, hold = (np.stack([r[j] for r in rows]) for j in (3,4,5))
    groups = np.asarray([r[2] for r in rows])
    per_task = {g:dict(windows=int((groups==g).sum()), prediction=errors(pred[groups==g],truth[groups==g]),
                      hold_current=errors(hold[groups==g],truth[groups==g])) for g in sorted(set(groups))}
    macro = {}
    for role in ('prediction','hold_current'):
        macro[role] = {}
        for metric in ('position_rmse','gripper_rmse','rotation_mean_degrees'):
            values = [v[role][metric] for v in per_task.values()]
            macro[role][metric] = float(np.mean(values)) if all(v is not None for v in values) else None
    result = dict(step=step, windows=len(rows), episodes=len({r[1] for r in rows}),
                  pooled=dict(prediction=errors(pred,truth),hold_current=errors(hold,truth)),
                  macro_task_mean=macro, per_task=per_task, seconds=time.monotonic()-start,
                  scope='Fixed task-coverage development validation; offline action errors, not task success or natural-pool average.')
    if accelerator.is_main_process:
        root=Path(run_dir)/'action_evaluation'; root.mkdir(exist_ok=True)
        write_json(root/f'update_{step:08d}.json',result)
        print('ACTION_EVALUATION '+json.dumps({k:result[k] for k in ('step','windows','pooled','seconds')}),flush=True)
    return {k:v for k,v in result.items() if k!='per_task'}
