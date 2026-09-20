"""Single-PPU multi-episode pilot using the existing full-state trainer.

Fixed coverage-weighted curves and final full validation are separate reports.
Evaluation preserves RNG and model mode. No soft-metric early stopping.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.training import train_umi_pretrain as training
from starVLA.training.trainer_utils.umi_checkpoint import preserve_rng, sha256_file, write_json
from starVLA.training.trainer_utils.umi_training_data import make_loader
from check_qwenpi_learnability import errors


class PilotEvaluation:
    def __init__(self, plan, run_dir):
        self.plan, self.root = plan, run_dir/'pilot_evaluation'
        self.root.mkdir(parents=True, exist_ok=True)
        self.views = {'train': plan['stages'][0], 'validation': plan['evaluation']}
        manifest = json.loads(Path(plan['pilot']['manifest']).read_text())
        for split, view in self.views.items():
            path = Path(view['index_dir'])
            meta = json.loads((path/'meta.json').read_text())
            if (meta['view_fingerprint'] != manifest['views'][split]['view_fingerprint']
                    or sha256_file(path/'episodes.json') != meta['episode_list_sha256']):
                raise ValueError(f'Pilot {split} labels or view identity changed')
        self.rows = {split: json.loads((Path(view['index_dir'])/'episodes.json').read_text())
                     for split, view in self.views.items()}
        self.episode_maps = {split: {r['episode_index']: r for r in rows} for split, rows in self.rows.items()}
        self.indices = {}
        for split, rows in self.rows.items():
            ids, offset, group_counts = [], 0, {}
            for row in sorted(rows, key=lambda r:r['episode_index']):
                group = row['group_id']
                keep = split=='validation' or group_counts.get(group,0) < plan['pilot']['curve_train_episodes_per_task']
                if keep:
                    n = len(row['anchors'])
                    ids.extend((offset+np.linspace(0,n-1,min(n,plan['pilot']['curve_windows_per_episode']),dtype=np.int64)).tolist())
                    group_counts[group] = group_counts.get(group,0)+1
                offset += len(row['anchors'])
            self.indices[split] = ids
        write_json(self.root/'fixed_curve_indices.json', self.indices)

    def measure(self, model, step, split, full=False):
        start = time.monotonic()
        loader = make_loader(self.plan,self.views[split],self.root,evaluation=True,record=False)
        dataset = loader.dataset
        indices = list(range(len(dataset))) if full else self.indices[split]
        label = f'{split}_{"full" if full else "curve"}'
        was_training = model.training
        predictions, targets, holds, groups, identities, losses = [],[],[],[],[],[]
        try:
            with preserve_rng(), torch.no_grad():
                model.eval()
                for i, index in enumerate(indices):
                    sample = dataset[index]
                    meta = sample['umi_metadata']
                    # Seeds depend on fixed view indices, not call order or workers.
                    seed = int(self.plan['evaluation']['seed']) + (0 if split=='train' else 1000000) + index
                    training.seed_start(seed)
                    loss = float(model(examples=[sample])['action_loss'])
                    training.seed_start(seed+2000000)
                    inputs = {k:v for k,v in sample.items() if k!='action'}
                    pred = np.asarray(model.predict_action(examples=[inputs])['normalized_actions'][0],dtype=np.float32)
                    truth = np.asarray(sample['action'],dtype=np.float32)
                    if not (np.isfinite(pred).all() and np.isfinite(loss)):
                        raise ValueError(f'Nonfinite evaluation at {label} index {index}')
                    raw_pred = dataset.normalizer.inverse_action(pred).astype(np.float64)
                    raw_true = dataset.normalizer.inverse_action(truth).astype(np.float64)
                    state = dataset.normalizer.inverse_state(np.asarray(sample['state']))
                    hold = np.repeat(state, truth.shape[0],axis=0).astype(np.float64)
                    predictions.append(raw_pred); targets.append(raw_true); holds.append(hold)
                    losses.append(loss)
                    groups.append(self.episode_maps[split][meta['episode_index']]['group_id'])
                    identities.append([int(index),int(meta['episode_index']),int(meta['frame_index'])])
                    if (i+1)%60==0 or i+1==len(indices):
                        print(f'PILOT_EVAL step={step} split={label} windows={i+1}/{len(indices)}',flush=True)
        finally:
            model.train(was_training)
            dataset.close()
        pred, truth, hold = map(np.asarray,(predictions,targets,holds))
        group_array = np.asarray(groups)
        per_task = {}
        for group in sorted(set(groups)):
            mask = group_array==group
            per_task[group] = dict(windows=int(mask.sum()), prediction=errors(pred[mask],truth[mask]),
                                  hold_current=errors(hold[mask],truth[mask]))
        macro = {}
        for role in ('prediction','hold_current'):
            macro[role] = {}
            for metric in ('position_rmse','gripper_rmse','rotation_mean_degrees'):
                values = [r[role][metric] for r in per_task.values()]
                macro[role][metric] = float(np.mean(values)) if all(v is not None for v in values) else None
        result = dict(step=step,split=label,windows=len(indices),episodes=len({r[1] for r in identities}),
            task_categories=len(per_task),fixed_fm_loss=float(np.mean(losses)),
            pooled=dict(prediction=errors(pred,truth),hold_current=errors(hold,truth)),
            macro_task_mean=macro,per_task=per_task,seconds=time.monotonic()-start,
            training_windows_overlap=split=='train',
            scope='Directory-task coverage evaluation; offline action errors, not natural-distribution performance or robot task success.',
            macro_definition='Unweighted arithmetic mean of per-task RMSE/angle metrics, not pooled RMSE.')
        prefix = self.root/f'{step:06d}_{label}'
        np.savez_compressed(str(prefix)+'.npz',prediction=pred,target=truth,hold_current=hold,
                            identities=np.asarray(identities,dtype=np.int64),group_ids=group_array)
        write_json(Path(str(prefix)+'.json'),result)
        print('PILOT_METRICS '+json.dumps({k:result[k] for k in ('step','split','windows','pooled','macro_task_mean','seconds')}),flush=True)
        return {k:v for k,v in result.items() if k!='per_task'}

    def evaluate(self, accelerator, model, plan, run_dir, step):
        if step not in plan['pilot']['evaluate_updates']:
            return {'step':step,'scheduled':False}
        unwrapped = accelerator.unwrap_model(model)
        result = {split:self.measure(unwrapped,step,split) for split in ('train','validation')}
        if step == plan['pilot']['full_validation_at']:
            result['validation_full'] = self.measure(unwrapped,step,'validation',full=True)
        write_json(self.root/f'{step:06d}_summary.json',result)
        return result


def main():
    args = training.arguments()
    if args.cpu or int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('This pilot recipe uses one PPU process')
    plan = OmegaConf.to_container(OmegaConf.load(args.plan),resolve=True)
    if sha256_file(Path(__file__)) != plan['pilot']['runner_sha256']:
        raise ValueError('Pilot evaluator differs from the frozen plan')
    if sha256_file(Path(plan['pilot']['manifest'])) != plan['pilot']['manifest_sha256']:
        raise ValueError('Pilot selection manifest changed')
    evaluator = None
    def start(accelerator,model,actual_plan,run_dir,step):
        nonlocal evaluator
        evaluator = PilotEvaluation(actual_plan,run_dir)
        # On resume, evaluate only future scheduled points. Initialization must
        # never overwrite an already measured baseline with trained weights.
        if not args.resume:
            evaluator.evaluate(accelerator,model,actual_plan,run_dir,step)
    def evaluate(*values):
        return evaluator.evaluate(*values)
    training.run(args,on_start=start,evaluation_fn=evaluate)


if __name__=='__main__':
    main()
