"""Single-PPU, fixed-32-window learnability run using the production trainer.

Creates a tiny derived access view, keeps existing normalization and model math,
and observes fixed-seed FM loss/action generation at step 0 and every 50 steps.
The same windows are trained/evaluated deliberately; no generalization claim.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from starVLA.dataloader.umi_indexed_dataset import UMIIndexedDataset, compute_view_fingerprint
from starVLA.training.trainer_utils.umi_checkpoint import preserve_rng, sha256_file, write_json


def small_view(source, destination, count=32):
    """Select existing valid windows; no public data or full index is rewritten."""
    raw = UMIIndexedDataset(source)
    try:
        if len(raw) < count: raise ValueError('Engineering source has too few windows')
        ids = np.linspace(0, len(raw)-1, count, dtype=np.int64)
        assert len(set(ids.tolist())) == count
        rows = [(ep, start, start+1) for ep, start in map(raw.locate, ids)]
        destination.mkdir(parents=True, exist_ok=False)
        np.save(destination/'ranges.npy', np.asarray(rows, dtype=np.int64), allow_pickle=False)
        np.save(destination/'cumulative.npy', np.arange(1,count+1,dtype=np.int64), allow_pickle=False)
        shutil.copyfile(Path(source)/'metadata.sqlite3', destination/'metadata.sqlite3')
        meta = deepcopy(raw.meta)
        meta.update(total_ranges=count,total_windows=count,total_episodes=len({x[0] for x in rows}),
                    trainable_episodes=len({x[0] for x in rows}))
        for name in ('ranges.npy','cumulative.npy','metadata.sqlite3'):
            p = destination/name
            meta['artifacts'][name] = dict(size_bytes=p.stat().st_size, sha256=sha256_file(p))
        meta['engineering_selection'] = dict(parent_index=str(source), parent_view=raw.view_fingerprint,
                                             indices=ids.tolist(), purpose='fixed_window_overfit_only')
        meta['view_fingerprint'] = compute_view_fingerprint(meta)
        write_json(destination/'meta.json', meta)
        return meta['engineering_selection']
    finally:
        raw.close()


def errors(pred, truth):
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


class FixedEvaluation:
    def __init__(self, plan, root):
        self.plan,self.root = plan,root
        self.samples,self.normalizer = None,None
        self.history=[]

    def load_samples(self):
        from starVLA.training.trainer_utils.umi_training_data import make_loader
        loader=make_loader(self.plan,self.plan['evaluation'],self.root/'train',evaluation=True,record=False)
        try:
            self.samples=[loader.dataset[i] for i in range(len(loader.dataset))]
            self.normalizer=loader.dataset.normalizer
        finally: loader.dataset.close()

    def measure(self, model, step):
        if self.samples is None: self.load_samples()
        was_training=model.training
        predictions,losses=[],[]
        from starVLA.training.train_umi_pretrain import seed_start
        try:
            with preserve_rng(), torch.no_grad():
                model.eval()
                for i,sample in enumerate(self.samples):
                    seed_start(9876+i)
                    losses.append(float(model(examples=[sample])['action_loss']))
                    seed_start(19876+i)
                    inputs={k:v for k,v in sample.items() if k!='action'}
                    predictions.append(model.predict_action(examples=[inputs])['normalized_actions'][0])
        finally: model.train(was_training)
        pred=np.asarray(predictions,dtype=np.float32)
        truth=np.stack([s['action'] for s in self.samples])
        if not (np.isfinite(pred).all() and np.isfinite(losses).all()):
            raise ValueError('Nonfinite fixed evaluation')
        raw_pred=self.normalizer.inverse_action(pred).astype(np.float64)
        raw_true=self.normalizer.inverse_action(truth).astype(np.float64)
        raw_state=self.normalizer.inverse_state(np.stack([s['state'] for s in self.samples]))
        hold=np.repeat(raw_state,truth.shape[1],axis=1).astype(np.float64)
        result=dict(step=step, fixed_fm_loss=float(np.mean(losses)),
                    normalized_action_mse=float(np.mean((pred.astype(np.float64)-truth)**2)),
                    prediction=errors(raw_pred,raw_true), hold_current=errors(hold,raw_true),
                    evaluation_windows=len(self.samples), training_windows_overlap=True)
        np.savez_compressed(self.root/f'predictions_{step:06d}.npz', prediction=raw_pred,
                            target=raw_true, hold_current=hold,normalized_prediction=pred)
        self.history.append(result)
        write_json(self.root/'fixed_evaluations.json',self.history)
        print('FIXED_EVAL '+json.dumps(result),flush=True)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=200,choices=(200,300,500))
    args=parser.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('This bounded learnability run is single-process only')
    root=args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    plan=yaml.safe_load((REPO/'examples/umi_pretrain/train_files/umi_training_qwenpi_engineering.yaml').read_text())
    selection=small_view(plan['stages'][0]['index_dir'],root/'access')
    plan['stages']=[dict(name='fixed_32',index_dir=str(root/'access'),updates=args.updates)]
    plan['evaluation']=dict(name='training_windows_fixed_seed',index_dir=str(root/'access'),seed=9876)
    plan['training'].update(batch_size=1,gradient_accumulation_steps=2,warmup_updates=10,
                             eval_every=50,trace_samples=True)
    plan['checkpoint']=dict(integrity='basic',every_updates=args.updates)
    plan['performance']=dict(enabled=True,warmup_updates=2,detail_updates=0)
    (root/'plan.yaml').write_text(yaml.safe_dump(plan,sort_keys=False))
    write_json(root/'selection.json',selection)
    write_json(root/'tool_identity.json',dict(sha256=sha256_file(Path(__file__)),
        initialization='fresh Qwen3-VL-2B-Instruct plus freshly initialized action head',
        eval_schedule=[0]+list(range(50,args.updates+1,50))))
    from starVLA.training import train_umi_pretrain as training
    evaluator=FixedEvaluation(plan,root)
    original_prepare=training.UMIAccelerator.prepare
    original_evaluate=training.evaluate
    def prepare_then_baseline(accelerator,*values,**kwargs):
        result=original_prepare(accelerator,*values,**kwargs)
        evaluator.measure(accelerator.unwrap_model(result[0]),0)
        return result
    def evaluate_all(accelerator,model,*unused):
        return evaluator.measure(accelerator.unwrap_model(model),len(evaluator.history)*50)
    training.UMIAccelerator.prepare=prepare_then_baseline
    training.evaluate=evaluate_all
    try:
        training.run(argparse.Namespace(plan=root/'plan.yaml',output_dir=root/'train',resume=None,
                                        stop_after_update=None,cpu=False))
    finally:
        training.UMIAccelerator.prepare=original_prepare
        training.evaluate=original_evaluate
    first,last=evaluator.history[0],evaluator.history[-1]
    checks=dict(fixed_fm_loss_decreased=last['fixed_fm_loss']<first['fixed_fm_loss'],
        generated_action_mse_decreased=last['normalized_action_mse']<first['normalized_action_mse'],
        position_beats_hold=last['prediction']['position_rmse']<last['hold_current']['position_rmse'],
        gripper_beats_hold=last['prediction']['gripper_rmse']<last['hold_current']['gripper_rmse'])
    write_json(root/'summary.json',dict(status='completed',updates=args.updates,windows=32,checks=checks,
        criteria_met=all(checks.values()),first=first,last=last,
        scope='Fixed training-window learnability only; not generalization or policy success.'))
    print('LEARNABILITY '+json.dumps(checks),flush=True)


if __name__=='__main__':
    main()
