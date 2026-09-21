"""Freeze formal UMI groups and compact views from the existing complete index.

Metadata only: no video decoding, sensor-row audit or new validity policy.
The original pilot validation sources are reserved before five-way allocation.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from umi_split_allocation import allocate_training_stages
from starVLA.dataloader.umi_indexed_dataset import compute_view_fingerprint
from starVLA.dataloader.umi_normalization import (
    build_representation, parent_fingerprint, content_fingerprint, validate_experiment_contract,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text())


def manifest(path, rows):
    # One row per episode, never one row per overlapping action window.
    fields = list(rows[0])
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k,v in row.items()})


def make_view(parent, destination, selected, rows, meta, manifest_path, scope):
    destination.mkdir()
    if not len(selected) or np.any(selected[:,2] <= selected[:,1]):
        raise ValueError('Empty/invalid formal view')
    lengths = selected[:,2] - selected[:,1]
    cumulative = np.cumsum(lengths, dtype=np.int64)
    np.save(destination/'ranges.npy', selected, allow_pickle=False)
    np.save(destination/'cumulative.npy', cumulative, allow_pickle=False)
    # Read-only metadata shared across views. RANGES define membership; unused
    # rows in this lookup database never enter sampling or statistics fitting.
    try:
        os.link(parent/'metadata.sqlite3', destination/'metadata.sqlite3')
        storage = 'immutable parent metadata hard link'
    except OSError:
        shutil.copyfile(parent/'metadata.sqlite3', destination/'metadata.sqlite3')
        if sha(destination/'metadata.sqlite3') != meta['artifacts']['metadata.sqlite3']['sha256']:
            raise ValueError('Metadata copy changed')
        storage = 'immutable parent metadata copy'
    child = deepcopy(meta)
    child.pop('engineering_selection', None)
    child.update(total_windows=int(cumulative[-1]), total_ranges=len(selected),
                 total_episodes=len(rows), trainable_episodes=len(np.unique(selected[:,0])),
                 episode_list=str(manifest_path), episode_list_sha256=sha(manifest_path),
                 verified_full_index_counts=False, scope=scope)
    child['metadata_scope'] = {'membership': 'ranges.npy only', 'lookup_rows': 'full parent index', 'storage': storage}
    child['formal_selection'] = {'version': 'umi-formal-partition-v1',
                                 'parent_view': compute_view_fingerprint(meta),
                                 'manifest_sha256': child['episode_list_sha256']}
    for name in ('ranges.npy', 'cumulative.npy'):
        path = destination/name
        child['artifacts'][name] = dict(size_bytes=path.stat().st_size, sha256=sha(path))
    child['view_fingerprint'] = compute_view_fingerprint(child)
    write(destination/'meta.json', child)
    return dict(path=str(destination), episodes=len(rows), trainable_episodes=child['trainable_episodes'],
                windows=int(cumulative[-1]), ranges=len(selected), view_fingerprint=child['view_fingerprint'])


def summarize(rows):
    windows = sum(r['valid_windows'] for r in rows)
    axes = {}
    for axis in ('task_class', 'scene', 'task_category'):
        counts = defaultdict(lambda: dict(episodes=0, valid_windows=0, duration_seconds=0.0))
        for r in rows:
            item = counts[r[axis]]
            item['episodes'] += 1
            item['valid_windows'] += r['valid_windows']
            item['duration_seconds'] += r['duration_seconds']
        for item in counts.values():
            item['window_fraction'] = item['valid_windows']/windows if windows else 0.0
        axes[axis] = dict(sorted(counts.items()))
    return dict(episodes=len(rows), source_groups=len({r['source_recording_key'] for r in rows}),
                valid_windows=windows, recorded_episode_hours=sum(r['duration_seconds'] for r in rows)/3600,
                duration_basis='catalog episode duration; not summed overlapping-window durations', distributions=axes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('private-root', 'index-dir', 'inventory-dir', 'pilot-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root, parent, output = args.private_root.resolve(), args.index_dir.resolve(), args.output_dir.resolve()
    if root not in output.parents or output.exists() or parent == output:
        raise ValueError('Require a NEW output directory strictly inside the private root')
    meta = read(parent/'meta.json')
    if meta['status'] != 'completed' or meta.get('engineering_selection'):
        raise ValueError('Require the complete parent index')
    print('Verifying existing compact index identities (no sensor scan)', flush=True)
    for name, identity in meta['artifacts'].items():
        if (parent/name).stat().st_size != identity['size_bytes'] or sha(parent/name) != identity['sha256']:
            raise ValueError(f'Parent index changed: {name}')
    ranges = np.load(parent/'ranges.npy', mmap_mode='r')
    lengths = ranges[:,2]-ranges[:,1]
    if int(lengths.sum()) != meta['total_windows']:
        raise ValueError('Parent compact window count mismatch')
    ep_ids, first = np.unique(ranges[:,0], return_index=True)
    counts = dict(zip(ep_ids.tolist(), np.add.reduceat(lengths, first).tolist()))
    inventory_path = args.inventory_dir/'inventory.json'
    candidate_path = args.inventory_dir/'episode_candidates.csv.gz'
    inventory = read(inventory_path)
    labels = {g['group_key']: g for g in inventory['groups']}
    pilot_meta = read(args.pilot_dir/'validation/meta.json')
    pilot_rows_path = args.pilot_dir/'validation/episodes.json'
    if sha(pilot_rows_path) != pilot_meta['episode_list_sha256']:
        raise ValueError('Frozen pilot validation manifest changed')
    pilot_rows = read(pilot_rows_path)
    held_episodes = {int(r['episode_index']) for r in pilot_rows}
    if len(held_episodes) != 120:
        raise ValueError('Expected the existing 120 validation episodes')
    rows = []
    with gzip.open(candidate_path, 'rt', encoding='utf-8', newline='') as stream:
        for r in csv.DictReader(stream):
            ep = int(r['episode_index'])
            label = labels[r['candidate_group_key']]
            if not r['source_recording_key'] or int(r['valid_windows']) != counts.get(ep,0):
                raise ValueError(f'Missing source group or existing-index count mismatch: {ep}')
            rows.append(dict(episode_index=ep, source_recording_key=r['source_recording_key'],
                             task_class=label['group_id'], task_directory=label['group_key'],
                             scene=label['path_scene'], task_category=label['path_category'],
                             task_ids=json.loads(r['task_ids']), valid_windows=counts.get(ep,0),
                             duration_seconds=float(r['duration_seconds'])))
    rows.sort(key=lambda r:r['episode_index'])
    by_ep = {r['episode_index']:r for r in rows}
    if len(by_ep) != len(rows) or len(rows) != meta['total_episodes'] or set(counts)-set(by_ep):
        raise ValueError('Episode inventory incomplete or duplicated')
    held_groups = {by_ep[ep]['source_recording_key'] for ep in held_episodes}
    # Compare the recorded source keys too, not just numeric episode IDs.
    for r in pilot_rows:
        key = r.get('source_recording_key')
        if key and key != by_ep[r['episode_index']]['source_recording_key']:
            raise ValueError('Validation source identity changed')
    train = [r for r in rows if r['source_recording_key'] not in held_groups]
    validation = [r for r in rows if r['source_recording_key'] in held_groups]
    groups = {}
    for r in train:
        g = groups.setdefault(r['source_recording_key'], dict(id=r['source_recording_key'], windows=0,
                             tasks=Counter(), sources=Counter()))
        w = r['valid_windows']; g['windows'] += w
        g['tasks'][r['task_class']] += w; g['sources'][r['scene']] += w
    print(f'Allocating {len(groups)} training groups; reserving {len(held_groups)} validation groups', flush=True)
    assignment, allocation = allocate_training_stages(groups.values(), seed=args.seed)
    for r in rows:
        r['split'] = 'validation' if r['source_recording_key'] in held_groups else 'train'
        r['stage'] = 'validation' if r['split']=='validation' else assignment[r['source_recording_key']]
    partitions = {'train':train, 'validation':validation}
    partitions.update({f'stage_{i:02d}':[r for r in train if r['stage']==f'stage_{i:02d}'] for i in range(1,6)})
    output.mkdir(parents=True)
    (output/'manifests').mkdir(); (output/'views').mkdir()
    write(output/'PREPARING.json', dict(pid=os.getpid(), status='building', seed=args.seed))
    views, reports = {}, {}
    for name, subset in partitions.items():
        mpath = output/'manifests'/f'{name}.csv.gz'
        manifest(mpath, subset)
        selected = np.asarray(ranges[np.isin(ranges[:,0],[r['episode_index'] for r in subset])])
        views[name] = make_view(parent, output/'views'/name, selected, subset, meta, mpath,
                                'All existing valid windows in the frozen '+name+' source groups')
        reports[name] = summarize(subset)
        if views[name]['windows'] != reports[name]['valid_windows']:
            raise ValueError('Partition count mismatch')
        print(name, views[name], flush=True)
    # Fixed monitoring windows are inherited from the old validation selection.
    # This does not shrink the formal validation pool or redraw its episodes.
    eval_rows = []
    for r in pilot_rows:
        anchors = list(r['anchors'])
        positions = np.linspace(0,len(anchors)-1,min(4,len(anchors)),dtype=np.int64)
        eval_rows.append(dict(by_ep[r['episode_index']], anchors=[anchors[int(i)] for i in positions]))
    monitor = np.array(sorted((r['episode_index'], a, a+1) for r in eval_rows for a in r['anchors']),dtype=np.int64)
    # Validate membership using compact intervals, no expansion to all frames.
    for ep, a, _ in monitor:
        er = ranges[ranges[:,0]==ep]
        if not np.any((er[:,1]<=a)&(a<er[:,2])):
            raise ValueError('Monitoring anchor outside the unchanged validity index')
    eval_manifest = output/'manifests/validation_monitor.json'
    write(eval_manifest, eval_rows)
    views['validation_monitor'] = make_view(parent, output/'views/validation_monitor',monitor,eval_rows,meta,
                                            eval_manifest,'Fixed 4-window-per-episode development monitoring; task coverage weighted')
    train_summary = reports['train']
    for name in [f'stage_{i:02d}' for i in range(1,6)]:
        report = reports[name]
        report['window_relative_deviation_from_one_fifth'] = report['valid_windows']/(train_summary['valid_windows']/5)-1
        report['total_variation_from_training_pool'] = {
            axis: sum(abs(report['distributions'][axis].get(k,{}).get('window_fraction',0)-v['window_fraction'])
                      for k,v in train_summary['distributions'][axis].items())/2
            for axis in ('task_class','scene','task_category')}
    if sum(reports[f'stage_{i:02d}']['valid_windows'] for i in range(1,6)) != train_summary['valid_windows']:
        raise ValueError('Stage union differs from training pool')
    if train_summary['valid_windows']+reports['validation']['valid_windows'] != meta['total_windows']:
        raise ValueError('Training/validation partition is not lossless')
    report = dict(version='umi-formal-partition-v1', status='completed', seed=args.seed,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), tool_sha256=sha(__file__),
        parent_index=str(parent), parent_view_fingerprint=compute_view_fingerprint(meta),
        inputs=dict(inventory_sha256=sha(inventory_path),candidate_sha256=sha(candidate_path),
                    preserved_validation_sha256=sha(pilot_rows_path)),
        labels='Existing directory labels accepted by user: 60 tasks and 7 scenes; original instructions unchanged',
        quality_policy='Existing valid-window policy unchanged; no new exclusions or threshold changes',
        reserved_validation_episodes=sorted(held_episodes), reserved_source_groups=sorted(held_groups),
        validation_source_group_expansion_episodes=sorted(set(r['episode_index'] for r in validation)-held_episodes),
        partition_disjoint=True, partition_covers_parent=True, allocation=allocation, views=views, counts=reports)
    write(output/'formal_manifest.json',report)
    contract = dict(schema_version='umi-normalization-experiment-v1', experiment_id='qwenpi-umi-formal-v1',
        training_pool_approved=True, approval_basis='User instruction: fixed original validation groups; natural full training pool',
        parent_data_fingerprint=parent_fingerprint(meta),representation_fingerprint=content_fingerprint(build_representation(meta)),
        fit_view_fingerprint=views['train']['view_fingerprint'],
        allowed_apply_views=list(dict.fromkeys(v['view_fingerprint'] for v in views.values())),
        formal_manifest_sha256=sha(output/'formal_manifest.json'))
    validate_experiment_contract(contract,access_meta=meta,fit_view_fingerprint=views['train']['view_fingerprint'])
    write(output/'normalization_experiment.json',contract)
    with (output/'distribution.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['partition','axis','label','episodes','valid_windows','window_fraction'])
        for name, summary in reports.items():
            for axis, labels in summary['distributions'].items():
                for label, value in labels.items():
                    writer.writerow([name,axis,label,value['episodes'],value['valid_windows'],value['window_fraction']])
    write(output/'COMPLETED.json',dict(status='completed',manifest_sha256=sha(output/'formal_manifest.json')))
    (output/'PREPARING.json').unlink()
    print('Formal manifests/views complete. Statistics are a separate resumable offline step.',flush=True)


if __name__=='__main__':
    main()
