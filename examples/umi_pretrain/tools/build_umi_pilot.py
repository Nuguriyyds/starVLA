"""Build a deterministic task-coverage pilot from existing labels and valid ranges.

Reads metadata only. Public data and the full access index remain unchanged.
Directory labels are accepted stratification labels, not newly inferred semantics.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import yaml

from starVLA.dataloader.umi_indexed_dataset import compute_view_fingerprint
from starVLA.training.trainer_utils.umi_checkpoint import sha256_file, write_json


def spread_anchors(ranges, maximum=32):
    """Evenly spaced ranks in the valid-anchor set, without expanding all frames."""
    sizes = ranges[:, 2] - ranges[:, 1]
    cumulative = np.cumsum(sizes)
    ranks = np.linspace(0, int(cumulative[-1])-1, min(maximum, int(cumulative[-1])), dtype=np.int64)
    buckets = np.searchsorted(cumulative, ranks, side='right')
    before = np.concatenate(([0], cumulative[:-1]))
    anchors = ranges[buckets, 1] + ranks - before[buckets]
    # Very short episodes can supply fewer well-separated windows. Do not pad.
    selected = []
    for anchor in anchors.tolist():
        if not selected or anchor-selected[-1] >= 16:
            selected.append(anchor)
    return selected


def historical_sources(private_root, source_db):
    """Conservatively exclude every episode in an earlier QwenPI train view.

    Old debug jobs predate plans; their private dataset was original episode 0.
    Whole training views are excluded even when a short job consumed only part.
    """
    evidence = [{'reason': 'legacy private debug/world-pose runs', 'episodes': [0]}]
    seen_views = set()
    for path in sorted((private_root/'runs').rglob('plan_requested.json')):
        plan = json.loads(path.read_text())
        if plan.get('model_kind') != 'qwenpi':
            continue
        for stage in plan['stages']:
            view = Path(stage['index_dir'])
            if str(view) in seen_views:
                continue
            seen_views.add(str(view))
            array = np.load(view/'ranges.npy', mmap_mode='r')
            evidence.append({'plan': str(path), 'view': str(view),
                             'episodes': np.unique(array[:, 0]).tolist()})
    episodes = sorted({ep for item in evidence for ep in item['episodes']})
    keys = set()
    for ep in episodes:
        row = source_db.execute('SELECT payload FROM episodes WHERE episode_index=?', (ep,)).fetchone()
        if not row:
            raise ValueError(f'Historical episode {ep} is absent from this source')
        key = json.loads(row[0])['source_recording_key']
        if not key:
            raise ValueError(f'Missing historical source identity for {ep}')
        keys.add(key)
    return keys, {'episodes': episodes, 'source_keys': sorted(keys), 'evidence': evidence}


def make_view(parent, destination, rows, source_db, parent_meta):
    """Copy selected metadata, not the 1.5 GB full database or any sensor data."""
    destination.mkdir(parents=True, exist_ok=False)
    episodes = sorted({r['episode_index'] for r in rows})
    anchors = sorted((r['episode_index'], a, a+1) for r in rows for a in r['anchors'])
    if not anchors or len(set(anchors)) != len(anchors):
        raise ValueError('Empty or duplicate pilot anchors')
    np.save(destination/'ranges.npy', np.asarray(anchors, dtype=np.int64), allow_pickle=False)
    np.save(destination/'cumulative.npy', np.arange(1, len(anchors)+1, dtype=np.int64), allow_pickle=False)
    target = sqlite3.connect(destination/'metadata.sqlite3')
    try:
        for (ddl,) in source_db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name IN ('episodes','segments','tasks','data_files') ORDER BY name"):
            target.execute(ddl)
        tasks, files = set(), set()
        for ep in episodes:
            payload = source_db.execute('SELECT payload FROM episodes WHERE episode_index=?', (ep,)).fetchone()[0]
            target.execute('INSERT INTO episodes VALUES (?,?)', (ep, payload))
            tasks.update(json.loads(payload)['task_ids'])
            segments = source_db.execute('SELECT * FROM segments WHERE episode_index=? ORDER BY episode_row_offset_start', (ep,)).fetchall()
            target.executemany('INSERT INTO segments VALUES (?,?,?,?,?,?,?,?)', segments)
            files.update(s[1] for s in segments)
        for task in sorted(tasks):
            target.execute('INSERT INTO tasks VALUES (?,?)', source_db.execute('SELECT * FROM tasks WHERE task_index=?', (task,)).fetchone())
        for file in sorted(files):
            target.execute('INSERT INTO data_files VALUES (?,?,?)', source_db.execute('SELECT * FROM data_files WHERE data_file=?', (file,)).fetchone())
        target.commit()
        counts = {t: target.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                  for t in ('episodes','segments','tasks','data_files')}
    finally:
        target.close()
    write_json(destination/'episodes.json', rows)
    meta = deepcopy(parent_meta)
    meta.update(total_windows=len(anchors), total_ranges=len(anchors), total_episodes=len(episodes),
                trainable_episodes=len(episodes), database_counts=counts, verified_full_index_counts=False,
                episode_list=str(destination/'episodes.json'),
                episode_list_sha256=sha256_file(destination/'episodes.json'))
    meta['engineering_selection'] = {'parent_index': str(parent),
        'parent_view': compute_view_fingerprint(parent_meta), 'purpose': 'task_coverage_pilot',
        'selection_sha256': meta['episode_list_sha256']}
    meta['scope'] = 'Fixed task-coverage pilot; no natural-distribution claim.'
    for name in ('ranges.npy','cumulative.npy','metadata.sqlite3'):
        p = destination/name
        meta['artifacts'][name] = {'size_bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    meta['view_fingerprint'] = compute_view_fingerprint(meta)
    write_json(destination/'meta.json', meta)
    return {'episodes': len(episodes), 'windows': len(anchors), 'view_fingerprint': meta['view_fingerprint']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root', type=Path, required=True)
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--inventory-dir', type=Path, required=True)
    parser.add_argument('--reference-plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inventory = json.loads((args.inventory_dir/'inventory.json').read_text())
    groups = {g['group_key']: g for g in inventory['groups']}
    if len(groups) != 60:
        raise ValueError('Expected the accepted 60 directory categories')
    parent_meta = json.loads((args.index_dir/'meta.json').read_text())
    full_ranges = np.load(args.index_dir/'ranges.npy', mmap_mode='r')
    if sha256_file(args.index_dir/'ranges.npy') != parent_meta['artifacts']['ranges.npy']['sha256']:
        raise ValueError('Parent valid ranges changed')
    db = sqlite3.connect(f'file:{args.index_dir}/metadata.sqlite3?mode=ro', uri=True)
    excluded, history = historical_sources(args.private_root, db)
    write_json(root/'historical_training_exclusions.json', history)
    candidates = defaultdict(list)
    with gzip.open(args.inventory_dir/'episode_candidates.csv.gz', 'rt', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if int(row['valid_windows']) <= 0 or row['source_recording_key'] in excluded:
                continue
            if not row['source_recording_key']:
                raise ValueError('Cannot split an episode without source identity')
            row['episode_index'] = int(row['episode_index'])
            candidates[row['candidate_group_key']].append(row)
    selected, used, quotas = {'train': [], 'validation': []}, set(), []
    for key, group in sorted(groups.items(), key=lambda kv: kv[1]['group_id']):
        def order(row):
            value = f"{args.seed}|{key}|{row['source_recording_key']}|{row['episode_index']}"
            return hashlib.sha256(value.encode()).hexdigest()
        pool = sorted(candidates[key], key=order)
        counts = {}
        for split, quota in [('validation', 2), ('train', 4)]:
            counts[split] = 0
            for row in pool:
                if row['source_recording_key'] in used:
                    continue
                ep = row['episode_index']
                left = np.searchsorted(full_ranges[:, 0], ep, side='left')
                right = np.searchsorted(full_ranges[:, 0], ep, side='right')
                anchors = spread_anchors(full_ranges[left:right])
                if not anchors:
                    continue
                payload = json.loads(db.execute('SELECT payload FROM episodes WHERE episode_index=?', (ep,)).fetchone()[0])
                if payload['source_recording_key'] != row['source_recording_key']:
                    raise ValueError('Inventory and access index disagree on source identity')
                selected[split].append(dict(episode_index=ep, group_id=group['group_id'],
                    group_key=key, path_scene=group['path_scene'], path_category=group['path_category'],
                    task_slug=group['task_slug'], source_recording_key=row['source_recording_key'],
                    source_set_id=row['source_set_id'], task_ids=payload['task_ids'], task_texts=payload['task_texts'],
                    valid_window_count=int(row['valid_windows']), anchors=anchors))
                used.add(row['source_recording_key'])
                counts[split] += 1
                if counts[split] == quota:
                    break
            if counts[split] == 0:
                raise ValueError(f'No {split} source for {key}; see candidate availability')
        quotas.append(dict(group_id=group['group_id'], group_key=key, **counts))
    train_keys = {r['source_recording_key'] for r in selected['train']}
    val_keys = {r['source_recording_key'] for r in selected['validation']}
    assert train_keys.isdisjoint(val_keys) and val_keys.isdisjoint(excluded)
    views = {split: make_view(args.index_dir, root/split, rows, db, parent_meta)
             for split, rows in selected.items()}
    db.close()
    scenes = sorted({g['path_scene'] for g in groups.values()})
    with (root/'scene_task_cross.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scene','group_id','task_slug','pool_episodes','pool_valid_windows','train_episodes','train_windows','validation_episodes','validation_windows'])
        writer.writeheader()
        for scene in scenes:
            for group in sorted(groups.values(), key=lambda g:g['group_id']):
                row = dict(scene=scene, group_id=group['group_id'], task_slug=group['task_slug'],
                    pool_episodes=group['episodes'] if scene==group['path_scene'] else 0,
                    pool_valid_windows=group['valid_windows'] if scene==group['path_scene'] else 0)
                for split in selected:
                    subset = [r for r in selected[split] if r['group_id']==group['group_id'] and r['path_scene']==scene]
                    row[split+'_episodes'] = len(subset)
                    row[split+'_windows'] = sum(len(r['anchors']) for r in subset)
                writer.writerow(row)
    manifest = {'version': 'umi-task-coverage-pilot-v1', 'seed': args.seed,
        'label_basis': 'existing directory labels accepted by user; original instructions retained',
        'source_group_basis': 'catalog source_recording_key (original MCAP recording)',
        'views': views, 'quotas': quotas, 'history_excluded_episodes': history['episodes'],
        'train_validation_source_overlap': 0, 'validation_historical_source_overlap': 0,
        'scenes': scenes, 'selection_policy': 'SHA256 seeded source ordering; validation first; <=32 dispersed anchors, spacing >=16 rows; no replacement',
        'normalization': 'fit train view only; separate state/action mean/std; apply same file to validation',
        'full_training_policy': 'natural valid-window distribution; five future shards match final training pool; not implemented by this pilot',
        'inventory_sha256': sha256_file(args.inventory_dir/'inventory.json'),
        'candidate_table_sha256': sha256_file(args.inventory_dir/'episode_candidates.csv.gz'),
        'tool_sha256': sha256_file(Path(__file__)),
        'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(root/'pilot_manifest.json', manifest)
    plan = yaml.safe_load(args.reference_plan.read_text())
    plan['stages'] = [dict(name='task_coverage_60', index_dir=str(root/'train'), updates=10000)]
    plan['evaluation'] = dict(name='independent_source_validation', index_dir=str(root/'validation'), seed=19876)
    plan['data'].update(normalization_statistics=str(root/'statistics.json'), shuffle_block_size=1)
    plan['training'].update(batch_size=1,gradient_accumulation_steps=2,eval_every=500,trace_samples=True)
    plan['checkpoint'] = dict(integrity='basic',every_updates=1000)
    plan['pilot'] = dict(manifest=str(root/'pilot_manifest.json'), manifest_sha256=sha256_file(root/'pilot_manifest.json'),
        evaluate_updates=[0,1000,2500,5000,7500,10000], curve_windows_per_episode=4,
        curve_train_episodes_per_task=1, full_validation_at=10000,
        runner_sha256=sha256_file(Path(__file__).with_name('train_qwenpi_pilot.py')))
    (root/'plan.yaml').write_text(yaml.safe_dump(plan,sort_keys=False),encoding='utf-8')
    print(json.dumps({'views': views,'history_excluded': history['episodes'], 'plan':str(root/'plan.yaml')},indent=2),flush=True)


if __name__ == '__main__':
    main()
