"""Inventory candidate semantics from existing metadata and compact window index.

Never opens frame payloads, MCAP, images or videos. Directory labels remain
unconfirmed candidates; this tool does not select a split or sampling weights.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

ROOMS = {
    'living_room': ('living room',), 'bedroom': ('bedroom',),
    'entrance_hall': ('entrance hall', 'entryway'), 'kitchen': ('kitchen',),
    'balcony': ('balcony',), 'study': ('study',), 'patio': ('patio',),
    'bathroom': ('bathroom',), 'storage_room': ('storage room',),
    'office': ('office',), 'dining_room': ('dining room',),
}
ROOM_RULES = {key: re.compile(r'\b(?:in|on|at)\s+(?:(?:the|a|an)\s+)?(?:' +
                             '|'.join(re.escape(v) for v in values) + r')\b', re.I)
              for key, values in ROOMS.items()}


@lru_cache(maxsize=120000)
def room_mentions(text):
    # Only explicit location phrases, not an object such as "kitchen towel".
    return tuple(key for key, pattern in ROOM_RULES.items() if pattern.search(text))


def path_labels(path):
    if not path or '/roban_umi/' not in path:
        return ('unknown',) * 4, 'missing_or_unrecognized_path'
    parts = path.split('/roban_umi/', 1)[1].split('/')
    if len(parts) != 7 or not re.fullmatch(r'\d{8}', parts[0]) or not parts[-1].endswith('.mcap'):
        return ('unknown',) * 4, 'unrecognized_path_layout'
    return tuple(parts[2:6]), 'source_path_segments'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (dict, list))
                             else row.get(k) for k in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--access-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    catalog, access, output = [p.resolve() for p in (args.catalog, args.access_index, args.output)]
    if output == catalog or output.is_relative_to(catalog) or output == access or output.is_relative_to(access):
        raise ValueError('Output must be separate from existing inputs')
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((access / 'meta.json').read_text())
    if Path(meta['catalog_path']).resolve() != catalog:
        raise ValueError('Catalog does not match the compiled access index')
    ranges = np.load(access / 'ranges.npy', mmap_mode='r', allow_pickle=False)
    assert ranges.ndim == 2 and ranges.shape[1] == 3 and np.all(ranges[:, 2] > ranges[:, 1])
    counts = np.zeros(int(ranges[:, 0].max()) + 1, dtype=np.int64)
    np.add.at(counts, ranges[:, 0], ranges[:, 2] - ranges[:, 1])
    assert int(counts.sum()) == meta['total_windows']
    tasks = {x['task_index']: x['task'] for x in pq.read_table(catalog / 'task_catalog.parquet',
                                                          columns=['task_index', 'task']).to_pylist()}
    task_usage = defaultdict(Counter)
    groups = {}
    seen = set()
    group_keys = set()
    examples = defaultdict(list)
    conflicts = defaultdict(list)
    columns = ['episode_index', 'length', 'duration_seconds', 'task_ids', 'task_texts',
               'source_mcap', 'source_recording_key', 'source_set_id']
    with gzip.open(output / 'episode_candidates.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['episode_index', 'candidate_group_key', 'source_recording_key',
                         'source_set_id', 'task_ids', 'duration_seconds', 'valid_windows',
                         'text_room_mentions', 'text_path_room_disagreement'])
        for path in sorted((catalog / 'episode_manifest').glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=columns):
                for row in batch.to_pylist():
                    ep = int(row['episode_index'])
                    assert ep not in seen
                    seen.add(ep)
                    labels, basis = path_labels(row['source_mcap'])
                    key = '/'.join(labels)
                    if key not in groups:
                        groups[key] = dict(group_key=key, domain=labels[0], path_scene=labels[1],
                            path_category=labels[2], task_slug=labels[3], label_basis=basis,
                            episodes=0, eligible_episodes=0, valid_windows=0, duration_seconds=0.,
                            explicit_room_episodes=0, room_disagreement_episodes=0,
                            task_ids=set(), source_groups=set(), text_counts=Counter(),
                            room_counts=Counter(), example_source_path=row['source_mcap'])
                    g = groups[key]
                    windows = int(counts[ep]) if ep < len(counts) else 0
                    g['episodes'] += 1; g['eligible_episodes'] += int(windows > 0)
                    g['valid_windows'] += windows; g['duration_seconds'] += row['duration_seconds']
                    ids = set(row['task_ids'] or [])
                    text = list(dict.fromkeys(tasks[t] for t in sorted(ids)))
                    assert set(text) == set(row['task_texts'] or [])
                    g['task_ids'].update(ids); g['source_groups'].add(row['source_recording_key'])
                    if row['source_recording_key']:
                        group_keys.add(row['source_recording_key'])
                    rooms = sorted({s for t in text for s in room_mentions(t)})
                    disagreement = any(s != labels[1] for s in rooms)
                    g['explicit_room_episodes'] += int(bool(rooms))
                    g['room_disagreement_episodes'] += int(disagreement)
                    g['room_counts'].update(rooms)
                    for t in text: g['text_counts'][t] += 1
                    for t in ids: task_usage[t][key] += 1
                    example = dict(episode_index=ep, task_ids=sorted(ids), task_texts=text,
                                   text_rooms=rooms, source_mcap=row['source_mcap'])
                    # Stable hash sample over episodes; not the first records in the dataset.
                    score = hashlib.sha256(f'42:{ep}'.encode()).hexdigest()
                    examples[key].append((score, example))
                    examples[key] = sorted(examples[key], key=lambda x: x[0])[:3]
                    if disagreement:
                        conflicts[key].append((score, example))
                        conflicts[key] = sorted(conflicts[key], key=lambda x: x[0])[:3]
                    writer.writerow([ep, key, row['source_recording_key'], row['source_set_id'],
                        json.dumps(sorted(ids)), row['duration_seconds'], windows,
                        json.dumps(rooms), int(disagreement)])
    assert len(seen) == meta['total_episodes']
    assert not any(counts[x] for x in range(len(counts)) if x not in seen)
    result = []
    for number, key in enumerate(sorted(groups), 1):
        g = groups[key]
        g['group_id'] = f'T{number:03d}'
        g['unique_task_texts'] = len(g.pop('task_ids'))
        g['source_groups'] = len(g['source_groups'])
        g['top_texts'] = [dict(text=t, episode_mentions=n) for t,n in g.pop('text_counts').most_common(3)]
        g['text_room_counts'] = dict(g.pop('room_counts'))
        g['examples'] = [x[1] for x in examples[key]]
        g['disagreement_examples'] = [x[1] for x in conflicts[key]]
        result.append(g)
    assert sum(x['valid_windows'] for x in result) == meta['total_windows']
    scene_rows = []
    for scene in sorted({x['path_scene'] for x in result}):
        selected = [x for x in result if x['path_scene'] == scene]
        row = dict(path_scene=scene, task_groups=len(selected))
        for name in ('episodes','eligible_episodes','valid_windows','duration_seconds',
                     'explicit_room_episodes','room_disagreement_episodes'):
            row[name] = sum(x[name] for x in selected)
        scene_rows.append(row)
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(),
        status='candidate_labels_pending_human_review', catalog=str(catalog), access_index=str(access),
        source_files=['episode_manifest/*.parquet','task_catalog.parquet','ranges.npy','meta.json'],
        episodes=len(seen), unique_task_texts=len(tasks), used_task_ids=len(task_usage),
        task_groups=len(result), path_scene_labels=len(scene_rows), source_groups=len(group_keys),
        valid_windows=int(counts.sum()), eligible_episodes=int(np.count_nonzero(counts)),
        duration_seconds=sum(x['duration_seconds'] for x in result),
        explicit_room_episodes=sum(x['explicit_room_episodes'] for x in result),
        room_disagreement_episodes=sum(x['room_disagreement_episodes'] for x in result),
        unparsed_episodes=sum(x['episodes'] for x in result if x['label_basis']!='source_path_segments'),
        task_ids_in_multiple_path_groups=sum(len(v)>1 for v in task_usage.values()),
        limitations=['Directory labels are candidates, not verified physical scenes.',
            'Explicit room-phrase matching flags text disagreement only, not bad data.',
            'Task IDs/text variants are not task categories; task labels describe intended, not observed, behavior.',
            'No image, video, MCAP or low-dimensional frame data read. No train/validation split or weights chosen.',
            'Window counts overlap in time. Duration is metadata length/fps, not independent usable-window hours.'])
    write_json(output / 'inventory.json', dict(summary=summary,scenes=scene_rows,groups=result))
    write_json(output / 'summary.json', summary)
    write_csv(output / 'task_groups.csv', result, ['group_id','group_key','path_scene','path_category',
        'task_slug','episodes','eligible_episodes','valid_windows','duration_seconds','unique_task_texts',
        'explicit_room_episodes','room_disagreement_episodes','example_source_path'])
    with gzip.open(output / 'task_text_candidates.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['task_index','task_text','episode_mentions','candidate_group_counts'])
        for task_id,text in sorted(tasks.items()):
            writer.writerow([task_id,text,sum(task_usage[task_id].values()),json.dumps(task_usage[task_id])])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
