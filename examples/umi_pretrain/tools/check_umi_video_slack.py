"""One bounded regression: original 48 requests, workers 0/2, existing PTS evidence.

No source metadata rescan, model load, replacement samples or window rebuilding.
The prior captured PTS are a nearest-candidate reference, not ownership proof.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from check_umi_video_boundaries import RecordEveryRequest
from starVLA.dataloader.umi_indexed_dataset import UMIIndexedDataset, collate_umi_samples
from starVLA.training.trainer_utils.umi_checkpoint import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.baseline.read_text())
    indices = old['original_indices']
    assert len(indices) == 48 and len(set(indices)) == 16
    p = old['provenance']
    options = {k: p[k] for k in ('source_root', 'allowed_video_roots', 'image_size',
                                 'source_identity', 'decode_tolerance_seconds')}
    options.update(video_time_policy='engineering_boundary_slack', boundary_slack_seconds=.001,
                   return_metadata=True)
    expected = {}
    for ep in old['episode_probes']:
        position = next(v for v in ep['positions'] if v['label'] == 'original_request')
        for c in position['cameras']:
            low, high = Fraction(str(c['from_timestamp'])), Fraction(str(c['to_timestamp']))
            target = Fraction(c['query_fraction'])
            eligible = [f for f in c['candidates'] if
                        low-Fraction(1,1000) <= Fraction(f['time_fraction']) < high+Fraction(1,1000)
                        and abs(Fraction(f['time_fraction'])-target) <= Fraction(str(options['decode_tolerance_seconds']))]
            best = min(eligible, key=lambda f: (abs(Fraction(f['time_fraction'])-target), Fraction(f['time_fraction']))) if eligible else None
            expected[(ep['dataset_index'], c['camera'])] = best
    report = dict(status='running', name='video_read_engineering_slack_regression',
                  original_indices=indices, baseline=str(args.baseline), cases=[],
                  physical_synchronization_verified=False, lowdim_index_rebuilt=False)
    for workers in (0, 2):
        raw = UMIIndexedDataset(p['index_dir'], **options)
        report['provenance'] = raw.provenance()
        kwargs = dict(batch_size=1, sampler=indices, collate_fn=collate_umi_samples,
                      num_workers=workers, generator=torch.Generator().manual_seed(42))
        if workers: kwargs.update(multiprocessing_context='spawn', prefetch_factor=2)
        case = dict(workers=workers, samples=[])
        report['cases'].append(case)
        iterator = iter(DataLoader(RecordEveryRequest(raw), **kwargs))
        try:
            for i, batch in enumerate(iterator):
                sample = batch[0]
                if sample['status'] == 'passed':
                    for frame in sample['metadata']['video_decode']:
                        ref = expected[(sample['dataset_index'], frame['camera'])]
                        frame['matches_captured_nearest_eligible'] = bool(ref and
                            frame['decoded_pts'] == ref['pts'] and frame['time_base'] == ref['time_base'])
                case['samples'].append(sample)
                if (i+1) % 12 == 0: print(f'workers={workers}: {i+1}/48 recorded', flush=True)
        finally:
            if hasattr(iterator, '_shutdown_workers'): iterator._shutdown_workers()
            raw.close()
        case['passed'] = sum(s['status'] == 'passed' for s in case['samples'])
        case['complete_original_sequence'] = [s['dataset_index'] for s in case['samples']] == indices
        write_json(args.output, report)
    report['worker_records_equal'] = report['cases'][0]['samples'] == report['cases'][1]['samples']
    frames = [f for s in report['cases'][0]['samples'] if s['status'] == 'passed' for f in s['metadata']['video_decode']]
    report['summary'] = dict(requests_per_worker=48, frames_per_worker=len(frames),
        used_boundary_slack=sum(f['used_boundary_slack'] for f in frames),
        nearest_reference_mismatches=sum(not f['matches_captured_nearest_eligible'] for f in frames),
        max_abs_offset_seconds=max((abs(f['offset_seconds']) for f in frames),default=None))
    passed = all(c['passed'] == 48 and c['complete_original_sequence'] for c in report['cases'])
    passed = passed and report['worker_records_equal'] and report['summary']['nearest_reference_mismatches'] == 0
    report['status'] = 'passed' if passed else 'failed'
    write_json(args.output, report)
    print(json.dumps(dict(status=report['status'], **report['summary'])), flush=True)
    if not passed: raise SystemExit(1)


if __name__ == '__main__':
    main()
