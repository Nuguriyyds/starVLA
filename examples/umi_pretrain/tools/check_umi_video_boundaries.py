"""CPU-only regression of the ORIGINAL cross-file first-window sequence.

Never replaces failed samples or changes the low-dimensional window index.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from starVLA.dataloader.umi_indexed_dataset import UMIIndexedDataset, collate_umi_samples
from starVLA.dataloader.umi_video_time import VideoTimeWindow
from starVLA.training.trainer_utils.umi_checkpoint import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Original loader_benchmark.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.baseline.read_text())
    provenance = previous["provenance"]
    indices = previous["sequences"]["cross_file_stress"]
    assert len(indices) == 48 and 126293227 in indices
    options = {k: provenance[k] for k in ("source_root", "allowed_video_roots", "image_size", "source_identity",
                                          "decode_tolerance_seconds")}
    report = {"status": "running", "original_indices": indices, "cases": []}
    for workload, requested in (("original_case", [126293227]), ("original_stress", indices)):
        for workers in (0, 2):
            raw = UMIIndexedDataset(provenance["index_dir"], return_metadata=True, **options)
            report["video_time_policy"] = raw.provenance()["video_time_policy"]
            kwargs = dict(batch_size=1, sampler=requested, collate_fn=collate_umi_samples,
                          num_workers=workers, generator=torch.Generator().manual_seed(42))
            if workers:
                kwargs.update(multiprocessing_context="spawn", prefetch_factor=2)
            records, error, iterator = [], None, None
            try:
                iterator = iter(DataLoader(raw, **kwargs))
                for batch in iterator:
                    sample = batch[0]
                    records.append({"metadata": {k: sample["umi_metadata"][k] for k in
                                     ("dataset_index", "episode_index", "frame_index", "video_decode")},
                                    "pixels_sha256": [hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                                                      for im in sample["image"]]})
            except Exception as exc:
                error = str(exc)
            finally:
                if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()
                raw.close()
            report["cases"].append({"workload": workload, "workers": workers,
                                    "requested": len(requested), "samples": records, "error": error})
            write_json(args.output, report)
            print(f"{workload} workers={workers} delivered={len(records)} error={error}", flush=True)
    for offset in (0, 2):
        assert report["cases"][offset]["samples"] == report["cases"][offset+1]["samples"]
    assert report["cases"][0]["error"] is None and report["cases"][1]["error"] is None
    failed_case = report["cases"][0]["samples"][0]
    camera = next(c for c in failed_case["metadata"]["video_decode"] if c["camera"].endswith("head_right"))
    assert camera["decoded_pts"] == 286066677000
    assert camera["time_base"] == "1/1000000000"
    raw = UMIIndexedDataset(provenance["index_dir"], return_metadata=True, **options)
    raw._worker()
    spans = []
    # Primary-key reads, including the real neighbours from the SAME video.
    for episode in (34299, 34300, 34301):
        metadata, _ = raw._episode(episode)
        camera = next(c for c in metadata["cameras"] if c["camera"].endswith("head_right"))
        duration = camera["to_timestamp"] - camera["from_timestamp"]
        start_image, start_info = raw._video_frame(camera, 0)
        _, tail_info = raw._video_frame(camera, duration - 1/raw.fps)
        rejected = False
        try:
            raw._video_frame(camera, duration + .000001)
        except ValueError:
            rejected = True
        assert rejected
        spans.append({"episode": episode, "camera": camera, "first": start_info,
                      "last_requested": tail_info, "outside_rejected": rejected})
    # Ensure the previously rejected frame cannot belong to either neighbour.
    from fractions import Fraction
    selected_time = Fraction(286066677000, 10**9)
    owners = []
    for record in spans:
        c = record["camera"]
        bounds = VideoTimeWindow(c["from_timestamp"], c["to_timestamp"], 0, Fraction(1,10**9), raw.decode_tolerance_seconds)
        if bounds.contains(selected_time):
            owners.append(record["episode"])
    assert owners == [34300]
    raw.close()
    report.update(status="failed" if any(c["error"] for c in report["cases"]) else "passed",
                  delivered_worker_pixels_and_locations_equal=True,
                  neighbouring_spans=spans, original_frame_owners=owners)
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "windows_per_worker_setting": len(indices),
                      "delivered_worker_pixels_and_locations_equal": True, "original_frame_owners": owners}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
