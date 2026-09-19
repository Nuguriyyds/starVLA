"""Check batches from the full indexed loader against independent raw rows.

No model or GPU is loaded. This verifies the access interface, not dataset
physical quality, training convergence, distributed throughput, or full resume.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from PIL import Image
    from starVLA.dataloader import build_dataloader
    from starVLA.dataloader.umi_indexed_dataset import SIGNALS, WIDTHS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "examples/umi_pretrain/train_files/umi_indexed_data.yaml")
    parser.add_argument("--output", type=Path, default=REPO.parent / "runs" / ("umi_access_check_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()
    if args.batches < max(1, args.num_workers):
        raise ValueError("Read at least one batch per requested worker")
    output = args.output.resolve()
    if not output.is_relative_to(REPO.parent / "runs") or output.exists():
        raise ValueError("Use a new private runs subdirectory for this access check")
    cfg = OmegaConf.load(args.config)
    cfg.output_dir = str(output)
    cfg.datasets.vla_data.num_workers = args.num_workers
    cfg.datasets.vla_data.per_device_batch_size = 2
    cfg.datasets.vla_data.return_metadata = True
    loader = build_dataloader(cfg, dataset_py="umi_indexed")
    dataset = loader.dataset
    print(f"Full dataset: {len(dataset):,} windows; {dataset.meta['total_ranges']:,} ranges", flush=True)
    iterator = iter(loader)
    batch = [sample for _ in range(args.batches) for sample in next(iterator)]
    task_rows = pq.read_table(dataset.source_root / "meta/tasks.parquet", columns=["task_index", "task"]).to_pylist()
    tasks = {row["task_index"]: row["task"] for row in task_rows}
    records = []
    try:
        for sample in batch:
            trace = sample["umi_metadata"]
            # Independent oracle: filter the physical Parquet by episode,
            # then use its original episode row order (no segment-map helper).
            paths = list(dict.fromkeys(part["data_file"] for part in trace["locations"]))
            if len(paths) != 1:
                raise ValueError("Oracle expects the current index version's one-file-per-episode assignment")
            raw = pq.read_table(dataset.source_root / paths[0], columns=["episode_index", "frame_index", *SIGNALS])
            episode = raw.filter(pc.equal(raw["episode_index"], trace["episode_index"]))
            rows = episode.slice(trace["episode_row_offset"], 17)
            expected = np.concatenate([
                np.asarray(rows[key].to_pylist(), dtype=np.float32).reshape(17, width)
                for key, width in zip(SIGNALS, WIDTHS)
            ], axis=1)
            for key, value in (("state", expected[:1]), ("action", expected[1:])):
                assert sample[key].dtype == np.dtype("float32")
                assert np.array_equal(sample[key], value), f"{key} differs from raw episode rows"
            assert rows["frame_index"].to_pylist()[1:] == trace["future_frame_indices"]
            assert len(sample["image"]) == 4
            assert all(isinstance(image, Image.Image) and image.mode == "RGB" and image.size == dataset.image_size
                       for image in sample["image"])
            assert isinstance(sample["lang"], str) and sample["lang"].strip()
            assert sample["lang"] == tasks[trace["task_index"]], "Language differs from original task table"
            records.append(dict(trace, state_shape=list(sample["state"].shape),
                                action_shape=list(sample["action"].shape), image_count=len(sample["image"]),
                                raw_values_exact_match=True, language=sample["lang"]))
            print(f"PASS episode={trace['episode_index']}, frame={trace['frame_index']}, "
                  "four RGB images; raw state/action exact match", flush=True)
        worker_pids = sorted({record["reader_pid"] for record in records})
        if len(worker_pids) != max(1, args.num_workers):
            raise ValueError("Not all requested reader processes returned a checked sample")
        report = {"status": "passed", "scope": "indexed batch access and raw lowdim/language equality; no model/training",
                  "total_windows": len(dataset), "num_workers": args.num_workers,
                  "reader_pids": worker_pids, "batches": args.batches,
                  "rule_fingerprint": dataset.meta["rule_fingerprint"], "samples": records}
        (output / "access_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Saved:", output / "access_check.json", flush=True)
    finally:
        # Owning-process teardown closes worker-local SQLite/video handles.
        del iterator
        del loader
        dataset.close()


if __name__ == "__main__":
    main()
