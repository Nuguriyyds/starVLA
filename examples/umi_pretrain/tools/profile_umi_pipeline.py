"""Read-only candidate-pool profiling and bounded engineering training benchmarks.

Training delegates to train_umi_pretrain; no optimizer/backward loop lives here.
Run loader/training separately, then report. Never clears OS/shared-storage caches.
"""
import argparse
from copy import deepcopy
from itertools import islice
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from starVLA.dataloader.umi_indexed_dataset import UMIIndexedDataset, apply_umi_normalization, collate_umi_samples
from starVLA.dataloader.umi_sampler import UMIBlockShuffleSampler
from starVLA.training.trainer_utils.umi_performance import distribution, process_resources
from starVLA.training.trainer_utils.umi_checkpoint import write_json
from starVLA.training.trainer_utils.umi_training_state import fingerprint


def raw_dataset(plan, index):
    keys = ("source_root", "allowed_video_roots", "image_size", "row_cache_mib", "source_identity")
    options = {k: plan["data"][k] for k in keys if k in plan["data"]}
    return UMIIndexedDataset(index, return_metadata=True, performance_diagnostics=True, **options)


def cross_file_sequence(raw, count):
    """Fixed spread across physical Parquet files, without reading their payloads."""
    with sqlite3.connect((raw.index_dir / "metadata.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        files = [r[0] for r in db.execute("SELECT data_file FROM data_files ORDER BY data_file")]
        selected = [files[i] for i in np.linspace(0, len(files)-1, min(16, len(files)), dtype=int)]
        anchors = []
        for file in selected:
            for (episode,) in db.execute("SELECT DISTINCT episode_index FROM segments WHERE data_file=? ORDER BY episode_index", (file,)):
                positions = np.flatnonzero(raw._ranges[:, 0] == episode)
                if len(positions):
                    pos = int(positions[0])
                    anchors.append(int(raw._cumulative[pos-1]) if pos else 0)
                    break
    if len(anchors) < 2:
        raise ValueError("Cross-file stress requires at least two indexed source files")
    return [anchors[i % len(anchors)] for i in range(count)]


def summarize_locality(metadata):
    episodes = [r["episode_index"] for r in metadata]
    longest, run, previous = 0, 0, None
    for episode in episodes:
        run = run+1 if episode == previous else 1
        longest, previous = max(longest, run), episode
    return dict(unique_episodes=len(set(episodes)), longest_same_episode_run=longest,
                unique_parquet_files=len({v["data_file"] for r in metadata for v in r["locations"]}),
                unique_video_files=len({v["video_path"] for r in metadata for v in r["video_decode"]}))


def loader_case(plan, index, indices, workers, seconds):
    start = time.monotonic()
    raw = raw_dataset(plan, index)
    dataset = apply_umi_normalization(raw, dict(plan["data"], normalization_purpose="engineering"))
    kwargs = dict(dataset=dataset, batch_size=1, sampler=indices, collate_fn=collate_umi_samples,
                  num_workers=workers, generator=torch.Generator().manual_seed(42))
    if workers:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**kwargs)
    creation = time.monotonic()-start
    waits, metadata, resources = [], [], []
    iterator = None
    error = None
    try:
        boot = time.monotonic()
        iterator = iter(loader)
        iteration = time.monotonic()-boot
        first = next(iterator)
        startup = time.monotonic()-boot
        metadata.extend(s["umi_metadata"] for s in first)
        resources.append(process_resources())
        stable = time.monotonic()
        for _ in range(len(indices)-1):
            before = time.monotonic()
            batch = next(iterator)
            waits.append(time.monotonic()-before)
            metadata.extend(s["umi_metadata"] for s in batch)
            resources.append(process_resources())
            if time.monotonic()-stable >= seconds:
                break
        stable_seconds = time.monotonic()-stable
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stable_seconds = None
        startup = locals().get("startup")
        iteration = locals().get("iteration")
    finally:
        close = time.monotonic()
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        dataset.close()
        close_seconds = time.monotonic()-close
    last_by_pid = {}
    for record in metadata:
        last_by_pid[record["reader_pid"]] = record["performance"]["caches"]
    return dict(num_workers=workers, prefetch_factor=2 if workers else None, error=error,
                requested_windows=len(indices), delivered_windows=len(metadata),
                loader_creation_seconds=creation, iterator_create_seconds=iteration,
                startup_through_first_batch_seconds=startup, stable_seconds=stable_seconds,
                stable_windows_per_second=len(waits)/stable_seconds if stable_seconds else None,
                close_seconds=close_seconds, total_seconds=time.monotonic()-start,
                batch_wait_seconds=distribution(waits), resources=resources,
                reader_lowdim_seconds=distribution([r["performance"]["lowdim_seconds"] for r in metadata]),
                reader_video_seconds=distribution([r["performance"]["video_seconds"] for r in metadata]),
                caches_last_delivered_by_pid=last_by_pid, locality=summarize_locality(metadata),
                samples=[{k: r[k] for k in ("dataset_index", "episode_index", "frame_index", "locations", "video_decode")} for r in metadata])


def loader_benchmark(args, plan):
    raw = raw_dataset(plan, args.candidate_index)
    sampler = UMIBlockShuffleSampler(len(raw), block_size=plan["data"].get("shuffle_block_size", 4096),
                                     seed=plan["training"].get("seed", 42), shuffle=True)
    sequences = {"sampler_order": list(islice(iter(sampler), args.loader_samples)),
                 "cross_file_stress": cross_file_sequence(raw, args.stress_samples)}
    report = dict(provenance=raw.provenance(), normalization_statistics=plan["data"].get("normalization_statistics"),
                  sampler=sampler.state_dict(), sequences=sequences, cases=[],
                  caveat="Candidate-pool read-only samples; engineering normalization is not formal statistics. Fresh workers are not cold OS/storage caches. Cases run sequentially: later cases may benefit from page cache.")
    raw.close()
    for name, indices in sequences.items():
        for workers in (0, 2, 4):
            print(f"loader workload={name} workers={workers}", flush=True)
            result = loader_case(plan, args.candidate_index, indices, workers, args.loader_seconds)
            result["workload"] = name
            report["cases"].append(result)
            write_json(args.output_dir / "loader_benchmark.json", report)
            print({k: result[k] for k in ("error", "delivered_windows", "stable_windows_per_second", "locality")}, flush=True)
    if any(c["error"] for c in report["cases"]):
        raise RuntimeError("A loader workload failed; report preserves its error and delivered prefix")


def training_benchmark(args, original):
    if original["purpose"] != "engineering":
        raise ValueError("Performance model updates must use explicit engineering views")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This benchmark driver currently launches one device only")
    calls = []
    for replay in (True, False):
        name = "decoded_replay" if replay else "end_to_end"
        plan = deepcopy(original)
        # Keep existing engineering stages/statistics. Both modes use the exact
        # same samples, budgets, initialization, full optimizer and scheduler.
        plan["stages"] = plan["stages"][:2]
        plan["stages"][0]["updates"] = args.updates+1
        plan["stages"][1]["updates"] = 1
        plan["training"].pop("save_every", None)
        plan.setdefault("checkpoint", {})["every_updates"] = 10000
        plan["training"].update(num_workers=0 if replay else 2,
                                eval_every=args.updates+2, trace_samples=True)
        plan["performance"] = dict(enabled=True, warmup_updates=2, detail_updates=2,
                                    max_training_seconds=args.training_seconds,
                                    decoded_replay=replay, replay_samples=64)
        plan_path = args.output_dir / f"{name}_plan.json"
        write_json(plan_path, plan)
        output = args.output_dir / name
        command = [sys.executable, "-m", "starVLA.training.train_umi_pretrain", "--plan", str(plan_path),
                   "--output-dir", str(output)]
        print(f"Starting {name}; training core unchanged, max update-loop seconds={args.training_seconds}", flush=True)
        launch(args.output_dir, calls, name, command)
        write_json(args.output_dir / "training_benchmark.json", dict(calls=calls, status="running"))
    # A completed-plan reload measures full verification/loading without new
    # updates. This new engineering run's identity is left intact.
    progress = json.loads((args.output_dir / "end_to_end/progress.json").read_text())
    if progress["status"] == "complete":
        launch(args.output_dir, calls, "end_to_end_reload", command + ["--resume", "latest"])
    write_json(args.output_dir / "training_benchmark.json", dict(calls=calls, status="finished"))
    summarize(args)


def launch(root, calls, name, command):
    log = root / (name + ".log")
    before = time.monotonic()
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(name=name, command=command, log=str(log), returncode=result.returncode,
                  process_wall_seconds=time.monotonic()-before)
    calls.append(record)
    print(record, flush=True)
    if result.returncode:
        write_json(root / "training_benchmark.json", dict(calls=calls, status="failed"))
        raise RuntimeError(log.read_text()[-12000:])


def summarize(args):
    path = args.output_dir / "training_benchmark.json"
    training = json.loads(path.read_text()) if path.exists() else dict(calls=[])
    training["runs"] = {}
    for name in ("decoded_replay", "end_to_end"):
        root = args.output_dir / name
        reports = [json.loads(p.read_text()) for p in sorted(root.glob("performance_rank_*.json"))]
        if not reports:
            continue
        active = [p for p in reports if p["completed_updates"]]
        seconds = sum(p["stable_seconds"] for p in active)
        windows = sum(p["measured_windows"] for p in active)
        process_wall = sum(c["process_wall_seconds"] for c in training["calls"] if c["name"] == name)
        updates = sum(p["completed_updates"] for p in active)
        plan = json.loads((args.output_dir / f"{name}_plan.json").read_text())
        batch = plan["training"]["batch_size"]*plan["training"]["gradient_accumulation_steps"]
        sections = {}
        for report in reports:
            for section in report["sections"]:
                sections.setdefault(section["name"], []).append(section["host_seconds"])
        traces = [json.loads(line) for line in (root / "trace_rank_0.jsonl").read_text().splitlines()]
        training["runs"][name] = dict(reports=reports, completed_updates=updates,
            stable_windows_per_second=windows/seconds if seconds else None,
            process_windows_per_second=updates*batch/process_wall if process_wall else None,
            samples_fingerprint=fingerprint([r["samples"] for r in traces]),
            losses=[r["loss"] for r in traces], sections={k: dict(distribution(v), total=sum(v)) for k,v in sections.items()})
    if len(training["runs"]) == 2:
        training["sample_streams_equal"] = len({r["samples_fingerprint"] for r in training["runs"].values()}) == 1
    write_json(path, training)
    lines = ["# UMI performance baseline", "", "Engineering measurement only; no full pretraining or multi-PPU claim.", "",
             "Fresh workers do not imply cold OS/storage caches. RSS sums include shared pages.",
             "Host sections overlap device work; CUDA events are warmup-only. Stable throughput uses synchronized interval boundaries.", "",
             "| Training | Updates | Stable windows/s | Process windows/s (startup/eval/save included) |",
             "|---|---:|---:|---:|"]
    for name, result in training["runs"].items():
        lines.append(f"| {name} | {result['completed_updates']} | {result['stable_windows_per_second']} | {result['process_windows_per_second']} |")
    loader_path = args.output_dir / "loader_benchmark.json"
    if loader_path.exists():
        lines += ["", "| Loader workload | Workers | Delivered | Windows/s | Episodes | Parquet files |", "|---|---:|---:|---:|---:|---:|"]
        for case in json.loads(loader_path.read_text())["cases"]:
            lines.append(f"| {case['workload']} | {case['num_workers']} | {case['delivered_windows']} | {case['stable_windows_per_second']} | {case['locality']['unique_episodes']} | {case['locality']['unique_parquet_files']} |")
    (args.output_dir / "PERFORMANCE.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("loader", "training", "report"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--loader-samples", type=int, default=256)
    parser.add_argument("--stress-samples", type=int, default=48)
    parser.add_argument("--loader-seconds", type=float, default=180)
    parser.add_argument("--updates", type=int, default=20, choices=range(1, 30))
    parser.add_argument("--training-seconds", type=float, default=1200)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = OmegaConf.to_container(OmegaConf.load(args.plan), resolve=True)
    config_path = args.output_dir / "performance_config.json"
    config = dict(base_plan=plan, base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "mode"})
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Benchmark configuration changed; use a new output directory")
    write_json(config_path, config)
    if args.mode == "loader":
        if args.candidate_index is None:
            parser.error("loader requires --candidate-index")
        loader_benchmark(args, plan)
    elif args.mode == "training":
        training_benchmark(args, plan)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
