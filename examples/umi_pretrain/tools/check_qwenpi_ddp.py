"""Real two-device engineering acceptance using the existing staged trainer.

`run` launches continuous A2/B2, then A2 pause + a fresh-process B2 resume.
No training algorithm, data policy, checkpoint format or installed package is
modified. The worker retains live model/optimizer references for a final probe;
the production builder's arguments and return values are forwarded unchanged.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    from starVLA.training import train_umi_pretrain as training
    original = training.build_optimizer
    original_end = training.UMIAccelerator.end_training
    captured = {}

    def retain_references(model, config):
        result = original(model, config)
        captured.update(model=model, optimizer=result[0])
        return result

    def probe_then_end(accelerator):
        # Record before native process-group destruction: this SDK crashes in
        # that destruction even in a standalone 16x16 DDP reproduction. Still
        # call the original cleanup and propagate the nonzero process exit.
        path = args.run_dir / "progress.json"
        plan = yaml.safe_load(args.plan.read_text())
        expected = args.stop_after_update or sum(s["updates"] for s in plan["stages"])
        if path.is_file() and captured:
            progress = json.loads(path.read_text())
            if progress["global_update"] == expected and progress["status"] in ("paused", "complete"):
                rank_probe(captured, args)
        return original_end(accelerator)

    training.build_optimizer = retain_references
    training.UMIAccelerator.end_training = probe_then_end
    try:
        training.run(argparse.Namespace(plan=args.plan, output_dir=args.run_dir,
                     resume=args.resume, stop_after_update=args.stop_after_update, cpu=False))
    finally:
        training.build_optimizer = original
        training.UMIAccelerator.end_training = original_end


def rank_probe(captured, args):
    # Outside the optimization/save timing. No additional RNG draws or parameter
    # updates. Sample each tensor's first/middle/last entries, NOT a full hash.
    model, optimizer = captured["model"], captured["optimizer"]
    records, vectors, locations = {}, [], []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        record = {"shape": list(parameter.shape), "requires_grad": parameter.requires_grad,
                  "adam_step": float(state["step"]) if "step" in state else None}
        records[name] = record
        for label, value in (("parameter", parameter), ("exp_avg", state.get("exp_avg")),
                             ("exp_avg_sq", state.get("exp_avg_sq"))):
            if value is None or not value.numel():
                continue
            flat = value.detach().reshape(-1)
            indices = [0, flat.numel() // 2, flat.numel() - 1]
            vectors.append(flat[indices].float())
            locations.append((name, label))
    values = torch.stack(vectors).cpu().tolist()
    for (name, label), value in zip(locations, values):
        records[name][label] = value
    if not all(math.isfinite(x) for row in values for x in row):
        raise RuntimeError("Nonfinite final rank probe")
    step = json.loads((args.run_dir / "progress.json").read_text())["global_update"]
    rank = int(os.environ["RANK"])
    write_json(args.run_dir / f"rank_probe_update_{step:08d}_rank_{rank}.json",
        {"rank": rank, "local_rank": int(os.environ["LOCAL_RANK"]), "world_size": int(os.environ["WORLD_SIZE"]),
         "device": str(next(model.parameters()).device), "global_update": step,
         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
         "probe": "first/middle/last entries of each parameter and existing Adam moment tensor",
         "full_tensor_equality_proven": False, "parameters": records})
    print(f"rank={rank} final parameter/Adam probe saved at update={step}", flush=True)


def equal_tree(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from tensors(child, f"{prefix}/{i}")


def tensor_comparison(left_path, right_path):
    # A bounded diagnostic: once differences are observed, scanning every byte
    # cannot establish exact repeatability. mmap plus three entries per tensor
    # avoids reading two full model+Adam copies merely to confirm inequality.
    start = time.monotonic()
    left = torch.load(left_path, map_location="cpu", weights_only=False, mmap=True)
    right = torch.load(right_path, map_location="cpu", weights_only=False, mmap=True)
    a, b = dict(tensors(left)), dict(tensors(right))
    if a.keys() != b.keys():
        raise ValueError("Checkpoint tensor keys differ")
    maximum, squared, reference_squared, count, mismatched = 0., 0., 0., 0, []
    for name, x in a.items():
        y = b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Shape/dtype differs: {name}")
        x, y = x.reshape(-1), y.reshape(-1)
        if x.numel():
            ids = [0, x.numel()//2, x.numel()-1]
            x, y = x[ids], y[ids]
        different = False
        for offset in range(0, x.numel(), 1024**2):
            u, v = x[offset:offset+1024**2], y[offset:offset+1024**2]
            if not torch.isfinite(u).all() or not torch.isfinite(v).all():
                raise ValueError(f"Nonfinite checkpoint: {name}")
            different |= not torch.equal(u, v)
            delta = u.double() - v.double()
            maximum = max(maximum, float(delta.abs().max()))
            squared += float(delta.square().sum())
            reference_squared += float(u.double().square().sum())
            count += u.numel()
        if different:
            mismatched.append(name)
    extra = {}
    if "state" in left and "param_groups" in left:
        extra["parameter_groups_equal"] = equal_tree(left["param_groups"], right["param_groups"])
        extra["left_adam_steps"] = dict(Counter(str(float(s["step"])) for s in left["state"].values()))
        extra["right_adam_steps"] = dict(Counter(str(float(s["step"])) for s in right["state"].values()))
    result = {"sampled": True, "tensors": len(a), "elements_compared": count,
              "mismatched_tensors": len(mismatched), "first_mismatched_names": mismatched[:8],
              "exact_equal": not mismatched, "max_abs_difference": maximum,
              "rms_difference": math.sqrt(squared/count) if count else 0.,
              "reference_rms": math.sqrt(reference_squared/count) if count else 0.,
              "seconds": time.monotonic()-start, **extra}
    del a, b, left, right
    gc.collect()
    return result


def report(root):
    from starVLA.dataloader.umi_sampler import UMIBlockShuffleSampler
    plan = yaml.safe_load((root / "plan.yaml").read_text())
    result = {"status": "verifying", "world_size": 2, "global_batch": 4, "updates": 4,
              "pause_update": 2, "stage_sequence": ["A", "A", "B", "B"],
              "rank_probes": [], "checkpoint_comparisons": {}, "performance": []}
    traces = {}
    for run in ("continuous", "resumed"):
        folder = root / run
        identity = json.loads((folder / "run_identity.json").read_text())
        assert identity["runtime"]["world_size"] == 2
        assert identity["runtime"]["collective_backend"] == "nccl"
        assert identity["runtime"]["device_type"] == "cuda"
        result[run + "_runtime"] = identity["runtime"]
        traces[run] = [[json.loads(line) for line in (folder / f"trace_rank_{rank}.jsonl").read_text().splitlines()]
                       for rank in (0, 1)]
        for trace in traces[run]:
            assert len(trace) == 4 and [e["stage"] for e in trace] == result["stage_sequence"]
        cursors = {}
        for i in range(4):
            event = traces[run][0][i]
            stage = next(s for s in plan["stages"] if s["name"] == event["stage"])
            size = json.loads((Path(stage["index_dir"]) / "meta.json").read_text())["total_windows"]
            key = event["stage"], event["epoch_used"]
            cursor = cursors.get(key, 0)
            import itertools
            sampler = UMIBlockShuffleSampler(size, block_size=plan["data"]["shuffle_block_size"],
                        seed=plan["training"]["seed"], epoch=key[1])
            expected = list(itertools.islice(iter(sampler), cursor, cursor+4))
            observed = [s["dataset_index"] for micro in range(2) for rank in (0, 1)
                        for s in traces[run][rank][i]["samples"][micro]]
            assert observed == expected and len(set(observed)) == 4
            assert traces[run][0][i]["grad_norm_before_clip"] == traces[run][1][i]["grad_norm_before_clip"]
            cursors[key] = cursor+4
        for step in ((4,) if run == "continuous" else (2, 4)):
            paths = [folder / f"rank_probe_update_{step:08d}_rank_{rank}.json" for rank in (0, 1)]
            if run == "continuous" and not any(p.exists() for p in paths):
                # Preserve the first actual run. It completed both published
                # checkpoints, then crashed before the original post-run probe.
                result["rank_probes"].append({"run": run, "step": step, "status": "not_recorded_before_shutdown_crash"})
                continue
            pair = [json.loads((folder / f"rank_probe_update_{step:08d}_rank_{rank}.json").read_text()) for rank in (0, 1)]
            same = pair[0]["parameters"] == pair[1]["parameters"]
            result["rank_probes"].append({"run": run, "step": step, "equal": same,
                "devices": [p["device"] for p in pair], "parameter_tensors": len(pair[0]["parameters"]),
                "peak_allocated_gib": [p["peak_allocated_gib"] for p in pair],
                "scope": "first/middle/last entries per tensor; not full rank tensor equality"})
            assert same and [p["local_rank"] for p in pair] == [0, 1]
        for p in sorted(folder.glob("performance_rank_*.json")):
            perf = json.loads(p.read_text())
            result["performance"].append({"run": run, "file": p.name, "elapsed_seconds": perf["elapsed_seconds"],
                "completed_updates": perf["completed_updates"], "sections": [s for s in perf["sections"] if
                s["name"].startswith("checkpoint") or s["name"].startswith("resume")],
                "peak_allocated_gib": max((r.get("device",{}).get("peak_allocated_bytes",0) for r in perf["resources"]),default=0)/2**30})
    for rank in (0, 1):
        for a, b in zip(traces["continuous"][rank], traces["resumed"][rank]):
            for key in ("global_update", "stage", "epoch_used", "samples", "lr_used", "lr_next", "progress"):
                assert a[key] == b[key], key
    result["samples_progress_learning_rates_equal"] = True
    result["samples_match_independent_global_stream"] = True
    result["losses"] = {name: [e["loss"] for e in traces[name][0]] for name in traces}
    for step in (2, 4):
        location = f"checkpoints/update_{step:08d}"
        left, right = root / "continuous" / location, root / "resumed" / location
        for filename in ("custom_checkpoint_0.pkl", "custom_checkpoint_1.pkl", "strict_rng_0.pt", "strict_rng_1.pt"):
            assert equal_tree(torch.load(left / filename, map_location="cpu", weights_only=False),
                              torch.load(right / filename, map_location="cpu", weights_only=False)), (step, filename)
        for folder in (left, right):
            manifest = json.loads((folder / "manifest.json").read_text())
            assert manifest["integrity"] == "basic"
        for filename in ("pytorch_model.bin", "optimizer.bin"):
            print(f"Compare update={step} {filename} sampled", flush=True)
            key = f"update_{step}/{filename}"
            result["checkpoint_comparisons"][key] = tensor_comparison(left / filename, right / filename)
            write_json(root / "acceptance.json", result)
    result["scheduler_progress_rank_rng_equal"] = True
    final = [v for k, v in result["checkpoint_comparisons"].items() if k.startswith("update_4/")]
    assert final[1]["parameter_groups_equal"] and final[1]["left_adam_steps"] == final[1]["right_adam_steps"]
    assert set(final[1]["left_adam_steps"]) == {"4.0"}
    result["numerical_reproducibility"] = "sampled_equal_full_not_verified" if all(v["exact_equal"] for v in final) else "differences_measured_not_yet_explained"
    result["full_tensor_equality_proven"] = False
    processes = json.loads((root / "processes.json").read_text())
    result["processes"] = processes
    result["all_processes_exited_cleanly"] = all(p["exit_code"] == 0 for p in processes)
    result["status"] = ("failed_process_group_shutdown" if not result["all_processes_exited_cleanly"] else
                        "control_flow_passed_sampled_numerics_reported")
    result["caveats"] = ["Engineering views do not validate video physical synchronization or policy quality.",
                           "Tensor comparisons sample three entries per tensor, not full-tensor equality; excluded from checkpoint costs.",
                           "Two PPU acceptance does not validate multi-node or sharded training."]
    write_json(root / "acceptance.json", result)
    print(json.dumps({"status": result["status"], "numerical_reproducibility": result["numerical_reproducibility"]}), flush=True)
    if not result["all_processes_exited_cleanly"]:
        raise SystemExit(1)


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((REPO / "examples/umi_pretrain/train_files/umi_training_qwenpi_engineering.yaml").read_text())
    config["performance"] = dict(enabled=True, warmup_updates=1, detail_updates=0)
    (root / "plan.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(root / "launcher.json", {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=REPO,text=True).strip(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "processes": 2, "sequence": ["continuous", "pause_after_A", "resume_B"]})
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1", USE_TF="0", PYTHONPATH=str(REPO))
    env.pop("ACCELERATE_USE_CPU", None)
    commands = []
    for name, pause, resume in (("continuous", None, None), ("resumed", 2, None), ("resumed", None, "latest")):
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(Path(__file__).resolve()), "worker", "--plan", str(root / "plan.yaml"), "--run-dir", str(root / name)]
        if pause:
            command += ["--stop-after-update", str(pause)]
        if resume:
            command += ["--resume", resume]
        log = root / f"process_{len(commands)}_{name}.log"
        start = time.monotonic()
        print(f"Launch {name} pause={pause} resume={resume}; log={log}", flush=True)
        with log.open("w") as stream:
            child = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        commands.append({"command": command, "exit_code": child.returncode, "seconds": time.monotonic()-start, "log": str(log)})
        write_json(root / "processes.json", commands)
        if child.returncode:
            raise RuntimeError(log.read_text()[-16000:])
    report(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "report"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
    elif args.mode == "report":
        report(args.output_dir.resolve())
    else:
        run(args.output_dir.resolve())
