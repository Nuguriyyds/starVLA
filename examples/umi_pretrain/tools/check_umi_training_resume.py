"""New-process acceptance for the SAME staged training entry, using CPU models.

Does not access UMI/video/model payloads. Results are written under --output-dir.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def assert_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_tree(x, y)
    else:
        assert left == right, (left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--performance", action="store_true", help="Also prove instrumentation on/off preserves exact training states")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = REPO / "examples/umi_pretrain/train_files/umi_training_tiny.yaml"
    unmeasured_plan = plan
    if args.performance:
        measured = yaml.safe_load(plan.read_text())
        measured["performance"] = dict(enabled=True, warmup_updates=2, detail_updates=2)
        plan = root / "measured_plan.yaml"
        plan.write_text(yaml.safe_dump(measured))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(REPO), ACCELERATE_USE_CPU="true")
    command = [sys.executable]
    if args.world_size > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.world_size}"]
    command += ["-m", "starVLA.training.train_umi_pretrain", "--cpu"]
    calls = []

    def launch(name, *, pause=None, resume=None, config=plan, expect_failure=False):
        cmd = command + ["--plan", str(config), "--output-dir", str(root / name)]
        if pause:
            cmd += ["--stop-after-update", str(pause)]
        if resume:
            cmd += ["--resume", str(resume)]
        log = root / f"process_{len(calls):02d}_{name}.log"
        with log.open("w") as stream:
            result = subprocess.run(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        calls.append({"command": cmd, "exit_code": result.returncode, "log": str(log)})
        print(f"{name} pause={pause} resume={resume} exit={result.returncode}", flush=True)
        if expect_failure:
            assert result.returncode != 0, f"Expected rejection: {log}"
        elif result.returncode:
            raise RuntimeError(log.read_text()[-10000:])

    launch("continuous")
    # Three separate process restarts cover A-mid, A->B boundary, and B-mid.
    for pause in (3, 6, 9, None):
        launch("resumed", pause=pause, resume="latest" if pause != 3 else None)
        if pause is not None:
            # Emulate an interrupted log append AFTER the durable checkpoint.
            for path in list((root / "resumed").glob("trace_rank_*.jsonl")) + [root / "resumed/updates.jsonl"]:
                with path.open("ab") as stream:
                    stream.write(b'{"global_update":')

    checkpoint = "checkpoints/update_00000012"
    filenames = ["pytorch_model.bin", "optimizer.bin", "custom_checkpoint_0.pkl", "custom_checkpoint_1.pkl"]
    filenames += [f"strict_rng_{r}.pt" for r in range(args.world_size)]
    for name in filenames:
        left = torch.load(root / "continuous" / checkpoint / name, map_location="cpu", weights_only=False)
        right = torch.load(root / "resumed" / checkpoint / name, map_location="cpu", weights_only=False)
        assert_tree(left, right)
    if args.performance:
        launch("instrumentation_disabled", config=unmeasured_plan)
        for name in filenames:
            assert_tree(torch.load(root / "continuous" / checkpoint / name, map_location="cpu", weights_only=False),
                        torch.load(root / "instrumentation_disabled" / checkpoint / name, map_location="cpu", weights_only=False))
    traces = []
    for rank in range(args.world_size):
        def read_trace(run):
            return [json.loads(line) for line in (root / run / f"trace_rank_{rank}.jsonl").read_text().splitlines()]
        left, right = read_trace("continuous"), read_trace("resumed")
        for a, b in zip(left, right):
            for key in ("global_update", "stage", "epoch_used", "samples", "lr_used", "lr_next", "progress", "loss"):
                assert_tree(a[key], b[key])
        assert len(left) == len(right) == 12
        traces.append(left)
    # Reconstruct each full global update from rank/micro-batch traces and match
    # the actual epoch permutation, rather than merely comparing two bad runs.
    from starVLA.dataloader.umi_sampler import UMIBlockShuffleSampler
    cfg = yaml.safe_load(plan.read_text())
    cursors = {}
    for step in range(12):
        event = traces[0][step]
        stage = next(x for x in cfg["stages"] if x["name"] == event["stage"])
        key = stage["name"], event["epoch_used"]
        cursor = cursors.get(key, 0)
        expected = list(UMIBlockShuffleSampler(stage["size"], block_size=7, seed=42, epoch=key[1]))
        observed = []
        for micro in range(2):
            for rank in range(args.world_size):
                observed += [x["dataset_index"] for x in traces[rank][step]["samples"][micro]]
        assert observed == expected[cursor:cursor + len(observed)]
        assert len(set(observed)) == len(observed)
        cursors[key] = cursor + len(observed)

    # A deliberately half-written checkpoint must never become latest.
    partial = root / "resumed/checkpoints/.update_00000099-partial-injected"
    partial.mkdir()
    torch.save({"not": "a full checkpoint"}, partial / "pytorch_model.bin")
    launch("resumed", resume=partial, expect_failure=True)
    assert json.loads((root / "resumed/latest.json").read_text())["step"] == 12
    launch("resumed", resume=root / "resumed" / checkpoint / "pytorch_model.bin", expect_failure=True)
    for key, change in (
        ("changed_plan", lambda p: p["stages"][0].update(updates=7)),
        ("changed_view", lambda p: p["stages"][0].update(offset=99)),
        ("formal_engineering", lambda p: p.update(purpose="formal")),
    ):
        changed = copy.deepcopy(cfg)
        change(changed)
        path = root / f"{key}.yaml"
        path.write_text(yaml.safe_dump(changed))
        launch("resumed", resume="latest", config=path, expect_failure=True)
    report = {"status": "passed", "world_size": args.world_size, "workers_per_rank": 2,
              "gradient_accumulation": 2, "updates": 12, "restart_boundaries": [3, 6, 9],
              "model_optimizer_scheduler_rng_bitwise_equal": True,
              "samples_match_global_sampler_exactly": True, "rejection_checks": 5, "processes": calls}
    report["instrumentation_on_off_bitwise_equal"] = bool(args.performance)
    (root / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "processes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
