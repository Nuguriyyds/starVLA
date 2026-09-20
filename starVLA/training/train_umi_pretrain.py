"""Staged UMI training with committed cursors and full process-restart recovery.

Launch with python -m starVLA.training.train_umi_pretrain --plan PLAN --output-dir RUN.
The immutable plan defines ALL optimizer updates. --stop-after-update is only a
process pause and does not shorten the scheduler. See umi_pretrain/training/README.md.
"""
import argparse
from copy import deepcopy
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, InitProcessGroupKwargs
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.umi_checkpoint import (
    UMICheckpoints, main_call, require_all, sha256_file, trim_uncommitted_log, write_json,
)
from starVLA.training.trainer_utils.umi_training_state import (
    UMITrainingState, checkpoint_config, checkpoint_reasons, fingerprint, validate_plan,
)
from starVLA.training.trainer_utils.umi_performance import Performance
from starVLA.training.trainer_utils.umi_training_data import (
    StageLoader, TinyModel, evaluate, inspect_views, make_loader,
)

REPO = Path(__file__).resolve().parents[2]


class UMIAccelerator(Accelerator):
    def unwrap_model(self, model, keep_fp32_wrapper=True, keep_torch_compile=True):
        # This entry explicitly supports plain/DDP only. Accelerate's generic
        # unwrap imports an installed DeepSpeed even during CPU-only tests,
        # which initializes this PPU image's Triton driver. Avoid that unrelated
        # optional backend; do not patch packages or suppress backend errors.
        if not keep_fp32_wrapper or not keep_torch_compile:
            raise ValueError("Wrapper removal/export requires a separate export path")
        while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            model = model.module
        return model


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", help="latest, or an explicit complete checkpoint directory")
    parser.add_argument("--stop-after-update", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def code_identity():
    # Content identity covers dirty development code as well as committed code.
    paths = sorted((REPO / "starVLA").rglob("*.py"))
    digest = fingerprint({str(p.relative_to(REPO)): sha256_file(p) for p in paths})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    return {"commit": commit, "source_sha256": digest}


def remove_location_fields(value):
    if isinstance(value, dict):
        return {k: remove_location_fields(v) for k, v in value.items() if k not in {
            "index_dir", "source_root", "allowed_video_roots", "statistics_path", "path",
            "normalization_statistics", "normalization_experiment_contract", "base_vlm",
        }}
    if isinstance(value, list):
        return [remove_location_fields(v) for v in value]
    return value


def build_identity(plan, records, accelerator):
    canonical = remove_location_fields(plan)
    canonical["views"] = [remove_location_fields(r) for r in records]
    # Field/row identities, rules, representation and normalization hashes are in
    # records; filesystem relocation itself need not change experiment identity.
    model_artifacts = {}
    if plan["model_kind"] == "qwenpi":
        model_root = Path(plan["framework"]["qwenvl"]["base_vlm"])
        if not (model_root / "config.json").is_file():
            raise ValueError("QwenPI requires a complete local model directory")
        for file in sorted(model_root.iterdir()):
            if file.is_file() and file.suffix in (".json", ".safetensors", ".bin", ".txt"):
                model_artifacts[file.name] = sha256_file(file)
    versions = {}
    for name in ("torch", "accelerate", "numpy", "transformers", "diffusers", "peft", "pyarrow", "av"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"version": "umi-run-identity-v1", "plan": canonical,
            "plan_fingerprint": fingerprint(canonical), "code": code_identity(),
            "model_artifacts": model_artifacts,
            "runtime": {"world_size": accelerator.num_processes,
                        "backend": str(accelerator.distributed_type),
                        "collective_backend": dist.get_backend() if dist.is_initialized() else None,
                        "device_type": accelerator.device.type,
                        "device_name": torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == "cuda" else "cpu",
                        "batch_size": plan["training"]["batch_size"],
                        "accumulation": plan["training"]["gradient_accumulation_steps"],
                        "precision": accelerator.mixed_precision, "versions": versions}}


def build_model(plan):
    if plan["model_kind"] == "tiny":
        return TinyModel()
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

    class QwenPITraining(Qwen_PI):
        def _encode_vl_hidden_states(self, *args, **kwargs):
            hidden, mask = super()._encode_vl_hidden_states(*args, **kwargs)
            # Preserve the already-verified FP32 action-head bridge. This casts
            # hidden states without detaching the pretrained VLM gradient path.
            return [x.float() for x in hidden], mask

    # The VLM prompt builder also consults datasets.vla_data (e.g. CoT_prompt);
    # inference reads its optional image resize setting. Keep this shared data
    # interface present, without binding the model to one stage's index path.
    cfg = OmegaConf.create({"framework": plan["framework"],
                            "datasets": {"vla_data": dict(plan["data"], include_state=True)}})
    model = QwenPITraining(cfg).float()
    model.qwen_vl_interface.model.config.use_cache = False
    if not (model.action_model.model.config.interleave_self_attention
            and model.action_model.model.config.use_canonical_forward):
        raise ValueError("Require the previously verified canonical state/action path")
    return model


def build_optimizer(model, train):
    frozen = train.get("freeze_prefixes", [])
    for prefix in frozen:
        if not any(name.startswith(prefix) for name, _ in model.named_parameters()):
            raise ValueError(f"Unknown freeze prefix: {prefix}")
    grouped, names = {}, {}
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen):
            parameter.requires_grad_(False)
        if not parameter.requires_grad:
            continue
        group = "vlm" if name.startswith("qwen_vl_interface.") else "other"
        grouped.setdefault(group, []).append(parameter)
        names.setdefault(group, []).append([name, list(parameter.shape)])
    if not grouped:
        raise ValueError("No trainable parameters")
    groups = [{"params": params, "lr": train.get("vlm_lr", train["lr"]) if group == "vlm" else train["lr"],
               "group_name": group} for group, params in grouped.items()]
    optimizer = torch.optim.AdamW(groups, lr=train["lr"],
                                 betas=tuple(train.get("betas", [.9, .999])),
                                 eps=train.get("eps", 1e-8), weight_decay=train.get("weight_decay", .01),
                                 fused=False, foreach=False)
    return optimizer, {"parameter_groups": names, "freeze_prefixes": frozen,
                       "optimizer": "AdamW", "fused": False, "foreach": False}


def seed_start(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(args, *, on_start=None, evaluation_fn=None):
    plan = OmegaConf.to_container(OmegaConf.load(args.plan), resolve=True)
    total_updates = validate_plan(plan)
    train = plan["training"]
    saving = checkpoint_config(plan)
    if args.stop_after_update is not None and not 0 < args.stop_after_update <= total_updates:
        raise ValueError("Pause step must be within the unchanged full training plan")
    handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Passing a non-null backend to a standalone CPU Accelerator can leave
        # its state expecting a process group that was never initialized.
        handlers.append(InitProcessGroupKwargs(backend="gloo" if args.cpu else "nccl"))
    accelerator = UMIAccelerator(
        cpu=args.cpu, mixed_precision=train.get("mixed_precision", "no"),
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=train["gradient_accumulation_steps"], sync_with_dataloader=False),
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False,
                                                  even_batches=False, use_seedable_sampler=False),
        rng_types=[], step_scheduler_with_optimizer=False,
        kwargs_handlers=handlers,
    )
    # ZeRO/FSDP checkpoint layouts and scheduler ownership need their own backend
    # acceptance. Refuse unsupported launch modes instead of silently mis-saving.
    if str(accelerator.distributed_type).split(".")[-1] not in ("NO", "MULTI_CPU", "MULTI_GPU"):
        raise ValueError("This checkpoint version supports single-device/DDP only; ZeRO/FSDP require backend validation")
    if accelerator.num_processes != int(os.environ.get("WORLD_SIZE", "1")):
        raise ValueError("Distributed runtime does not match launcher WORLD_SIZE")
    run_dir = args.output_dir.resolve()
    perf = Performance(plan.get("performance"), accelerator)
    perf.timing_probe()
    lock = None

    def open_run():
        nonlocal lock
        if args.resume:
            if not (run_dir / "run_identity.json").is_file():
                raise ValueError("Resume requires an existing run_identity.json; no fresh-start fallback")
        elif run_dir.exists():
            raise ValueError("New run requires a new output directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        if os.name != "posix":
            raise ValueError("This entry currently requires POSIX shared-filesystem locking")
        import fcntl
        lock = (run_dir / "run.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    main_call(accelerator, open_run)
    loader = None
    try:
        accelerator.print("Inspecting data identities and immutable training plan", flush=True)
        records = perf.call("inspect_views", main_call, accelerator, lambda: inspect_views(plan, run_dir))
        identity = perf.call("build_identity", main_call, accelerator, lambda: build_identity(plan, records, accelerator))
        if args.resume:
            def check_before_model():
                saved = json.loads((run_dir / "run_identity.json").read_text())
                base = {k: v for k, v in saved.items() if k not in ("parameter_policy", "resolved_model_config")}
                if base != identity:
                    raise ValueError("Run identity changed; resume rejected before model allocation")
            main_call(accelerator, check_before_model)
        global_batch = accelerator.num_processes * train["batch_size"] * train["gradient_accumulation_steps"]
        progress = UMITrainingState([s["updates"] for s in plan["stages"]],
                                    [r["num_windows"] for r in records[:len(plan["stages"])]], global_batch)
        # Initialize only before constructing the model. load() subsequently
        # restores rank RNG; no seed_start call is permitted after that point.
        seed_start(int(train.get("seed", 42)))
        accelerator.print("Building model/optimizer once for the whole plan", flush=True)
        model = perf.call("build_model", build_model, plan)
        optimizer, groups = perf.call("build_optimizer", build_optimizer, model, train)
        identity["parameter_policy"] = groups
        if hasattr(model, "config"):
            identity["resolved_model_config"] = remove_location_fields(OmegaConf.to_container(model.config, resolve=True))

        def persist_identity():
            identity_path = run_dir / "run_identity.json"
            if args.resume:
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError("Run identity changed; resume rejected")
            else:
                write_json(identity_path, identity)
                write_json(run_dir / "plan_requested.json", plan)
                write_json(run_dir / "data_access" / "all_views.json", records)
                (run_dir / "exports").mkdir()
                if plan["model_kind"] == "qwenpi" and plan["data"].get("normalization_statistics"):
                    path = Path(plan["data"]["normalization_statistics"])
                    # Persist the exact bytes, including provenance, once per run.
                    target = run_dir / "normalization" / "statistics.json"
                    target.parent.mkdir()
                    target.write_bytes(path.read_bytes())
                    if sha256_file(target) != sha256_file(path):
                        raise ValueError("Statistics snapshot mismatch")
                    contract = plan["data"].get("normalization_experiment_contract")
                    if contract:
                        (target.parent / "experiment_contract.json").write_bytes(Path(contract).read_bytes())
        main_call(accelerator, persist_identity)
        if accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        warmup = train.get("warmup_updates", 0)

        def multiplier(step):
            if step < warmup:
                return (step + 1) / warmup
            return max(0., (total_updates - step) / (total_updates - warmup))

        # Scheduler is NOT passed to prepare(): this loop is its single owner.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        model, optimizer = perf.call("prepare_model_optimizer", accelerator.prepare, model, optimizer)
        accelerator.register_for_checkpointing(progress, scheduler)
        checkpoints = UMICheckpoints(accelerator, run_dir, identity, progress,
                                     performance=perf, integrity=saving["integrity"])
        resolved = perf.call("resume_verify", checkpoints.resolve, args.resume) if args.resume else None
        if resolved:
            progress.load_state_dict(resolved["state"])

        def construct_loader():
            stage = plan["stages"][progress.stage_index]
            raw = perf.call("loader_construct_or_replay_preload", make_loader, plan, stage, run_dir)
            expected = records[progress.stage_index]["view_fingerprint"]
            if raw.dataset.provenance()["view_fingerprint"] != expected:
                raise ValueError("Stage view changed after preflight")
            return StageLoader(accelerator, raw, progress.stage_index, train.get("seed", 42), performance=perf)

        if not progress.done:
            loader = construct_loader()
        if resolved:
            accelerator.print(f"Loading full checkpoint {resolved['path']}", flush=True)
            perf.call("resume_load", checkpoints.load, resolved, loader.generator if loader else None)
        # A fresh run uses rank-specific model-noise streams after identical
        # parameter initialization and DDP broadcast.
        else:
            seed_start(train.get("seed", 42) + accelerator.process_index)
        # Optional task-specific evaluation. Restored state is available here;
        # callbacks must preserve RNG and model mode and must not update weights.
        if on_start is not None:
            on_start(accelerator, model, plan, run_dir, progress.global_update_step)
        if args.stop_after_update is not None and args.stop_after_update <= progress.global_update_step:
            raise ValueError("Pause target must exceed the restored committed step")

        stopping = {"requested": False}
        old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: stopping.update(requested=True))
        trace_path = run_dir / f"trace_rank_{accelerator.process_index}.jsonl"
        # Records beyond the durable checkpoint belong to lost updates; remove
        # them on resume so engineering traces reflect the committed history.
        if args.resume:
            repairs = {"trace": trim_uncommitted_log(trace_path, progress.global_update_step)}
            if accelerator.is_main_process:
                repairs["updates"] = trim_uncommitted_log(run_dir / "updates.jsonl", progress.global_update_step)
            write_json(run_dir / f"log_recovery_rank_{accelerator.process_index}.json", repairs)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        try:
            while not progress.done:
                stage_i = progress.stage_index
                before = deepcopy(progress.current())
                if loader is None or loader.stage_index != stage_i:
                    perf.suspend()
                    if loader is not None:
                        perf.call("loader_close_stage", loader.close)
                    loader = construct_loader()
                if loader.iterator is None or loader.epoch != before["epoch"]:
                    perf.suspend()
                    iterator_error = None
                    try:
                        perf.call("loader_start_first_batch", loader.start, before["epoch"], before["cursor"])
                    except Exception as error:
                        iterator_error = str(error)
                    require_all(accelerator, iterator_error is None, f"Iterator initialization failed: {iterator_error}")
                lr_used = [group["lr"] for group in optimizer.param_groups]
                perf.begin_update()
                loss_sum, data_wait, identities = 0., 0., []
                for micro in range(train["gradient_accumulation_steps"]):
                    start = time.monotonic()
                    batch_error = None
                    try:
                        batch = perf.call("local_batch_wait", loader.next)
                    except Exception as error:
                        batch_error = f"{type(error).__name__}: {error}"
                    data_wait += time.monotonic() - start
                    with perf.span("batch_rank_checks"):
                        require_all(accelerator, batch_error is None, f"Data read failed on a rank: {batch_error}")
                        require_all(accelerator, len(batch) == train["batch_size"], "Incomplete micro-batch")
                    identities.append([{k: sample["umi_metadata"].get(k) for k in
                                        ("dataset_index", "episode_index", "frame_index", "view_fingerprint")}
                                       for sample in batch])
                    with accelerator.accumulate(model):
                        with perf.span("forward_including_processor_transfer"):
                            loss = model(examples=batch)["action_loss"]
                        with perf.span("loss_rank_check"):
                            require_all(accelerator, loss.ndim == 0 and bool(torch.isfinite(loss)),
                                        "Nonfinite loss; no cursor/checkpoint commit")
                        perf.call("backward", accelerator.backward, loss)
                        loss_sum += float(loss.detach())
                        expected_sync = micro == train["gradient_accumulation_steps"] - 1
                        if accelerator.sync_gradients != expected_sync:
                            raise RuntimeError("Unexpected accumulation boundary")
                        if accelerator.sync_gradients:
                            grad_norm = perf.call("clip_grad", accelerator.clip_grad_norm_, model.parameters(), train["max_grad_norm"])
                            require_all(accelerator, bool(torch.isfinite(grad_norm)),
                                        "Nonfinite gradients; no cursor/checkpoint commit")
                            perf.call("optimizer", optimizer.step)
                            require_all(accelerator, not accelerator.optimizer_step_was_skipped,
                                        "Optimizer update skipped; no scheduler/cursor commit")
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                progress.commit_update()
                # Committed global cursor, never the prefetched iterator cursor.
                loader.sampler.set_start_index(progress.stages[stage_i]["cursor"])
                step = progress.global_update_step
                perf.end_update(step, global_batch)
                loss_mean = torch.tensor(loss_sum / train["gradient_accumulation_steps"], device=accelerator.device)
                if dist.is_initialized():
                    dist.all_reduce(loss_mean)
                    loss_mean /= accelerator.num_processes
                event = {"global_update": step, "stage": plan["stages"][stage_i]["name"],
                         "stage_update": progress.stages[stage_i]["updates"], "epoch_used": before["epoch"],
                         "progress": progress.state_dict(), "lr_used": lr_used,
                         "lr_next": [g["lr"] for g in optimizer.param_groups],
                         "loss": float(loss_mean), "grad_norm_before_clip": float(grad_norm),
                         "gradients_finite": True, "data_wait_seconds": data_wait}
                if train.get("trace_samples", False):
                    with trace_path.open("a") as stream:
                        stream.write(json.dumps(dict(event, rank=accelerator.process_index, samples=identities)) + "\n")
                if train.get("eval_every", 0) and step % train["eval_every"] == 0:
                    perf.suspend()
                    eval_error = None
                    try:
                        if evaluation_fn is None:
                            event["evaluation"] = perf.call("evaluation", evaluate, accelerator, model, plan, run_dir)
                        else:
                            event["evaluation"] = perf.call("evaluation", evaluation_fn, accelerator, model, plan, run_dir, step)
                    except Exception as error:
                        eval_error = str(error)
                    require_all(accelerator, eval_error is None, f"Evaluation failed: {eval_error}")
                stop_flag = torch.tensor(int(stopping["requested"] or perf.expired()), device=accelerator.device)
                if dist.is_initialized():
                    dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                pause = bool(stop_flag.item()) or args.stop_after_update == step
                changed_stage = progress.stage_index != stage_i
                reasons = checkpoint_reasons(step, saving["every_updates"], pause=pause,
                                             complete=progress.done, stage_boundary=changed_stage)
                event["checkpoint_reasons"] = reasons
                if reasons:
                    perf.suspend()
                    accelerator.print(f"Saving complete checkpoint at update {step}", flush=True)
                    perf.call("checkpoint_total", checkpoints.save, loader.generator, reasons=reasons)
                event["latest_complete_checkpoint"] = checkpoints.latest
                if accelerator.is_main_process:
                    with (run_dir / "updates.jsonl").open("a") as stream:
                        stream.write(json.dumps(event) + "\n")
                    write_json(run_dir / "progress.json", dict(event, status="complete" if progress.done else "paused" if pause else "running",
                               peak_allocated_memory_gib=torch.cuda.max_memory_allocated(accelerator.device) / 2**30
                               if accelerator.device.type == "cuda" else None))
                    print(f"update={step}/{total_updates} stage={event['stage']} loss={float(loss_mean):.6f} "
                          f"next_lr={event['lr_next']} cursor={progress.stages[stage_i]['cursor']}", flush=True)
                if pause:
                    break
            accelerator.wait_for_everyone()
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        accelerator.print(f"Training {'complete' if progress.done else 'paused'}: {run_dir}", flush=True)
    finally:
        perf.finish(run_dir / f"performance_rank_{accelerator.process_index}_{os.getpid()}.json")
        if loader is not None:
            loader.close()
        if lock is not None:
            lock.close()
        accelerator.end_training()


if __name__ == "__main__":
    run(arguments())
