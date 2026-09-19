"""Private QwenPI smoke training for raw FP32 world-pose UMI windows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO.parent / "datasets/roban_umi_debug")
    parser.add_argument("--model", type=Path, default=REPO.parent / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, help="New directory underneath the private runs directory")
    parser.add_argument("--max-steps", type=positive_int, default=20)
    parser.add_argument("--save-every", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-only", action="store_true", help="Audit raw rows and sample the actual loader without importing a model")
    return parser.parse_args(argv)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_sample(sample, label, horizon=16):
    import numpy as np
    from PIL import Image

    if horizon != 16:
        raise ValueError(f"{label}: model horizon must equal 16, got {horizon}")
    # Refuse a rounded FP16 array instead of casting it back to FP32 here.
    for key, shape in (("state", (1, 16)), ("action", (horizon, 16))):
        value = sample.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{label}: {key} must arrive as a raw float32 numpy array")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{label}: invalid {key} shape/values: {value.shape}")
    images = sample.get("image", [])
    if len(images) != 4 or not all(isinstance(image, Image.Image) for image in images):
        raise ValueError(f"{label}: expected four decoded PIL images")
    if not isinstance(sample.get("lang"), str) or not sample["lang"].strip():
        raise ValueError(f"{label}: expected a nonempty task string")


def model_identity(model_path):
    identity = {"path": str(model_path), "revision": None}
    config_path = model_path / "config.json"
    if config_path.is_file():
        identity["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        identity["revision"] = model_config.get("_commit_hash")
    revision_path = model_path / "download_revision.txt"
    if not identity["revision"] and revision_path.is_file():
        recorded_revision = revision_path.read_text(encoding="utf-8").strip()
        if recorded_revision:
            identity["revision"] = recorded_revision
            identity["revision_source"] = str(revision_path)
    parts = model_path.parts
    if not identity["revision"] and "snapshots" in parts:
        position = parts.index("snapshots")
        if position + 1 < len(parts):
            identity["revision"] = parts[position + 1]
    if not identity["revision"]:
        identity["revision_note"] = "No revision is recorded in local config, download_revision.txt, or snapshot path"
    return identity


def audit_data(args, cfg):
    import numpy as np
    from torch.utils.data import Subset
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    from umi_valid_windows import HORIZON, VIDEO_KEYS, dataset_indices, load_private_umi, raw_window_statistics

    action_cfg = cfg.framework.action_model
    if (action_cfg.state_dim, action_cfg.action_dim, action_cfg.action_horizon) != (16, 16, HORIZON):
        raise ValueError("Expected state_dim=action_dim=16 and action_horizon=16")
    registry_cfg = ROBOT_TYPE_CONFIG_MAP["roban_umi"]
    if list(registry_cfg.state_indices) != [0] or list(registry_cfg.action_indices) != list(range(1, HORIZON + 1)):
        raise ValueError("Registry must use current state and future offsets 1..16")
    if list(registry_cfg.observation_indices) != [0] or len(registry_cfg.video_keys) != 4:
        raise ValueError("Registry must use four current-frame views")

    # Audit every source row and reject invalid/incomplete windows BEFORE
    # constructing the loader or any model. The raw parquet is never rewritten.
    source = load_private_umi(args.dataset, horizon=HORIZON)
    if not source["valid_steps"]:
        raise ValueError("No valid complete 16-frame future windows")
    cfg.datasets.vla_data.raw_lowdim_statistics = raw_window_statistics(source)
    dataset = make_LeRobotSingleDataset(
        data_root_dir=args.dataset.parent,
        data_name=args.dataset.name,
        robot_type="roban_umi",
        delete_pause_frame=False,
        data_cfg=cfg.datasets.vla_data,
    )
    allowed_indices = dataset_indices(dataset, source["valid_steps"])
    if len(allowed_indices) != len(source["valid_steps"]) or len(set(allowed_indices)) != len(allowed_indices):
        raise ValueError("The valid windows could not be mapped uniquely onto dataset indices")
    subset = Subset(dataset, allowed_indices)

    mapping = source["mapping"]
    observed_video_keys = [
        mapping["video"][key.split(".", 1)[1]]["original_key"]
        for key in registry_cfg.video_keys
    ]
    if observed_video_keys != list(VIDEO_KEYS):
        raise ValueError("Four-view order differs from the audited contract")

    raw = source["raw"]
    episode_rows = {}
    for row_index, episode_index in enumerate(raw["episode_index"]):
        episode_rows.setdefault(int(episode_index), []).append(row_index)
    checks = []
    for position in sorted({0, min(100, len(subset) - 1), len(subset) - 1}):
        episode, row_offset = source["valid_steps"][position]
        sample = subset[position]
        check_sample(sample, f"preflight[{position}]")
        rows = episode_rows[episode]
        state_rows = [rows[row_offset]]
        action_rows = rows[row_offset + 1:row_offset + HORIZON + 1]
        frame = int(raw.iloc[state_rows[0]]["frame_index"])
        np.testing.assert_array_equal(sample["state"], source["values"][state_rows])
        np.testing.assert_array_equal(sample["action"], source["values"][action_rows])
        task_index = int(raw.iloc[state_rows[0]]["task_index"])
        if sample["lang"] != source["tasks"][task_index]:
            raise ValueError(f"preflight[{position}]: task text does not match the source")
        checks.append({
            "subset_index": position,
            "dataset_index": allowed_indices[position],
            "episode_index": episode,
            "frame_index": frame,
            "episode_row_index": row_offset,
            "future_frame_indices": [int(value) for value in raw.iloc[action_rows]["frame_index"]],
            "state_shape": list(sample["state"].shape),
            "action_shape": list(sample["action"].shape),
            "dtype": str(sample["state"].dtype),
            "image_sizes": [list(image.size) for image in sample["image"]],
            "exact_raw_fp32_match": True,
        })
    return subset, source, allowed_indices, checks


def gradient_probe(named_parameters, group_name):
    """Check a whole group's gradient norm; snapshot <=32 selected parameter entries."""
    import torch

    active = [(name, param) for name, param in named_parameters if param.grad is not None]
    if not active:
        raise RuntimeError(f"{group_name}: no gradients")
    norms = torch.stack([param.grad.detach().norm() for _, param in active])
    group_norm = norms.norm()
    if not torch.isfinite(group_norm).item() or group_norm.item() == 0:
        raise RuntimeError(f"{group_name}: gradient norm is nonfinite or zero")
    # A small nonzero-gradient tensor keeps update probes inexpensive.
    norm_values = norms.detach().cpu().tolist()
    name, parameter = min(
        (item for item, norm in zip(active, norm_values) if norm > 0),
        key=lambda item: item[1].numel(),
    )
    flat_gradient = parameter.grad.detach().reshape(-1)
    count = min(32, flat_gradient.numel())
    indices = flat_gradient.abs().topk(count).indices
    before = parameter.detach().reshape(-1)[indices].clone()
    return {
        "group": group_name,
        "gradient_norm_before_clip": float(group_norm.item()),
        "gradient_tensor_count": len(active),
        "probe_parameter": name,
        "probe_indices": indices.detach().cpu().tolist(),
        "parameter": parameter,
        "indices": indices,
        "before": before,
    }


def verify_update(probe):
    import torch

    parameter = probe.pop("parameter")
    indices = probe.pop("indices")
    before = probe.pop("before")
    after = parameter.detach().reshape(-1)[indices]
    if not torch.isfinite(after).all().item():
        raise RuntimeError(f"{probe['group']}: update probe contains NaN/Inf")
    max_delta = (after - before).abs().max().item()
    if max_delta == 0:
        raise RuntimeError(f"{probe['group']}: no change in the selected parameter entries")
    probe["max_abs_parameter_delta"] = max_delta
    probe["updated_entries"] = int(torch.count_nonzero(after - before).item())
    probe["checked_entries"] = len(probe["probe_indices"])
    return probe


def train(args, cfg, subset, run_dir, progress, metadata):
    # Deliberately deferred: --data-only never imports the model/Transformers.
    import torch
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import collate_fn
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

    class QwenPITrainCheck(Qwen_PI):
        # Preserve the existing smoke entry's precision bridge and gradient graph.
        def _encode_vl_hidden_states(self, *call_args, **call_kwargs):
            hidden_states, attention_mask = super()._encode_vl_hidden_states(*call_args, **call_kwargs)
            return [hidden.float() for hidden in hidden_states], attention_mask

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The smoke training entry requires an available CUDA device")
    if not args.model.is_dir():
        raise FileNotFoundError(f"Local model directory is missing: {args.model}")
    torch.manual_seed(args.seed)
    train_loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    print("加载模型；保持既有 FP32 参数与 VLM 内部 autocast 设置", flush=True)
    model = QwenPITrainCheck(cfg).to(device=args.device, dtype=torch.float32)
    actual_action_cfg = model.config.framework.action_model
    if int(model.action_horizon) != 16 or (
        actual_action_cfg.state_dim, actual_action_cfg.action_dim,
        actual_action_cfg.action_horizon,
    ) != (16, 16, 16):
        raise ValueError("The constructed model must use state/action dimensions 16 and horizon 16")
    dit = model.action_model.model
    if not (dit.config.interleave_self_attention and dit.config.use_canonical_forward):
        raise ValueError("UMI debug training requires canonical cross/self interleaving")
    print("动作头模式：canonical；偶数层 cross、奇数层 self", flush=True)
    model.qwen_vl_interface.model.config.use_cache = False
    OmegaConf.save(model.config, run_dir / "config.yaml")
    revision = getattr(model.qwen_vl_interface.model.config, "_commit_hash", None)
    if revision:
        metadata["model"]["revision"] = revision
        metadata["model"].pop("revision_note", None)
    write_json(run_dir / "run_metadata.json", metadata)
    parameter_groups = {
        "vlm": list(model.qwen_vl_interface.named_parameters()),
        "action": list(model.action_model.named_parameters()),
        "state_encoder": list(model.action_model.state_encoder.named_parameters()),
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": model.qwen_vl_interface.parameters(), "lr": 1e-5},
            {"params": model.action_model.parameters(), "lr": 1e-4},
        ],
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        fused=False, foreach=False,
    )
    model.train()
    iterator = iter(train_loader)
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        for index, sample in enumerate(batch):
            check_sample(sample, f"step={step}, sample={index}", horizon=int(model.action_horizon))
        optimizer.zero_grad(set_to_none=True)
        loss = model(examples=batch)["action_loss"]
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise RuntimeError(f"step={step}: loss is nonscalar or nonfinite")
        loss.backward()
        probes = [gradient_probe(parameters, name) for name, parameters in parameter_groups.items()]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True, foreach=False,
        )
        optimizer.step()
        updates = [verify_update(probe) for probe in probes]
        loss_value = float(loss.detach().item())
        progress.update({
            "status": "training",
            "step": step,
            "last_train_loss": loss_value,
            "peak_allocated_memory_gib": torch.cuda.max_memory_allocated(args.device) / 1024**3,
        })
        progress["steps"].append({
            "step": step, "loss": loss_value,
            "gradient_norm_before_clip": float(grad_norm.item()),
            "groups": updates,
        })
        write_json(run_dir / "progress.json", progress)
        if step == 1 or step % 10 == 0 or step == args.max_steps:
            print(f"step={step}/{args.max_steps} loss={loss_value:.6f} grad_norm={grad_norm.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            checkpoint = run_dir / "pytorch_model.pt"
            temporary = run_dir / "pytorch_model.pt.tmp"
            print(f"step={step}，保存权重：{checkpoint}", flush=True)
            torch.save(model.state_dict(), temporary)
            temporary.replace(checkpoint)
            progress["checkpoint_step"] = step
            progress["checkpoint_path"] = str(checkpoint)
            write_json(run_dir / "progress.json", progress)
    progress["status"] = "completed"
    write_json(run_dir / "progress.json", progress)


def make_debug_config(dataset, model):
    """One explicit data/attention configuration for training and state checks."""
    from omegaconf import OmegaConf
    from umi_valid_windows import HORIZON

    dataset, model = Path(dataset).resolve(), Path(model).resolve()
    return OmegaConf.create({
        "framework": {
            "name": "QwenPI",
            "qwenvl": {"base_vlm": str(model), "attn_implementation": "sdpa"},
            "action_model": {
                "action_dim": 16, "state_dim": 16, "action_horizon": HORIZON,
                "diffusion_model_cfg": {
                    "interleave_self_attention": True,
                    "use_canonical_forward": True,
                },
            },
        },
        "datasets": {"vla_data": {
            "lerobot_version": "v3.0", "include_state": True,
            "action_mode": "abs", "video_backend": "torchvision_av",
            "lowdim_dtype": "float32", "strict_window_sampling": True,
            "data_root_dir": str(dataset.parent), "data_mix": dataset.name,
        }},
    })


def main(argv=None):
    args = parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.model = args.model.resolve()
    from omegaconf import OmegaConf
    from umi_valid_windows import DATA_VERSION, HORIZON, SIGNAL_COLUMNS, VIDEO_KEYS

    cfg = make_debug_config(args.dataset, args.model)
    print(f"审计原始 FP32 数据与完整未来窗口：{args.dataset}", flush=True)
    subset, source, allowed_indices, checks = audit_data(args, cfg)
    runs_root = (REPO.parent / "runs").resolve()
    default_name = "qwenpi_" + DATA_VERSION + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (args.output_dir or runs_root / default_name).resolve()
    if not run_dir.is_relative_to(runs_root) or run_dir == runs_root:
        raise ValueError(f"--output-dir must be a new private subdirectory of {runs_root}")
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, run_dir / "config_requested.yaml")
    metadata = {
        "data_version": DATA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "mode": "data_only" if args.data_only else "training",
        "model": model_identity(args.model),
        "contract": {
            "signal_columns": list(SIGNAL_COLUMNS), "video_keys": list(VIDEO_KEYS),
            "state_shape": [1, 16], "action_shape": [HORIZON, 16],
            "action_offsets": list(range(1, HORIZON + 1)),
            "lowdim_dtype": "float32", "normalization": "none", "action_mode": "abs",
            "mapping": source["mapping"], "mapping_sha256": fingerprint(source["mapping"]),
            "valid_windows_only": True,
        },
        "allowed_dataset_indices": allowed_indices,
        "allowed_indices_sha256": fingerprint(allowed_indices),
        "valid_steps": source["valid_steps"],
        "valid_steps_sha256": fingerprint(source["valid_steps"]),
        "window_report": source["report"],
        "preflight_samples": checks,
        "training": {
            "max_steps": args.max_steps, "batch_size": args.batch_size,
            "num_workers": args.num_workers, "seed": args.seed, "device": args.device,
            "save_every": args.save_every,
            "action_forward": {
                "reference_commit": "c521decb7441c7dfea282c61dc758456bcffbb8f",
                "interleave_self_attention": True, "use_canonical_forward": True,
            },
            "precision": "FP32 parameters and action hidden states; existing QwenPI VLM BF16 autocast retained",
        },
        "verification_scope": (
            "All raw rows/windows are audited; representative loader samples match raw FP32 exactly. "
            "Training checks finite loss, finite nonzero gradient norm per group, and <=32 parameter "
            "entries from one nonzero-gradient tensor in each VLM/action/state_encoder group per step. "
            "This is a limited update probe, not a state-conditioning causal test or task-quality evaluation."
        ),
    }
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "valid_windows.json", source["report"])
    progress = {
        "status": "data_only_passed" if args.data_only else "data_validated",
        "step": 0, "max_steps": args.max_steps, "checkpoint_step": None,
        "dataset_size": len(subset), "steps": [],
    }
    write_json(run_dir / "progress.json", progress)
    print(f"有效训练位置：{len(subset)}；输出目录：{run_dir}", flush=True)
    if args.data_only:
        print("数据审计通过；未加载模型、未运行训练。", flush=True)
        return
    try:
        train(args, cfg, subset, run_dir, progress, metadata)
    except Exception as error:
        progress.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(run_dir / "progress.json", progress)
        raise
    print(f"训练完成：{run_dir}", flush=True)


if __name__ == "__main__":
    main()