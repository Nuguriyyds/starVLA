"""Engineering-only normalized UMI -> QwenPI backward/sampling interface check.

No optimizer step, checkpoint write, source rewrite, or statistics fitting is
performed. Metrics describe tensor connectivity, not policy quality.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=REPO.parent / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed-video-root", type=Path, action="append", default=None)
    return parser.parse_args(argv)


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_fp32(values, shape, label):
    import numpy as np

    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32"):
        raise TypeError(f"{label}: expected a float32 numpy array")
    if tuple(values.shape) != tuple(shape) or not np.isfinite(values).all():
        raise ValueError(f"{label}: invalid shape or NaN/Inf: {values.shape}")


def tensor_summary(values):
    import numpy as np

    values64 = np.asarray(values, dtype=np.float64)
    return {"shape": list(values.shape), "dtype": str(values.dtype),
            "min": float(values64.min()), "max": float(values64.max()),
            "rms": float(np.sqrt(np.mean(values64 ** 2)))}


def run(args, run_dir, report):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from starVLA.dataloader.umi_indexed_dataset import make_umi_dataloader
    from train_qwenpi_debug import check_sample, gradient_probe, make_debug_config, model_identity

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This model connectivity check requires an available CUDA/PPU device")
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"A complete local model directory is required: {args.model}")
    if not args.statistics.is_file():
        raise FileNotFoundError(args.statistics)
    selected_device = torch.device(args.device)
    torch.cuda.set_device(selected_device.index if selected_device.index is not None else torch.cuda.current_device())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = make_debug_config(args.index_dir, args.model)
    allowed_roots = args.allowed_video_root or [
        Path("/mnt/nas/public/roban_umi/restricted_data/source/umi/videos")
    ]
    cfg.output_dir = str(run_dir)
    cfg.datasets.vla_data = {
        "dataset_py": "umi_indexed", "data_mix": "normalization_engineering_check",
        "index_dir": str(args.index_dir), "include_state": True,
        "allowed_video_roots": [str(path.resolve()) for path in allowed_roots],
        "normalization": "mean_std", "normalization_statistics": str(args.statistics),
        "normalization_purpose": "engineering", "image_size": [224, 224],
        "per_device_batch_size": args.samples, "num_workers": 0,
        "drop_last": False, "shuffle": False, "seed": args.seed,
        "return_metadata": True, "pin_memory": False,
    }
    OmegaConf.save(cfg, run_dir / "config_requested.yaml")
    print("[1/4] 读取少量真实数据，核对归一化接口和正反变换", flush=True)
    loader = make_umi_dataloader(cfg)
    dataset = loader.dataset
    if not hasattr(dataset, "normalizer") or not hasattr(dataset, "raw_dataset"):
        raise TypeError("Expected the explicit normalized Dataset wrapper")
    if len(dataset) < args.samples:
        raise ValueError("The requested engineering view does not contain enough windows")
    samples = next(iter(loader))
    if len(samples) != args.samples:
        raise ValueError("Unexpected engineering batch length")
    raw_state, raw_action, checks = [], [], []
    for index, sample in enumerate(samples):
        check_sample(sample, f"normalized sample {index}")
        raw = dataset.raw_dataset.read_lowdim(index)
        trace = sample.get("umi_metadata", {})
        if trace.get("dataset_index") != index:
            raise ValueError("Sequential loader did not deliver the expected indexed window")
        for key in ("state", "action"):
            normalize = getattr(dataset.normalizer, f"normalize_{key}")
            inverse = getattr(dataset.normalizer, f"inverse_{key}")
            np.testing.assert_array_equal(sample[key], normalize(raw[key]))
            restored = inverse(sample[key])
            check_fp32(restored, raw[key].shape, f"inverse {key}")
            np.testing.assert_allclose(restored, raw[key], rtol=1e-5, atol=1e-6)
            # The wrapper must not mutate any source/cache-backed arrays.
            np.testing.assert_array_equal(dataset.raw_dataset.read_lowdim(index)[key], raw[key])
        raw_state.append(raw["state"])
        raw_action.append(raw["action"])
        checks.append({"trace": trace, "state_raw": tensor_summary(raw["state"]),
                       "state_normalized": tensor_summary(sample["state"]),
                       "action_raw": tensor_summary(raw["action"]),
                       "action_normalized": tensor_summary(sample["action"]),
                       "inverse_roundtrip_passed": True, "raw_reread_unchanged": True})
    raw_state, raw_action = np.stack(raw_state), np.stack(raw_action)
    state = np.stack([sample["state"] for sample in samples])
    action = np.stack([sample["action"] for sample in samples])
    report.update({"dataset": dataset.provenance(), "normalization": dataset.normalizer.provenance(),
                   "model": model_identity(args.model), "input_checks": checks,
                   "roundtrip_tolerance": {"rtol": 1e-5, "atol": 1e-6},
                   "state_interface": tensor_summary(state), "action_interface": tensor_summary(action)})
    write_report(run_dir / "normalization_model_check.json", report)

    # Import the model only after the data/normalizer compatibility checks pass.
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

    class QwenPINormalizationCheck(Qwen_PI):
        def _encode_vl_hidden_states(self, *call_args, **call_kwargs):
            hidden, mask = super()._encode_vl_hidden_states(*call_args, **call_kwargs)
            return [value.float() for value in hidden], mask

    print("[2/4] 加载本地 QwenPI，执行一次前向和反向，不更新参数", flush=True)
    torch.cuda.reset_peak_memory_stats(args.device)
    model = QwenPINormalizationCheck(cfg).to(device=args.device, dtype=torch.float32)
    model.qwen_vl_interface.model.config.use_cache = False
    action_cfg = model.config.framework.action_model
    if (action_cfg.state_dim, action_cfg.action_dim, int(model.action_horizon)) != (16, 16, 16):
        raise ValueError("Expected the unchanged world-pose-16D / horizon-16 model interface")
    if not (model.action_model.model.config.interleave_self_attention
            and model.action_model.model.config.use_canonical_forward):
        raise ValueError("Expected the previously verified canonical action/state attention path")
    OmegaConf.save(model.config, run_dir / "config_resolved.yaml")
    seen = []

    def check_action_head_input(module, inputs):
        targets, states = inputs[1:3]
        if targets.shape[0] % args.samples:
            raise ValueError("Unexpected action head repeat count")
        repeats = targets.shape[0] // args.samples
        expected_action = torch.as_tensor(action, device=targets.device, dtype=targets.dtype).repeat(repeats, 1, 1)
        expected_state = torch.as_tensor(state, device=states.device, dtype=states.dtype).repeat(repeats, 1, 1)
        torch.testing.assert_close(targets, expected_action, rtol=0, atol=0)
        torch.testing.assert_close(states, expected_state, rtol=0, atol=0)
        seen.append({"normalized_actions_reach_fm_before_noise": True,
                     "normalized_states_reach_head": True,
                     "repeats": int(repeats), "action_shape": list(targets.shape),
                     "state_shape": list(states.shape), "dtype": str(targets.dtype)})

    handle = model.action_model.register_forward_pre_hook(check_action_head_input)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        loss = model(examples=samples)["action_loss"]
    finally:
        handle.remove()
    if not seen or loss.ndim != 0 or not torch.isfinite(loss).item():
        raise RuntimeError("The normalized-input forward check failed or loss is nonfinite")
    loss.backward()
    groups = {
        "vlm": model.qwen_vl_interface.named_parameters(),
        "action_head": model.action_model.named_parameters(),
        "state_encoder": model.action_model.state_encoder.named_parameters(),
    }
    gradients = []
    for name, parameters in groups.items():
        probe = gradient_probe(list(parameters), name)
        gradients.append({key: value for key, value in probe.items()
                          if key not in ("parameter", "indices", "before")})
        del probe
    report.update({"forward": {"action_loss": float(loss.detach().item()), "head_inputs": seen},
                   "backward": {"finite_nonzero_groups": gradients}, "optimizer_steps": 0})
    del loss
    model.zero_grad(set_to_none=True)
    write_report(run_dir / "normalization_model_check.json", report)

    print("[3/4] 在归一化动作空间完成全部采样步骤，再反变换最终输出", flush=True)
    sampling_calls = []
    sampling_state_calls = []

    def count_sampling_steps(module, inputs, output):
        sampling_calls.append(1)

    def check_sampling_state(module, inputs):
        actual = inputs[0]
        expected = torch.as_tensor(state, device=actual.device, dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        sampling_state_calls.append(1)

    step_handle = model.action_model.model.register_forward_hook(count_sampling_steps)
    state_handle = model.action_model.state_encoder.register_forward_pre_hook(check_sampling_state)
    model.eval()
    # The generation call has no ground-truth action, even though training used it.
    inference_samples = [{key: value for key, value in sample.items() if key != "action"} for sample in samples]
    try:
        generated = model.predict_action(examples=inference_samples)["normalized_actions"]
    finally:
        step_handle.remove()
        state_handle.remove()
    expected_steps = int(model.action_model.num_inference_timesteps)
    if len(sampling_calls) != expected_steps or len(sampling_state_calls) != 1:
        raise RuntimeError("The complete sampler or its normalized state condition did not run as expected")
    check_fp32(generated, (args.samples, 16, 16), "generated normalized action")
    restored = dataset.normalizer.inverse_action(generated)
    check_fp32(restored, (args.samples, 16, 16), "generated original-coordinate action")
    # Tiny scales around nonzero means amplify FP32 rounding when mapping back
    # to normalized units. Raw-data roundtrips were checked above; report this
    # generated-output roundtrip without treating the amplification as a bug.
    regenerated = dataset.normalizer.normalize_action(restored)
    check_fp32(regenerated, generated.shape, "generated normalized roundtrip")
    arrays_path = run_dir / "engineering_normalization_arrays.npz"
    np.savez_compressed(arrays_path, raw_state=raw_state, normalized_state=state,
                        raw_action_label=raw_action, normalized_action_label=action,
                        predicted_normalized_action=generated, predicted_raw_action=restored)
    report.update({
        "status": "passed", "arrays_path": str(arrays_path),
        "sampling": {"steps": expected_steps, "observed_dit_calls": len(sampling_calls),
                     "observed_normalized_state_encoder_calls": len(sampling_state_calls),
                     "normalized_output": tensor_summary(generated),
                     "inverse_output": tensor_summary(restored),
                     "normalized_roundtrip_max_abs_error": float(np.max(np.abs(
                         regenerated.astype(np.float64) - generated.astype(np.float64)))),
                     "raw_coordinate_rmse_connectivity_only": float(np.sqrt(np.mean(
                         (restored.astype(np.float64) - raw_action.astype(np.float64)) ** 2))),
                     "unit_quaternion_projection_applied": False},
        "peak_allocated_memory_gib": torch.cuda.max_memory_allocated(args.device) / 1024 ** 3,
    })
    write_report(run_dir / "normalization_model_check.json", report)
    print(f"[4/4] PASS：归一化输入、反向、完整采样和反变换已连接；报告：{run_dir}", flush=True)


def main(argv=None):
    args = arguments(argv)
    args.index_dir, args.statistics, args.model = args.index_dir.resolve(), args.statistics.resolve(), args.model.resolve()
    runs_root = (REPO.parent / "runs").resolve()
    name = "qwenpi_normalization_engineering_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = (args.output_dir or runs_root / name).resolve()
    if run_dir == runs_root or not run_dir.is_relative_to(runs_root):
        raise ValueError(f"--output-dir must be a new private subdirectory of {runs_root}")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "purpose": "engineering", "formal_training_approved": False,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "samples": args.samples, "device": args.device, "optimizer_steps": 0,
              "scope": "One forward/backward and complete sampling/inverse-transform connection check. "
                       "No optimizer update; this does not establish policy quality, an optimal scaling "
                       "strategy, final task/scene splits, or physically valid quaternion predictions."}
    write_report(run_dir / "normalization_model_check.json", report)
    try:
        run(args, run_dir, report)
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_report(run_dir / "normalization_model_check.json", report)
        raise


if __name__ == "__main__":
    main()
