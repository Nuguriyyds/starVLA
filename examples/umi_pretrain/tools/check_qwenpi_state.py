"""Manually verify QwenPI state information flow on one private UMI sample.

This command performs inference and one backward pass, but never creates an
optimizer, updates weights, or loads/overwrites an old training checkpoint.
"""
import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO.parent / "datasets/roban_umi_debug")
    parser.add_argument("--model", type=Path, default=REPO.parent / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.model = args.model.resolve()
    private_root = REPO.parent.resolve()
    if not args.dataset.is_relative_to(private_root / "datasets"):
        raise ValueError("--dataset must remain under the private datasets directory")
    if not args.model.is_relative_to(private_root / "models"):
        raise ValueError("--model must remain under the private models directory")
    run_dir = private_root / "runs" / ("qwenpi_state_check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "passed": False, "output_dir": str(run_dir), "seed": args.seed, "parameter_updates": False}
    write_json(run_dir / "report.json", report)
    print("诊断输出目录：", run_dir, flush=True)
    try:
        execute(args, run_dir, report)
    except Exception as error:
        report["status"] = "failed"
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(run_dir / "report.json", report)
        raise
    write_json(run_dir / "report.json", report)
    summary = {
        "status": report["status"],
        "passed": report["passed"],
        "forward_state_response": report["forward_state_response"],
        "state_encoder_gradients": report["state_encoder_gradients"],
        "state_encoder_gradient_check": report["state_encoder_gradient_check"],
        "sampling_state_response": report["sampling_state_response"],
        "all_runs_finite": all(item["finite"] for item in report["runs"].values()),
        "all_routes_passed": all(item["routing_passed"] for item in report["runs"].values()),
        "report_path": str(run_dir / "report.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if not report["passed"]:
        raise SystemExit("FAIL：state 通路检查未全部通过，详见 report.json；没有更新模型参数。")
    print("PASS：真实前向/采样路由、state 扰动响应和 state_encoder 梯度通过；没有更新模型参数。", flush=True)


def execute(args, run_dir, report):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from train_qwenpi_debug import audit_data, make_debug_config, model_identity
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires an available CUDA-compatible device")
    cfg = make_debug_config(args.dataset, args.model)
    print("[1/4] 审计私人数据并选取一个有效窗口", flush=True)
    subset, source, allowed, checks = audit_data(args, cfg)
    sample = subset[0]
    report["data"] = {"dataset": str(args.dataset), "valid_windows": len(allowed), "sample_dataset_index": int(allowed[0]), "preflight": checks}
    report["model"] = model_identity(args.model)
    report["precision"] = "FP32 parameters and action head; existing VLM BF16 autocast; FP32 hidden-state bridge"
    report["scope"] = "Fresh action head. Checks connectivity, not learned action quality. eval() disables dropout; no optimizer is created."

    class QwenPIStateCheck(Qwen_PI):
        def _encode_vl_hidden_states(self, *call_args, **call_kwargs):
            hidden, mask = super()._encode_vl_hidden_states(*call_args, **call_kwargs)
            return [value.float() for value in hidden], mask

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("[2/4] 加载真实 QwenPI；检查逐层 cross/self 路由和相同噪声下的 state 响应", flush=True)
    model = QwenPIStateCheck(cfg).to(device=args.device, dtype=torch.float32)
    model.qwen_vl_interface.model.config.use_cache = False
    model.eval()
    OmegaConf.save(model.config, run_dir / "config.yaml")
    head = model.action_model
    dit = head.model
    if head.state_encoder is None or model.action_horizon != 16:
        raise ValueError("Expected a state encoder and 16-step actions")
    expected_layers = len(dit.transformer_blocks)
    if not bool(dit.config.interleave_self_attention):
        raise ValueError("interleave_self_attention must be enabled")
    if not bool(dit.config.use_canonical_forward):
        raise ValueError("use_canonical_forward must be enabled")

    # Save and restore both CPU and accelerator RNG states. FM uses CPU Beta
    # samples and accelerator noise; restoring only one does not fix the input.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()])

    def reset_rng():
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])

    variants = {"control": sample}
    for name, dimension in (("robot1_x_plus_0_05", 0), ("robot2_x_plus_0_05", 8)):
        changed = dict(sample)
        changed["state"] = sample["state"].copy()
        changed["state"][0, dimension] += np.float32(0.05)
        variants[name] = changed

    active = {"calls": [], "context": None, "errors": [], "velocities": [], "decoder": []}
    handles = []

    def bind(module, args_, kwargs_):
        return inspect.signature(module.forward).bind_partial(*args_, **kwargs_).arguments

    def before_dit(module, args_, kwargs_):
        arguments = bind(module, args_, kwargs_)
        contexts = arguments.get("encoder_hidden_states")
        mask = arguments.get("encoder_attention_mask")
        record = {"blocks": [], "layerwise_context_count": len(contexts) if isinstance(contexts, (list, tuple)) else None}
        active["calls"].append(record)
        active["context"] = (contexts, mask, record)
        if not isinstance(contexts, (list, tuple)) or len(contexts) != expected_layers:
            active["errors"].append("DiT must receive one VLM context per original layer; do not compress to cross-only layers")
        if mask is None:
            active["errors"].append("DiT did not receive the real VLM padding mask")

    handles.append(dit.register_forward_pre_hook(before_dit, with_kwargs=True))

    def make_block_hook(index):
        def before_block(module, args_, kwargs_):
            arguments = bind(module, args_, kwargs_)
            context = arguments.get("encoder_hidden_states")
            mask = arguments.get("encoder_attention_mask")
            if active["context"] is None:
                active["errors"].append(f"block {index}: called outside canonical DiT.forward")
                return
            contexts, expected_mask, record = active["context"]
            item = {"index": index, "expected": "cross" if index % 2 == 0 else "self"}
            if index % 2:
                item["passed"] = context is None and mask is None
            else:
                valid_contexts = isinstance(contexts, (list, tuple)) and len(contexts) == expected_layers
                expected = contexts[index] if valid_contexts else None
                item["original_vlm_layer_index"] = index
                item["context_matches_original_layer"] = bool(expected is not None and context is not None and (context is expected or torch.equal(context, expected)))
                item["mask_matches_vlm_mask"] = bool(expected_mask is not None and mask is not None and (mask is expected_mask or torch.equal(mask, expected_mask)))
                item["passed"] = item["context_matches_original_layer"] and item["mask_matches_vlm_mask"]
            if not item["passed"]:
                active["errors"].append(f"block {index}: incorrect context or mask")
            record["blocks"].append(item)
        return before_block

    for index, block in enumerate(dit.transformer_blocks):
        handles.append(block.register_forward_pre_hook(make_block_hook(index), with_kwargs=True))

    def decoder_output(module, args_, output):
        active["velocities"].append(output[:, -model.action_horizon:].detach().float().cpu().clone())

    def before_relu(module, args_, output):
        detached = output[:, -model.action_horizon:].detach().float()
        active["decoder"].append({
            "token_scope": "last 16 action positions",
            "pre_relu_positive_fraction": finite_number((detached > 0).float().mean().item()),
            "pre_relu_rms": finite_number(detached.square().mean().sqrt().item()),
            "decoder_input_rms": finite_number(args_[0][:, -model.action_horizon:].detach().float().square().mean().sqrt().item()),
        })

    handles.append(head.action_decoder.register_forward_hook(decoder_output))
    handles.append(head.action_decoder.layer1.register_forward_hook(before_relu))
    report["runs"] = {}

    def run(name, example, *, predict=False, backward=False):
        reset_rng()
        active.update(calls=[], context=None, errors=[], velocities=[], decoder=[])
        if backward:
            model.zero_grad(set_to_none=True)
            result = model([example])
            loss = result["action_loss"]
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite action_loss in gradient check")
            loss.backward()
            output = active["velocities"][-1]
        elif predict:
            with torch.no_grad():
                result = model.predict_action([example])
            output = torch.as_tensor(result["normalized_actions"]).float().cpu().clone()
        else:
            with torch.no_grad():
                result = model([example])
            output = active["velocities"][-1]
        expected_calls = int(head.num_inference_timesteps) if predict else 1
        if len(active["calls"]) != expected_calls:
            active["errors"].append(f"Expected {expected_calls} canonical DiT calls, got {len(active['calls'])}")
        for call in active["calls"]:
            if [item["index"] for item in call["blocks"]] != list(range(expected_layers)):
                active["errors"].append("DiT block execution order/count differs from its original depth")
        finite = bool(torch.isfinite(output).all().item())
        item = {
            "kind": "predict_action" if predict else "forward_with_backward" if backward else "forward",
            "finite": finite,
            "output_shape": list(output.shape),
            "output_rms": finite_number(output.double().square().mean().sqrt().item()),
            "routing_passed": not active["errors"],
            "routing_errors": list(active["errors"]),
            "dit_calls": active["calls"],
            "decoder_diagnostics": active["decoder"],
        }
        if not predict:
            item["action_loss"] = finite_number(result["action_loss"].item())
            item["finite"] = finite and item["action_loss"] is not None
        report["runs"][name] = item
        write_json(run_dir / "report.json", report)
        active["context"] = None  # Do not retain the VLM computation graph.
        print(name, "finite=", item["finite"], "routing_passed=", item["routing_passed"], flush=True)
        return output

    def difference(first, second):
        return finite_number((first.double() - second.double()).abs().max().item())

    def response_group(prefix, predict=False):
        baseline = run(prefix + "_control", variants["control"], predict=predict)
        repeated = run(prefix + "_control_repeat", variants["control"], predict=predict)
        repeat_error = difference(baseline, repeated)
        threshold = max(1e-7, 10 * repeat_error) if repeat_error is not None else None
        repeat_passed = bool(torch.allclose(baseline, repeated, rtol=1e-6, atol=1e-7))
        response = {"repeat_max_abs_difference": repeat_error, "repeat_passed": repeat_passed, "repeat_rtol": 1e-6, "repeat_atol": 1e-7, "required_perturbation_difference": threshold, "perturbations": {}}
        for name in ("robot1_x_plus_0_05", "robot2_x_plus_0_05"):
            changed = run(prefix + "_" + name, variants[name], predict=predict)
            delta = difference(baseline, changed)
            response["perturbations"][name] = {"max_abs_difference": delta, "passed": threshold is not None and delta is not None and delta > threshold}
        response["passed"] = repeat_passed and all(item["passed"] for item in response["perturbations"].values())
        return response

    try:
        report["forward_state_response"] = response_group("forward")
        print("[3/4] 对真实动作损失反传，检查 state_encoder 梯度", flush=True)
        run("backward_control", sample, backward=True)
        gradients = {}
        for name, parameter in head.state_encoder.named_parameters():
            gradient = parameter.grad
            finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
            norm = finite_number(gradient.detach().float().norm().item()) if gradient is not None else None
            gradients[name] = {"has_gradient": gradient is not None, "finite": finite, "norm": norm, "passed": finite and norm is not None}
        report["state_encoder_gradients"] = gradients
        all_finite = bool(gradients) and all(item["passed"] for item in gradients.values())
        aggregate_norm = finite_number(math.hypot(*(item["norm"] for item in gradients.values()))) if all_finite else None
        report["state_encoder_gradient_check"] = {
            "all_parameter_gradients_finite": all_finite,
            "aggregate_norm": aggregate_norm,
            "passed": all_finite and aggregate_norm is not None and aggregate_norm > 0,
        }
        model.zero_grad(set_to_none=True)
        print("[4/4] 通过真实 predict_action 入口检查完整采样的 state 响应", flush=True)
        report["sampling_state_response"] = response_group("sampling", predict=True)
    finally:
        for handle in handles:
            handle.remove()
        active["context"] = None

    report["passed"] = (
        all(item["finite"] and item["routing_passed"] for item in report["runs"].values())
        and report["forward_state_response"]["passed"]
        and report["state_encoder_gradient_check"]["passed"]
        and report["sampling_state_response"]["passed"]
    )
    report["status"] = "completed" if report["passed"] else "failed"


if __name__ == "__main__":
    main()
