"""Committed optimizer-update progress; never infer consumption from prefetch."""
from copy import deepcopy
import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def checkpoint_config(plan):
    config = plan.get("checkpoint", {})
    if not isinstance(config, dict) or set(config) - {"integrity", "every_updates"}:
        raise ValueError("checkpoint supports integrity and every_updates only")
    if "every_updates" in config and "save_every" in plan["training"]:
        raise ValueError("Use checkpoint.every_updates OR legacy training.save_every, not both")
    mode = config.get("integrity", "basic")
    if mode not in ("basic", "full"):
        raise ValueError("Checkpoint integrity must be basic or full")
    interval = config.get("every_updates", plan["training"].get("save_every"))
    return {"integrity": mode, "every_updates": positive(interval, "checkpoint.every_updates")}


def checkpoint_reasons(step, interval, *, pause=False, complete=False, stage_boundary=False):
    reasons = []
    if step % interval == 0:
        reasons.append("periodic")
    if pause:
        reasons.append("pause")
    if stage_boundary:
        reasons.append("stage_boundary")
    if complete:
        reasons.append("complete")
    return reasons


def validate_plan(plan):
    if plan.get("version") != "umi-training-plan-v1":
        raise ValueError("Expected version=umi-training-plan-v1")
    if plan.get("purpose") not in ("engineering", "formal"):
        raise ValueError("Explicit engineering/formal purpose is required")
    if plan.get("model_kind") not in ("tiny", "qwenpi"):
        raise ValueError("model_kind must be tiny or qwenpi")
    if plan["model_kind"] == "tiny" and plan["purpose"] != "engineering":
        raise ValueError("Synthetic tiny data cannot be used for formal training")
    stages = plan.get("stages", [])
    if not stages or len({s["name"] for s in stages}) != len(stages):
        raise ValueError("Require nonempty stages with unique names")
    for stage in stages:
        if not stage["name"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("Stage names must be path-safe identifiers")
        positive(stage["updates"], "stage updates")
    train = plan["training"]
    checkpoint_config(plan)
    for key in ("batch_size", "gradient_accumulation_steps"):
        positive(train[key], key)
    if type(train.get("num_workers", 0)) is not int or train.get("num_workers", 0) < 0:
        raise ValueError("num_workers must be nonnegative")
    if train.get("mixed_precision", "no") not in ("no", "bf16"):
        raise ValueError("This entry currently supports no/bf16 precision")
    if train.get("eval_every", 0) < 0:
        raise ValueError("eval_every must be nonnegative")
    if train.get("eval_every", 0) and not plan.get("evaluation"):
        raise ValueError("eval_every requires a separate evaluation view")
    total = sum(s["updates"] for s in stages)
    if not 0 <= train.get("warmup_updates", 0) < total:
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if train.get("lr", 0) <= 0 or train.get("max_grad_norm", 0) <= 0:
        raise ValueError("lr and max_grad_norm must be positive")
    return total


class UMITrainingState:
    """State always describes the NEXT update, including at stage boundaries."""
    def __init__(self, budgets, sizes, global_batch):
        self.budgets = [positive(x, "budget") for x in budgets]
        self.sizes = [positive(x, "view size") for x in sizes]
        self.global_batch = positive(global_batch, "global_batch")
        if len(sizes) != len(budgets) or any(n < global_batch for n in sizes):
            raise ValueError("Each stage must contain at least one complete global update")
        self.global_update_step = 0
        self.stage_index = 0
        self.stages = [dict(updates=0, epoch=0, cursor=0, used_samples=0,
                            dropped_tail_samples=0) for _ in sizes]

    @property
    def done(self):
        return self.stage_index == len(self.budgets)

    def current(self):
        if self.done:
            raise ValueError("Training plan is complete")
        return self.stages[self.stage_index]

    def commit_update(self):
        """Call only AFTER a finite successful optimizer step and scheduler step."""
        state, i = self.current(), self.stage_index
        usable = self.sizes[i] // self.global_batch * self.global_batch
        state["updates"] += 1
        state["used_samples"] += self.global_batch
        state["cursor"] += self.global_batch
        self.global_update_step += 1
        if state["cursor"] > usable:
            raise ValueError("Committed cursor exceeds the complete-update epoch")
        if state["cursor"] == usable:
            state["dropped_tail_samples"] += self.sizes[i] - usable
            state["epoch"] += 1
            state["cursor"] = 0
        if state["updates"] == self.budgets[i]:
            self.stage_index += 1

    def state_dict(self):
        return deepcopy(dict(version="umi-training-state-v1", budgets=self.budgets,
                             sizes=self.sizes, global_batch=self.global_batch,
                             global_update_step=self.global_update_step,
                             stage_index=self.stage_index, stages=self.stages))

    def load_state_dict(self, value):
        expected = self.state_dict()
        if set(value) != set(expected):
            raise ValueError("Invalid training-state fields")
        for key in ("version", "budgets", "sizes", "global_batch"):
            if value[key] != expected[key]:
                raise ValueError(f"Training-state {key} differs")
        # Derive all counters arithmetically, avoiding replay of optimizer steps.
        if type(value["global_update_step"]) is not int:
            raise ValueError("Invalid global update")
        remaining = value["global_update_step"]
        if not 0 <= remaining <= sum(self.budgets):
            raise ValueError("Global update outside plan")
        derived, stage_index = [], len(self.budgets)
        for i, (budget, size) in enumerate(zip(self.budgets, self.sizes)):
            updates = min(remaining, budget)
            remaining -= updates
            epoch, steps_in_epoch = divmod(updates, size // self.global_batch)
            derived.append(dict(updates=updates, epoch=epoch,
                                cursor=steps_in_epoch * self.global_batch,
                                used_samples=updates * self.global_batch,
                                dropped_tail_samples=epoch * (size % self.global_batch)))
            if updates < budget and stage_index == len(self.budgets):
                stage_index = i
        if value["stages"] != derived or value["stage_index"] != stage_index:
            raise ValueError("Inconsistent committed sample/stage counters")
        self.stages = deepcopy(derived)
        self.stage_index = stage_index
        self.global_update_step = value["global_update_step"]
