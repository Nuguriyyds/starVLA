"""Immutable world-pose-16D scaling and mergeable offline FP64 statistics.

No source reading, fitting at training startup, clipping, quaternion repair,
or model dependencies live here. Statistics weight rows by their appearances
in valid windows: one state and ``horizon`` action positions per window.
"""
from copy import deepcopy
import hashlib
import json
import math
import operator
from pathlib import Path
import sys

import numpy as np


SCHEMA_VERSION = "umi-normalization-v1"
ALGORITHM_VERSION = "weighted-chan-fp64-v1"
REPRESENTATION_VERSION = "umi-world-pose-16d-v1"
EXPERIMENT_VERSION = "umi-normalization-experiment-v1"
SIGNALS = (
    "observation.umi.robot1_finger_eef_pose",
    "observation.umi.robot1_sensor_magnetic_encoder",
    "observation.umi.robot2_finger_eef_pose",
    "observation.umi.robot2_sensor_magnetic_encoder",
)
WIDTHS = (7, 1, 7, 1)
RAW_REPRESENTATION = "robot1 xyz+xyzw+gripper then robot2; raw float32; no normalization"
FIELD_ORDER = tuple(f"robot{hand}.{field}" for hand in (1, 2)
                    for field in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper"))
MAX_EXACT_COUNT = 2**53


def content_fingerprint(value):
    """Canonical JSON identity; callers exclude the fingerprint field itself."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < minimum or result > MAX_EXACT_COUNT:
        raise ValueError(f"{label} must be between {minimum} and {MAX_EXACT_COUNT}")
    return result


def _digest(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _vector(value, dim, label, nonnegative=False):
    array = np.asarray(value)
    if array.shape != (dim,) or array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must contain {dim} real numbers")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all() or (nonnegative and np.any(array < 0)):
        raise ValueError(f"{label} contains non-finite or negative values")
    return array


def _min_std(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("min_std must be positive and finite")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_std must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("min_std must be positive and finite")
    return value


class WeightedMoments:
    """Frequency-weighted count/mean/M2 with stable, mergeable FP64 updates.

    ``weights`` are nonnegative integer occurrence counts, not normalized
    sampling probabilities. Population variance is M2/count (ddof=0).
    The count is bounded by 2**53 so integer counts are exact in FP64 merges.
    """
    def __init__(self, dim=16):
        self.dim = _integer(dim, "dim", 1)
        self.count = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, values, weights=None):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.dim or values.dtype.kind not in "fiu":
            raise ValueError(f"values must be a real [N,{self.dim}] array")
        values = values.astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("values contain NaN/Inf")
        if weights is None:
            weights = np.ones(len(values), dtype=np.int64)
        else:
            weights = np.asarray(weights)
            if weights.shape != (len(values),) or weights.dtype.kind not in "iu":
                raise ValueError("weights must be an integer [N] array")
            if np.any(weights < 0) or np.any(weights > MAX_EXACT_COUNT):
                raise ValueError("weights must be nonnegative exact integer counts")
        selected = weights > 0
        if not selected.any():
            return self
        # Frequency weights are small in this dataset (at most horizon=16).
        # Sum in NumPy when the upper bound is safe; retain an exact fallback
        # for arbitrary caller weights without silent integer overflow.
        selected_weights = weights[selected]
        if int(selected_weights.max()) * len(selected_weights) <= MAX_EXACT_COUNT:
            count = int(selected_weights.sum(dtype=np.int64))
        else:
            count = sum(int(weight) for weight in selected_weights)
        _integer(count + self.count, "merged count")
        x = values[selected]
        w = weights[selected].astype(np.float64)
        # Work around a reference observation, avoiding a huge weighted sum
        # of offsets when a feature has large mean but small variance.
        reference = x[0]
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = x - reference
            shift_mean = np.sum(shifted * w[:, None], axis=0) / count
            mean = reference + shift_mean
            residual = shifted - shift_mean
            M2 = np.sum((residual * residual) * w[:, None], axis=0)
        other = WeightedMoments(self.dim)
        other.load_state_dict({"algorithm_version": ALGORITHM_VERSION, "dim": self.dim,
                               "count": count, "mean": mean.tolist(), "M2": M2.tolist()})
        return self.merge(other)

    def merge(self, other):
        if not isinstance(other, WeightedMoments) or other.dim != self.dim:
            raise ValueError("Can only merge WeightedMoments of the same dimension")
        # Validate even objects whose public arrays were mutated by a caller.
        incoming = WeightedMoments.from_state_dict(other.state_dict())
        current = WeightedMoments.from_state_dict(self.state_dict())
        total = _integer(current.count + incoming.count, "merged count")
        if incoming.count == 0:
            return self
        if current.count == 0:
            self.count = incoming.count
            self.mean, self.M2 = incoming.mean.copy(), incoming.M2.copy()
            return self
        with np.errstate(over="ignore", invalid="ignore"):
            delta = incoming.mean - current.mean
            mean = current.mean + delta * (incoming.count / total)
            M2 = current.M2 + incoming.M2 + delta**2 * (current.count * (incoming.count / total))
        # load_state_dict validates all fields before changing this instance.
        return self.load_state_dict({"algorithm_version": ALGORITHM_VERSION, "dim": self.dim,
                                     "count": total, "mean": mean.tolist(), "M2": M2.tolist()})

    def state_dict(self):
        return {"algorithm_version": ALGORITHM_VERSION, "dim": self.dim, "count": self.count,
                "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError("Unknown weighted-moments state version")
        if _integer(state.get("dim"), "dim", 1) != self.dim:
            raise ValueError("Weighted-moments dimension mismatch")
        count = _integer(state.get("count"), "count")
        mean = _vector(state.get("mean"), self.dim, "mean")
        M2 = _vector(state.get("M2"), self.dim, "M2", nonnegative=True)
        if count == 0 and (np.any(mean != 0) or np.any(M2 != 0)):
            raise ValueError("Empty moments must have zero mean and M2")
        if count == 1 and np.any(M2 != 0):
            raise ValueError("One observation must have zero M2")
        self.count, self.mean, self.M2 = count, mean, M2
        return self

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict):
            raise ValueError("Weighted-moments state must be a dictionary")
        return cls(_integer(state.get("dim"), "dim", 1)).load_state_dict(state)

    def statistics(self, min_std=1e-6):
        min_std = _min_std(min_std)
        checked = WeightedMoments.from_state_dict(self.state_dict())
        if checked.count == 0:
            raise ValueError("Cannot finalize empty statistics")
        variance = checked.M2 / checked.count
        std = np.sqrt(variance)
        constant = std < min_std
        scale = np.where(constant, 1.0, std)
        return {"count": checked.count, "mean": checked.mean.tolist(),
                "variance": variance.tolist(), "scale": scale.tolist(),
                "constant_dimensions": np.flatnonzero(constant).tolist()}


def build_representation(meta):
    """Validate access metadata and return the exact low-dimensional contract."""
    if (tuple(meta.get("signals", ())) != SIGNALS
            or tuple(meta.get("signal_widths", ())) != WIDTHS
            or meta.get("representation") != RAW_REPRESENTATION
            or _integer(meta.get("horizon"), "horizon", 1) != 16):
        raise ValueError("Access metadata is not the established world-pose-16D representation")
    return {"version": REPRESENTATION_VERSION, "dimension": 16, "horizon": 16,
            "signals": list(SIGNALS), "signal_widths": list(WIDTHS),
            "field_order": list(FIELD_ORDER), "coordinate_frame": "world",
            "quaternion_order": "xyzw", "quaternion_policy": "raw_no_sign_flip_no_projection",
            "input_dtype": "float32", "representation": RAW_REPRESENTATION}


def parent_fingerprint(meta):
    """Common parent identity shared by fit, validation, and stage access views.

    Excludes paths, timestamps, selected-episode lists, and view payload hashes.
    The frozen rule fingerprint and catalog hashes retain the source identity
    and data/quality rules; they must agree for all views of one experiment.
    """
    build_representation(meta)
    catalog = meta.get("catalog_artifacts_sha256")
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("Missing parent catalog artifact hashes")
    for key, value in catalog.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Invalid catalog hash key")
        _digest(value, "catalog artifact hash")
    source_hash = meta.get("window_contract", {}).get("source_info_sha256")
    if source_hash is None:
        source_hash = meta.get("source_artifact_sha256", {}).get("source_info.json")
    return content_fingerprint({"version": "umi-parent-data-v1",
                                "rule_fingerprint": _digest(meta.get("rule_fingerprint"), "rule_fingerprint"),
                                "catalog_artifacts_sha256": catalog,
                                "source_info_sha256": _digest(source_hash, "source_info_sha256")})


def _experiment_contract(contract, parent, representation, fit_view, current_view=None):
    if not isinstance(contract, dict) or contract.get("schema_version") != EXPERIMENT_VERSION:
        raise ValueError("Formal normalization requires an explicit experiment contract")
    if not isinstance(contract.get("experiment_id"), str) or not contract["experiment_id"].strip():
        raise ValueError("experiment_id is required")
    if contract.get("training_pool_approved") is not True:
        raise ValueError("Formal statistics require an explicitly approved training pool")
    if (contract.get("parent_data_fingerprint") != parent
            or contract.get("representation_fingerprint") != content_fingerprint(representation)
            or contract.get("fit_view_fingerprint") != fit_view):
        raise ValueError("Experiment contract does not match statistics fit source/representation")
    allowed = contract.get("allowed_apply_views")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("allowed_apply_views must be a nonempty unique list")
    for value in allowed:
        _digest(value, "allowed_apply_views")
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed_apply_views must be a nonempty unique list")
    if current_view is not None and current_view not in allowed:
        raise ValueError("Current access view is not approved by the experiment contract")
    return deepcopy(contract)


def validate_experiment_contract(contract, *, access_meta, fit_view_fingerprint,
                                 current_view_fingerprint=None):
    """Validate a formal fit/apply declaration before a potentially long scan."""
    fit_view = _digest(fit_view_fingerprint, "fit_view_fingerprint")
    if current_view_fingerprint is not None:
        _digest(current_view_fingerprint, "current_view_fingerprint")
    return _experiment_contract(contract, parent_fingerprint(access_meta),
                                build_representation(access_meta), fit_view,
                                current_view_fingerprint)


def build_statistics(state, action, *, access_meta, fit_view_fingerprint,
                     purpose="engineering", fit_details, code_version,
                     min_std=1e-6, experiment_contract=None):
    """Create a completed, self-fingerprinted artifact from accumulated moments."""
    if purpose not in ("engineering", "formal"):
        raise ValueError("purpose must be engineering or formal")
    if state.dim != 16 or action.dim != 16 or state.count <= 0 or action.count != 16 * state.count:
        raise ValueError("Require 16D state count=N and action count=16*N")
    if not isinstance(fit_details, dict):
        raise ValueError("fit_details must explicitly describe the selected input")
    if not isinstance(code_version, str) or not code_version.strip():
        raise ValueError("code_version is required")
    representation = build_representation(access_meta)
    parent = parent_fingerprint(access_meta)
    fit_view = _digest(fit_view_fingerprint, "fit_view_fingerprint")
    min_std = _min_std(min_std)
    if purpose == "formal":
        contract = _experiment_contract(experiment_contract, parent, representation, fit_view)
    else:
        if experiment_contract is not None:
            raise ValueError("Engineering statistics must not declare formal approval")
        contract = None
    result = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "method": "mean_std",
        "purpose": purpose, "representation": representation,
        "parent_data_fingerprint": parent, "fit_view_fingerprint": fit_view,
        "fit_details": deepcopy(fit_details), "code_version": code_version,
        "algorithm_version": ALGORITHM_VERSION,
        "counting": {"version": "valid-window-frequency-v1", "state_per_window": 1,
                     "action_per_window": 16, "variance_ddof": 0,
                     "overlapping_rows": "count_every_target_occurrence"},
        "min_std": min_std, "scale_policy": "std_below_min_std_uses_one",
        "state": state.statistics(min_std), "action": action.statistics(min_std),
        "experiment_contract": contract,
        "experiment_contract_fingerprint": None if contract is None else content_fingerprint(contract),
    }
    result["fingerprint"] = content_fingerprint(result)
    return result


def _validate_statistics(artifact, access_meta):
    if not isinstance(artifact, dict):
        raise ValueError("Statistics must be a JSON object")
    body = {key: value for key, value in artifact.items() if key != "fingerprint"}
    if _digest(artifact.get("fingerprint"), "fingerprint") != content_fingerprint(body):
        raise ValueError("Statistics content fingerprint mismatch")
    if (artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("status") != "completed"
            or artifact.get("method") != "mean_std"
            or artifact.get("algorithm_version") != ALGORITHM_VERSION):
        raise ValueError("Unsupported statistics schema, method, algorithm, or incomplete artifact")
    if artifact.get("representation") != build_representation(access_meta):
        raise ValueError("Statistics representation/field order differs from the access view")
    if artifact.get("parent_data_fingerprint") != parent_fingerprint(access_meta):
        raise ValueError("Statistics parent data differs from the access view")
    _digest(artifact.get("fit_view_fingerprint"), "fit_view_fingerprint")
    expected_counting = {"version": "valid-window-frequency-v1", "state_per_window": 1,
                         "action_per_window": 16, "variance_ddof": 0,
                         "overlapping_rows": "count_every_target_occurrence"}
    if artifact.get("counting") != expected_counting:
        raise ValueError("Statistics counting policy mismatch")
    if artifact.get("scale_policy") != "std_below_min_std_uses_one":
        raise ValueError("Unknown scale policy")
    if (not isinstance(artifact.get("fit_details"), dict)
            or not isinstance(artifact.get("code_version"), str) or not artifact["code_version"].strip()):
        raise ValueError("Missing fitting/code provenance")
    min_std = _min_std(artifact.get("min_std"))
    checked = {}
    for name in ("state", "action"):
        params = artifact.get(name)
        if not isinstance(params, dict):
            raise ValueError(f"Missing {name} statistics")
        count = _integer(params.get("count"), f"{name}.count", 1)
        mean = _vector(params.get("mean"), 16, f"{name}.mean")
        variance = _vector(params.get("variance"), 16, f"{name}.variance", nonnegative=True)
        if count == 1 and np.any(variance != 0):
            raise ValueError(f"{name} with one observation must have zero variance")
        scale = _vector(params.get("scale"), 16, f"{name}.scale")
        std = np.sqrt(variance)
        constant = np.flatnonzero(std < min_std).tolist()
        expected_scale = np.where(std < min_std, 1.0, std)
        if (np.any(scale <= 0) or not np.allclose(scale, expected_scale, rtol=1e-12, atol=0)
                or params.get("constant_dimensions") != constant):
            raise ValueError(f"{name}.scale/constant dimensions violate declared variance policy")
        checked[name] = (count, mean, scale)
    if checked["action"][0] != 16 * checked["state"][0]:
        raise ValueError("Statistics require action count=16*state count")
    return checked


class UMINormalizer:
    """Fixed FP32 state/action transforms for NumPy arrays or torch tensors.

    Any leading sample/batch axes are preserved; the final axis must be 16.
    All transforms allocate new storage. Torch tensors retain their device and
    autograd connection; torch is not imported for NumPy-only statistics jobs.
    """
    def __init__(self, mode="none", statistics_path=None, *, access_meta=None,
                 current_view_fingerprint=None, purpose="engineering", experiment_contract=None):
        if mode not in ("none", "mean_std"):
            raise ValueError("normalization must be explicitly none or mean_std")
        if purpose not in ("engineering", "formal"):
            raise ValueError("purpose must be engineering or formal")
        self.mode, self.purpose = mode, purpose
        self.current_view_fingerprint = (None if current_view_fingerprint is None else
                                         _digest(current_view_fingerprint, "current_view_fingerprint"))
        self.representation = None if access_meta is None else build_representation(access_meta)
        self.parent_data_fingerprint = None if access_meta is None else parent_fingerprint(access_meta)
        self.statistics_path = self.statistics_file_sha256 = None
        self._artifact = None
        self._params = {}
        if mode == "none":
            if statistics_path is not None or experiment_contract is not None:
                raise ValueError("none mode must not claim a fitted normalization artifact")
            return
        if statistics_path is None or access_meta is None or self.current_view_fingerprint is None:
            raise ValueError("mean_std requires statistics_path, access_meta and current_view_fingerprint")
        path = Path(statistics_path).resolve()
        raw = path.read_bytes()  # Missing artifacts fail; no fitting or fallback.
        artifact = json.loads(raw)
        checked = _validate_statistics(artifact, access_meta)
        if artifact.get("purpose") != purpose:
            raise ValueError("Statistics purpose differs from this run; engineering is not formal")
        if purpose == "formal":
            stored = _experiment_contract(artifact.get("experiment_contract"), self.parent_data_fingerprint,
                                          self.representation, artifact["fit_view_fingerprint"],
                                          self.current_view_fingerprint)
            supplied = _experiment_contract(experiment_contract, self.parent_data_fingerprint,
                                            self.representation, artifact["fit_view_fingerprint"],
                                            self.current_view_fingerprint)
            digest = content_fingerprint(stored)
            if artifact.get("experiment_contract_fingerprint") != digest or content_fingerprint(supplied) != digest:
                raise ValueError("Formal experiment contract fingerprint mismatch")
        elif (experiment_contract is not None or artifact.get("experiment_contract") is not None
              or artifact.get("experiment_contract_fingerprint") is not None):
            raise ValueError("Engineering artifacts must not claim formal approval")
        self._artifact = deepcopy(artifact)
        self.statistics_path = str(path)
        self.statistics_file_sha256 = hashlib.sha256(raw).hexdigest()
        for name, (_, mean, scale) in checked.items():
            # Parameters are fixed and private; provenance returns a deep copy.
            mean, scale = mean.astype(np.float32), scale.astype(np.float32)
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Statistics cannot be represented safely in FP32")
            mean.flags.writeable = scale.flags.writeable = False
            self._params[name] = (mean, scale)

    def _transform(self, value, name, inverse):
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(value, torch.Tensor):
            if value.ndim < 1 or value.shape[-1] != 16 or not value.is_floating_point():
                raise ValueError("Expected a floating tensor with final dimension 16")
            x = value.to(dtype=torch.float32)
            if not torch.isfinite(x).all().item():
                raise ValueError("Transform input contains NaN/Inf or FP32 overflow")
            if self.mode == "none":
                result = x.clone()
            else:
                mean, scale = (torch.tensor(v.tolist(), dtype=torch.float32, device=x.device)
                               for v in self._params[name])
                result = x * scale + mean if inverse else (x - mean) / scale
            if not torch.isfinite(result).all().item():
                raise ValueError("Transform output contains NaN/Inf")
            return result
        x = np.asarray(value)
        if x.ndim < 1 or x.shape[-1] != 16 or x.dtype.kind not in "fiu":
            raise ValueError("Expected a real array with final dimension 16")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = x.astype(np.float32, copy=True)
            if not np.isfinite(x).all():
                raise ValueError("Transform input contains NaN/Inf or FP32 overflow")
            if self.mode == "none":
                result = x
            else:
                mean, scale = self._params[name]
                result = x * scale + mean if inverse else (x - mean) / scale
        if not np.isfinite(result).all():
            raise ValueError("Transform output contains NaN/Inf")
        return result

    def normalize_state(self, value):
        return self._transform(value, "state", False)

    def normalize_action(self, value):
        return self._transform(value, "action", False)

    def inverse_state(self, value):
        return self._transform(value, "state", True)

    def inverse_action(self, value):
        return self._transform(value, "action", True)

    def provenance(self):
        return deepcopy({"method": self.mode, "purpose": self.purpose,
                         "transform_version": SCHEMA_VERSION,
                         "raw_representation": self.representation,
                         "model_space": "raw" if self.mode == "none" else "per_dimension_mean_std",
                         "parent_data_fingerprint": self.parent_data_fingerprint,
                         "current_view_fingerprint": self.current_view_fingerprint,
                         "statistics_path": self.statistics_path,
                         "statistics_file_sha256": self.statistics_file_sha256,
                         "statistics_fingerprint": None if self._artifact is None else self._artifact["fingerprint"],
                         "fit_view_fingerprint": None if self._artifact is None else self._artifact["fit_view_fingerprint"],
                         "fit_details": None if self._artifact is None else self._artifact["fit_details"],
                         "field_order": list(FIELD_ORDER),
                         "min_std": None if self._artifact is None else self._artifact["min_std"],
                         "experiment_contract_fingerprint": None if self._artifact is None else self._artifact["experiment_contract_fingerprint"]})
