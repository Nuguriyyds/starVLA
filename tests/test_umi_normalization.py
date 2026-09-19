"""CPU numerical and artifact-contract tests; no source data or model needed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "starVLA/dataloader/umi_normalization.py"
SPEC = importlib.util.spec_from_file_location("umi_normalization_test_module", MODULE_PATH)
normalization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalization)
WeightedMoments = normalization.WeightedMoments
UMINormalizer = normalization.UMINormalizer


def fixture_meta():
    return {"signals": list(normalization.SIGNALS), "signal_widths": list(normalization.WIDTHS),
            "horizon": 16, "representation": normalization.RAW_REPRESENTATION,
            "rule_fingerprint": "a" * 64,
            "catalog_artifacts_sha256": {"manifest.parquet": "b" * 64},
            "window_contract": {"source_info_sha256": "c" * 64}}


def fixture_statistics(meta=None, purpose="engineering", contract=None):
    meta = fixture_meta() if meta is None else meta
    rng = np.random.default_rng(23)
    state_values = rng.normal(3.0, 0.2, size=(10, 16))
    action_values = rng.normal(-0.5, 0.7, size=(160, 16))
    state_values[:, 0] = 2
    action_values[:, 1] = 4
    # Near-constant dimensions use scale=1, rather than blowing up differences.
    state_values[:, 2] = np.linspace(0, 1e-9, 10)
    state = WeightedMoments().update(state_values)
    action = WeightedMoments().update(action_values)
    return normalization.build_statistics(state, action, access_meta=meta,
                                         fit_view_fingerprint="d" * 64, purpose=purpose,
                                         fit_details={"selection": "synthetic fixture"},
                                         code_version="test-version", experiment_contract=contract)


def fixture_contract(meta=None):
    meta = fixture_meta() if meta is None else meta
    return {"schema_version": normalization.EXPERIMENT_VERSION,
            "experiment_id": "unit-test-approved-pool", "training_pool_approved": True,
            "parent_data_fingerprint": normalization.parent_fingerprint(meta),
            "representation_fingerprint": normalization.content_fingerprint(normalization.build_representation(meta)),
            "fit_view_fingerprint": "d" * 64, "allowed_apply_views": ["d" * 64, "e" * 64]}


def resign(value):
    value["fingerprint"] = normalization.content_fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
    return value


class MomentTests(unittest.TestCase):
    def test_weighted_matches_expanded_population(self):
        rng = np.random.default_rng(8)
        values = rng.normal(size=(39, 16))
        weights = rng.integers(0, 17, len(values))
        expected = np.repeat(values, weights, axis=0)
        result = WeightedMoments().update(values, weights).statistics()
        self.assertEqual(result["count"], len(expected))
        np.testing.assert_allclose(result["mean"], expected.mean(0), atol=1e-14)
        np.testing.assert_allclose(result["variance"], expected.var(0), rtol=1e-13, atol=1e-14)

    def test_merge_and_serialized_resume_match_one_pass(self):
        rng = np.random.default_rng(7)
        values = 1e6 + rng.normal(scale=0.1, size=(400, 16))
        weights = rng.integers(1, 17, len(values))
        one = WeightedMoments().update(values, weights)
        first = WeightedMoments().update(values[:133], weights[:133])
        restored = WeightedMoments.from_state_dict(json.loads(json.dumps(first.state_dict())))
        restored.update(values[133:222], weights[133:222])
        restored.merge(WeightedMoments().update(values[222:], weights[222:]))
        self.assertEqual(restored.count, one.count)
        np.testing.assert_allclose(restored.mean, one.mean, rtol=0, atol=2e-10)
        np.testing.assert_allclose(restored.M2, one.M2, rtol=2e-9, atol=1e-10)

    def test_zero_weight_and_empty_batch(self):
        result = WeightedMoments().update(np.zeros((3, 16)), np.zeros(3, dtype=np.int64))
        result.update(np.empty((0, 16)))
        self.assertEqual(result.count, 0)
        with self.assertRaises(ValueError):
            result.statistics()
        result.merge(WeightedMoments())
        self.assertEqual(result.count, 0)

    def test_zero_and_near_zero_variance(self):
        values = np.full((20, 16), 3.0)
        values[:, 1] += np.linspace(-1e-9, 1e-9, 20)
        result = WeightedMoments().update(values).statistics()
        self.assertEqual(result["scale"], [1.0] * 16)
        self.assertEqual(result["constant_dimensions"], list(range(16)))

    def test_update_errors_are_atomic(self):
        moments = WeightedMoments().update(np.ones((2, 16)))
        original = moments.state_dict()
        for values, weights in [(np.ones((2, 15)), None),
                                (np.full((2, 16), np.nan), None),
                                (np.ones((2, 16)), np.array([1.0, 2.0])),
                                (np.ones((2, 16)), np.array([1, -2])),
                                (np.ones((2, 16)), np.array([2**53, 1], dtype=np.int64))]:
            with self.subTest(values=values.shape, weights=weights):
                with self.assertRaises(ValueError):
                    moments.update(values, weights)
                self.assertEqual(moments.state_dict(), original)

    def test_restore_rejects_bad_state_atomically(self):
        moments = WeightedMoments().update(np.ones((2, 16)))
        original = moments.state_dict()
        for key, value in [("dim", 15), ("count", -1), ("count", 1.5),
                           ("mean", [0.0] * 15), ("M2", [-1.0] * 16),
                           ("M2", [float("inf")] * 16), ("count", 0),
                           ("algorithm_version", "unknown")]:
            broken = deepcopy(original)
            broken[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    moments.load_state_dict(broken)
                self.assertEqual(moments.state_dict(), original)

    def test_statistics_requires_window_counting(self):
        state = WeightedMoments().update(np.ones((2, 16)))
        action = WeightedMoments().update(np.ones((16, 16)))
        with self.assertRaisesRegex(ValueError, "count"):
            normalization.build_statistics(state, action, access_meta=fixture_meta(),
                                           fit_view_fingerprint="d" * 64,
                                           fit_details={}, code_version="test")


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "statistics.json"
        self.meta = fixture_meta()

    def load(self, artifact=None, *, meta=None, purpose="engineering", contract=None, view="e" * 64):
        artifact = fixture_statistics() if artifact is None else artifact
        self.path.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
        return UMINormalizer("mean_std", self.path, access_meta=self.meta if meta is None else meta,
                             current_view_fingerprint=view, purpose=purpose, experiment_contract=contract)

    def test_numpy_sample_batch_roundtrip_and_independent_state_action(self):
        normalizer = self.load()
        rng = np.random.default_rng(6)
        for shape, kind in [((1, 16), "state"), ((3, 1, 16), "state"),
                            ((16, 16), "action"), ((3, 16, 16), "action"), ((16,), "action")]:
            raw = rng.normal(size=shape).astype(np.float32)
            saved = raw.copy()
            normalized = getattr(normalizer, "normalize_" + kind)(raw)
            recovered = getattr(normalizer, "inverse_" + kind)(normalized)
            self.assertEqual(normalized.shape, shape)
            self.assertEqual(normalized.dtype, np.float32)
            self.assertFalse(np.shares_memory(normalized, raw))
            np.testing.assert_array_equal(raw, saved)
            np.testing.assert_allclose(recovered, raw, rtol=2e-6, atol=1e-6)
        vector = np.zeros(16, dtype=np.float32)
        self.assertFalse(np.array_equal(normalizer.normalize_state(vector), normalizer.normalize_action(vector)))

    def test_none_preserves_values_with_new_storage(self):
        normalizer = UMINormalizer("none")
        raw = np.arange(256, dtype=np.float32).reshape(16, 16)
        for method in (normalizer.normalize_state, normalizer.normalize_action,
                       normalizer.inverse_state, normalizer.inverse_action):
            output = method(raw)
            np.testing.assert_array_equal(output, raw)
            self.assertFalse(np.shares_memory(output, raw))
        self.assertEqual(normalizer.provenance()["model_space"], "raw")

    def test_torch_roundtrip_device_fp32_and_grad(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional for NumPy-only installations")
        normalizer = self.load()
        raw = torch.randn(2, 16, 16, dtype=torch.float32, requires_grad=True)
        result = normalizer.normalize_action(raw)
        recovered = normalizer.inverse_action(result)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, raw.device)
        self.assertNotEqual(result.data_ptr(), raw.data_ptr())
        torch.testing.assert_close(recovered, raw, rtol=2e-6, atol=1e-6)
        result.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all().item())
        bf16 = normalizer.inverse_action(raw.detach().to(torch.bfloat16))
        self.assertEqual(bf16.dtype, torch.float32)
        original = raw.detach().clone()
        UMINormalizer("none").normalize_action(raw).detach().zero_()
        torch.testing.assert_close(original, raw.detach())

    def test_transform_rejects_nonfinite_shape_and_overflow(self):
        normalizer = self.load()
        for value in [np.zeros((2, 15)), np.array(1), np.full((16,), np.nan),
                      np.full((16,), np.inf), np.full((16,), 1e100), np.array(["x"] * 16)]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalizer.normalize_action(value)

    def test_missing_file_and_no_silent_fallback(self):
        with self.assertRaises(FileNotFoundError):
            UMINormalizer("mean_std", self.path, access_meta=self.meta, current_view_fingerprint="e" * 64)
        with self.assertRaises(ValueError):
            UMINormalizer("mean_std")
        with self.assertRaises(ValueError):
            UMINormalizer("percentiles")
        with self.assertRaises(ValueError):
            UMINormalizer("none", self.path)

    def test_parent_ignores_view_and_relocation_but_not_source(self):
        normalizer = self.load()  # fit d, current e is intentionally allowed.
        self.assertEqual(normalizer.provenance()["fit_view_fingerprint"], "d" * 64)
        moved = deepcopy(self.meta)
        moved.update(source_path="/different/mount", view_fingerprint="f" * 64, total_windows=12)
        self.assertEqual(normalization.parent_fingerprint(moved), normalization.parent_fingerprint(self.meta))
        self.load(meta=moved)
        moved["rule_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "parent"):
            self.load(meta=moved)

    def test_tampering_fingerprint_detected(self):
        artifact = fixture_statistics()
        artifact["state"]["mean"][0] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.load(artifact)

    def test_resigned_invalid_statistics_are_rejected(self):
        for category, field, value in [
                ("state", "mean", [0] * 15), ("state", "variance", [-1] * 16),
                ("state", "scale", [0] * 16), ("state", "scale", [2] * 16),
                ("state", "constant_dimensions", []), ("state", "count", True),
                ("action", "count", 159), ("action", "mean", ["zero"] * 16)]:
            artifact = fixture_statistics()
            artifact[category][field] = value
            resign(artifact)
            with self.subTest(category=category, field=field):
                with self.assertRaises(ValueError):
                    self.load(artifact)

    def test_nonfinite_json_rejected(self):
        artifact = fixture_statistics()
        artifact["action"]["scale"][0] = float("inf")
        self.path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaises(ValueError):
            UMINormalizer("mean_std", self.path, access_meta=self.meta, current_view_fingerprint="e" * 64)

    def test_field_order_representation_and_metadata_mismatch(self):
        artifact = fixture_statistics()
        artifact["representation"]["field_order"].reverse()
        with self.assertRaisesRegex(ValueError, "representation"):
            self.load(resign(artifact))
        artifact = fixture_statistics()
        artifact["representation"]["version"] = "another"
        with self.assertRaisesRegex(ValueError, "representation"):
            self.load(resign(artifact))
        for field, value in [("signals", list(reversed(normalization.SIGNALS))),
                             ("signal_widths", [8, 0, 8, 0]), ("horizon", 8)]:
            meta = fixture_meta()
            meta[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.load(meta=meta)

    def test_provenance_is_a_copy(self):
        normalizer = self.load()
        first = normalizer.provenance()
        first["fit_details"]["selection"] = "mutated"
        first["field_order"][0] = "mutated"
        second = normalizer.provenance()
        self.assertEqual(second["fit_details"]["selection"], "synthetic fixture")
        self.assertEqual(second["field_order"][0], "robot1.x")
        self.assertEqual(len(second["statistics_file_sha256"]), 64)

    def test_formal_rejects_engineering_artifact(self):
        with self.assertRaisesRegex(ValueError, "purpose"):
            self.load(purpose="formal", contract=fixture_contract())

    def test_formal_contract_accepts_validation_apply_and_rejects_other_view(self):
        contract = fixture_contract()
        artifact = fixture_statistics(purpose="formal", contract=contract)
        self.load(artifact, purpose="formal", contract=contract, view="e" * 64)
        with self.assertRaisesRegex(ValueError, "not approved"):
            self.load(artifact, purpose="formal", contract=contract, view="f" * 64)
        wrong = deepcopy(contract)
        wrong["experiment_id"] = "different-experiment"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.load(artifact, purpose="formal", contract=wrong)
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.load(artifact, purpose="formal")

    def test_formal_fit_requires_explicit_training_pool_approval(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            fixture_statistics(purpose="formal")
        contract = fixture_contract()
        contract["training_pool_approved"] = False
        with self.assertRaisesRegex(ValueError, "approved"):
            normalization.validate_experiment_contract(contract, access_meta=self.meta,
                                                        fit_view_fingerprint="d" * 64)
        with self.assertRaisesRegex(ValueError, "Engineering"):
            fixture_statistics(contract=fixture_contract())


if __name__ == "__main__":
    unittest.main()
