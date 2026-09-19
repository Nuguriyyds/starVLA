"""Factory and cache ownership checks; no video or model is used."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.umi_indexed_dataset import (
    UMINormalizedDataset, apply_umi_normalization, make_umi_dataloader,
)
from starVLA.dataloader.umi_normalization import (
    SIGNALS, WIDTHS, RAW_REPRESENTATION, WeightedMoments, build_statistics,
)


def example_meta():
    return {"signals": list(SIGNALS), "signal_widths": list(WIDTHS), "horizon": 16,
            "representation": RAW_REPRESENTATION, "rule_fingerprint": "a" * 64,
            "catalog_artifacts_sha256": {"manifest.parquet": "b" * 64},
            "window_contract": {"source_info_sha256": "c" * 64}}


class RawExampleDataset:
    """Intentionally returns cache-owned arrays to expose in-place mutations."""
    def __init__(self):
        self.meta = example_meta()
        self.view_fingerprint = "e" * 64  # deliberately differs from fit view
        self.state = np.arange(16, dtype=np.float32).reshape(1, 16)
        self.action = np.arange(256, dtype=np.float32).reshape(16, 16) / 100
        self.trace = {"episode_index": 8, "view_fingerprint": self.view_fingerprint}

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"state": self.state, "action": self.action, "lang": "synthetic task",
                "image": [], "umi_metadata": self.trace, "robot_tag": "new_embodiment"}

    def read_lowdim(self, index):
        return {"state": self.state, "action": self.action}

    def provenance(self):
        return {"normalization": "none", "view_fingerprint": self.view_fingerprint}

    def close(self):
        pass


class NormalizationDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = RawExampleDataset()
        values = np.arange(32, dtype=np.float32).reshape(2, 16)
        state, action = WeightedMoments(), WeightedMoments()
        state.update(values)
        action.update(np.repeat(values, 16, axis=0))
        statistics = build_statistics(state, action, access_meta=self.raw.meta,
                                      fit_view_fingerprint="d" * 64, purpose="engineering",
                                      fit_details={"selection": "synthetic"}, code_version="test")
        self.statistics = self.root / "statistics.json"
        self.statistics.write_text(json.dumps(statistics), encoding="utf-8")
        self.data = {"normalization": "mean_std", "normalization_statistics": str(self.statistics),
                     "normalization_purpose": "engineering"}

    def tearDown(self):
        self.temp.cleanup()

    def test_none_retains_original_dataset_and_values(self):
        result = apply_umi_normalization(self.raw, {"normalization": "none"})
        self.assertIs(result, self.raw)
        np.testing.assert_array_equal(result[0]["action"], self.raw.action)
        with self.assertRaises(ValueError):
            apply_umi_normalization(self.raw, {"normalization": "none", "normalization_statistics": str(self.statistics)})

    def test_wrapper_does_not_mutate_cached_arrays_or_trace(self):
        original_state, original_action = self.raw.state.copy(), self.raw.action.copy()
        wrapped = apply_umi_normalization(self.raw, self.data)
        self.assertIsInstance(wrapped, UMINormalizedDataset)
        first, second = wrapped[0], wrapped[0]
        for key, original in (("state", original_state), ("action", original_action)):
            self.assertEqual(first[key].dtype, np.float32)
            self.assertEqual(first[key].shape, original.shape)
            self.assertFalse(np.shares_memory(first[key], self.raw.read_lowdim(0)[key]))
            np.testing.assert_array_equal(first[key], second[key])
            np.testing.assert_allclose(getattr(wrapped.normalizer, "inverse_" + key)(first[key]),
                                       original, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(self.raw.read_lowdim(0)[key], original)
        self.assertNotIn("normalization", self.raw.trace)
        self.assertIn("normalization", first["umi_metadata"])
        clone = pickle.loads(pickle.dumps(wrapped))
        np.testing.assert_array_equal(clone[0]["action"], first["action"])

    def test_factory_records_actual_model_space_and_fit_apply_identity(self):
        data = dict(self.data, index_dir="synthetic", num_workers=0, shuffle=False,
                    drop_last=False, per_device_batch_size=2)
        cfg = OmegaConf.create({"framework": {"action_model": {"state_dim": 16, "action_dim": 16, "action_horizon": 16}},
                               "datasets": {"vla_data": data}, "output_dir": str(self.root / "run")})
        with patch("starVLA.dataloader.umi_indexed_dataset.UMIIndexedDataset", return_value=self.raw):
            loader = make_umi_dataloader(cfg)
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)
        saved = json.loads((self.root / "run/dataset_access.json").read_text())
        self.assertEqual(saved["normalization"], "mean_std")
        self.assertEqual(saved["raw_reader_normalization"], "none")
        self.assertEqual(saved["view_fingerprint"], "e" * 64)
        self.assertIn("d" * 64, json.dumps(saved["normalization_details"]))
        self.assertIn("e" * 64, json.dumps(saved["normalization_details"]))

    def test_missing_file_and_engineering_stats_in_formal_config_fail(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            apply_umi_normalization(self.raw, dict(self.data, normalization_statistics=str(self.root / "absent.json")))
        with self.assertRaises(ValueError):
            apply_umi_normalization(self.raw, dict(self.data, normalization_purpose="formal"))
        with self.assertRaises(ValueError):
            apply_umi_normalization(self.raw, {"normalization": "mean_std"})
        with self.assertRaises(ValueError):
            apply_umi_normalization(self.raw, {"normalization": "unsupported"})


if __name__ == "__main__":
    unittest.main()
