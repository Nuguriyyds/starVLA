"""Guard raw FP32 packing without changing other datasets' default precision."""
import unittest
import numpy as np
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class LowdimPackingTests(unittest.TestCase):
    def make_dataset(self, dtype=None):
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {"include_state": True}
        if dtype is not None:
            dataset.data_cfg["lowdim_dtype"] = dtype
        dataset._modality_keys = {
            "video": [], "language": ["annotation.task"],
            "state": ["state.left", "state.right"],
            "action": ["action.left", "action.right"],
        }
        dataset.tag = "test"
        values = np.array([[0.12345679, -0.00012345679]], dtype=np.float32)
        data = {"annotation.task": ["move"]}
        for group in ("state", "action"):
            data[f"{group}.left"] = values.copy()
            data[f"{group}.right"] = values.copy() * 2
        return dataset, data, np.concatenate([values, values * 2], axis=1)

    def test_opt_in_fp32_preserves_values_before_model(self):
        dataset, data, expected = self.make_dataset("float32")
        self.assertFalse(np.array_equal(expected, expected.astype(np.float16).astype(np.float32)))
        sample = dataset._pack_sample(data)
        for key in ("state", "action"):
            self.assertEqual(sample[key].dtype, np.float32)
            np.testing.assert_array_equal(sample[key], expected)

    def test_default_retains_repository_fp16(self):
        dataset, data, expected = self.make_dataset()
        sample = dataset._pack_sample(data)
        for key in ("state", "action"):
            self.assertEqual(sample[key].dtype, np.float16)
            np.testing.assert_array_equal(sample[key], expected.astype(np.float16))

    def test_non_float_dtype_rejected(self):
        dataset, data, _ = self.make_dataset("int16")
        with self.assertRaisesRegex(ValueError, "lowdim_dtype"):
            dataset._pack_sample(data)


if __name__ == "__main__":
    unittest.main()
