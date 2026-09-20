"""Checkpoint publication/failure tests without models, source data or devices."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from starVLA.training.trainer_utils.umi_checkpoint import (
    UMICheckpoints, inspect_checkpoint, trim_uncommitted_log, sha256_file, write_json,
)
from starVLA.training.trainer_utils.umi_training_state import (
    UMITrainingState, checkpoint_config, checkpoint_reasons,
)


class FakeAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1

    def __init__(self):
        self.fail = False

    def wait_for_everyone(self):
        pass

    def save_state(self, directory, safe_serialization):
        directory = Path(directory)
        torch.save({"parameter": torch.ones(2)}, directory / "pytorch_model.bin")
        if self.fail:
            raise OSError("Injected crash during checkpoint write")
        for name in ("optimizer.bin", "custom_checkpoint_0.pkl", "custom_checkpoint_1.pkl", "random_states_0.pkl"):
            torch.save({"test": 1}, directory / name)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = {"runtime": {"world_size": 1}, "view": "view_A", "statistics_sha256": "stats_A", "plan": "plan_A"}
        self.state = UMITrainingState([4], [23], 4)
        self.accelerator = FakeAccelerator()
        self.manager = UMICheckpoints(self.accelerator, self.root, self.identity, self.state)

    def tearDown(self):
        self.temp.cleanup()

    def test_half_write_never_publishes_or_replaces_latest(self):
        self.state.commit_update()
        saved = self.manager.save()
        before = (self.root / "latest.json").read_bytes()
        self.state.commit_update()
        self.accelerator.fail = True
        with self.assertRaisesRegex(OSError, "Injected"):
            self.manager.save()
        self.assertEqual((self.root / "latest.json").read_bytes(), before)
        partials = list((self.root / "checkpoints").glob(".*partial*"))
        self.assertEqual(len(partials), 1)
        with self.assertRaises(ValueError):
            inspect_checkpoint(partials[0], self.identity)
        self.assertEqual(self.manager.resolve("latest")["path"], saved)
        self.accelerator.fail = False
        retry = self.manager.save()
        self.assertEqual(inspect_checkpoint(retry, self.identity)["global_update_step"], 2)

    def test_rejects_view_statistics_plan_and_world_changes(self):
        self.state.commit_update()
        path = self.manager.save()
        for key in ("view", "statistics_sha256", "plan", "runtime"):
            changed = deepcopy(self.identity)
            changed[key] = {"world_size": 2} if key == "runtime" else "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                inspect_checkpoint(path, changed)

    def test_rejects_corrupted_bytes_weights_only_and_missing_path(self):
        self.manager.integrity = "full"
        self.state.commit_update()
        path = Path(self.manager.save())
        weights = path / "pytorch_model.bin"
        with self.assertRaises(ValueError):
            inspect_checkpoint(weights, self.identity)
        with self.assertRaises(ValueError):
            inspect_checkpoint(self.root / "nonexistent", self.identity)
        with weights.open("r+b") as stream:
            stream.seek(20)
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            inspect_checkpoint(path, self.identity)

    def test_basic_never_hashes_model_optimizer_and_records_guarantee(self):
        def metadata_only(path):
            self.assertNotIn(Path(path).name, ("pytorch_model.bin", "optimizer.bin"))
            return sha256_file(path)
        self.state.commit_update()
        with patch("starVLA.training.trainer_utils.umi_checkpoint.sha256_file", side_effect=metadata_only):
            path = Path(self.manager.save(reasons=["periodic", "stage_boundary"]))
            self.assertEqual(inspect_checkpoint(path, self.identity), self.state.state_dict())
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertEqual(manifest["integrity"], "basic")
        self.assertEqual(manifest["save_reasons"], ["periodic", "stage_boundary"])
        self.assertEqual(manifest["files"]["optimizer.bin"]["check"], "size")
        self.assertNotIn("sha256", manifest["files"]["optimizer.bin"])
        # Document the intentional residual risk: equal-length content mutation
        # in a size-only payload is NOT detected by this layer.
        weights = path / "pytorch_model.bin"
        data = bytearray(weights.read_bytes()); data[20] ^= 1; weights.write_bytes(data)
        self.assertEqual(inspect_checkpoint(path, self.identity), self.state.state_dict())

    def test_both_modes_reject_missing_truncated_and_changed_small_files(self):
        for mode in ("basic", "full"):
            with self.subTest(mode=mode):
                manager = UMICheckpoints(self.accelerator, self.root / mode, self.identity,
                                         self.state, integrity=mode)
                path = Path(manager.save())
                weights = path / "pytorch_model.bin"
                original = weights.read_bytes()
                weights.unlink()
                with self.assertRaisesRegex(ValueError, "Invalid checkpoint file"):
                    inspect_checkpoint(path, self.identity)
                weights.write_bytes(original[:-1])
                with self.assertRaisesRegex(ValueError, "truncated"):
                    inspect_checkpoint(path, self.identity)
                weights.write_bytes(original)
                state = path / "training_state.json"
                data = bytearray(state.read_bytes()); data[2] ^= 1; state.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    inspect_checkpoint(path, self.identity)

    def rewrite_manifest(self, path, manifest):
        write_json(path / "manifest.json", manifest)
        write_json(path / "COMPLETED.json", {"manifest_sha256": sha256_file(path / "manifest.json")})

    def test_legacy_v1_always_requires_full_hashes(self):
        self.manager.integrity = "full"
        path = Path(self.manager.save())
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["version"] = "umi-full-checkpoint-v1"
        manifest.pop("integrity")
        for record in manifest["files"].values():
            record.pop("check")
        self.rewrite_manifest(path, manifest)  # Synthetic legacy fixture only.
        self.assertEqual(inspect_checkpoint(path, self.identity), self.state.state_dict())
        self.manager.integrity = "basic"
        self.assertEqual(self.manager.resolve(str(path))["state"], self.state.state_dict())
        manifest["files"]["optimizer.bin"].pop("sha256")
        self.rewrite_manifest(path, manifest)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            inspect_checkpoint(path, self.identity)

    def test_full_cannot_silently_downgrade_file_checks(self):
        self.manager.integrity = "full"
        path = Path(self.manager.save())
        original = json.loads((path / "manifest.json").read_text())
        for change in (lambda m: m.pop("integrity"),
                       lambda m: m["files"]["optimizer.bin"].pop("sha256"),
                       lambda m: m["files"]["optimizer.bin"].update(check="size")):
            manifest = deepcopy(original); change(manifest); self.rewrite_manifest(path, manifest)
            with self.assertRaises(ValueError):
                inspect_checkpoint(path, self.identity)

    def test_save_triggers_are_combined_and_config_is_explicit(self):
        reasons = checkpoint_reasons(12, 6, pause=True, complete=True, stage_boundary=True)
        self.assertEqual(set(reasons), {"periodic", "pause", "complete", "stage_boundary"})
        path = self.manager.save(reasons=reasons)
        self.assertEqual(self.manager.save(reasons=reasons), path)
        self.assertEqual(len(list((self.root / "checkpoints").glob("update_*"))), 1)
        self.assertEqual(checkpoint_config({"training": {"save_every": 6}}),
                         {"integrity": "basic", "every_updates": 6})
        with self.assertRaisesRegex(ValueError, "not both"):
            checkpoint_config({"training": {"save_every": 6}, "checkpoint": {"every_updates": 6}})

    def test_streaming_log_recovery_discards_only_uncommitted_suffix(self):
        path = self.root / "updates.jsonl"
        committed = b'{"global_update": 1}\n{"global_update": 2}\n'
        path.write_bytes(committed + b'{"global_update": 3}\n{"global_upd')
        result = trim_uncommitted_log(path, 2)
        self.assertGreater(result["truncated_bytes"], 0)
        self.assertEqual(path.read_bytes(), committed)
        path.write_bytes(committed + b'{"global_upd')
        self.assertTrue(trim_uncommitted_log(path, 2)["torn_final_line"])
        self.assertEqual(path.read_bytes(), committed)
        path.write_bytes(b'{bad json}\n{"global_update": 2}\n')
        with self.assertRaisesRegex(ValueError, "Interior"):
            trim_uncommitted_log(path, 2)

    def test_complete_record_missing_only_newline_can_be_appended(self):
        path = self.root / "trace.jsonl"
        path.write_bytes(b'{"global_update": 1}')
        self.assertTrue(trim_uncommitted_log(path, 1)["restored_final_newline"])
        with path.open("ab") as stream:
            stream.write(b'{"global_update": 2}\n')
        self.assertEqual([json.loads(line)["global_update"] for line in path.read_text().splitlines()], [1, 2])


if __name__ == "__main__":
    unittest.main()
