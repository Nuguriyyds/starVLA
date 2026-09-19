"""Metadata-only regression tests for the production UMI catalog builder."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

TOOLS = Path(__file__).resolve().parents[1] / "examples/umi_pretrain/tools"
sys.path.insert(0, str(TOOLS))
from build_umi_catalog import build_catalog

CAMERAS = (
    "observation.images.head_left",
    "observation.images.head_right",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


class UMICatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "meta/episodes/chunk-000").mkdir(parents=True)

    def episode(self, episode_index, start, length, tasks=None, source_mcap="/source/capture-a.mcap"):
        row = {
            "episode_index": episode_index,
            "tasks": ["pick cup"] if tasks is None else tasks,
            "length": length,
            "data/chunk_index": 3,
            "data/file_index": 7,
            "dataset_from_index": start,
            "dataset_to_index": start + length,
            "source_mcap": source_mcap,
        }
        for index, camera in enumerate(CAMERAS):
            prefix = f"videos/{camera}/"
            offset = 0.125 + index * 0.25
            row.update({
                prefix + "chunk_index": 2 + index,
                prefix + "file_index": 10 + index,
                prefix + "from_timestamp": offset,
                prefix + "to_timestamp": offset + length / 30,
                prefix + "source_frame_count": length + 20 + index,
            })
        return row

    def prepare(self, rows=None, tasks=None, separate_files=False):
        rows = rows or [
            self.episode(41, 0, 1000),
            self.episode(42, 1000, 20, ["pick cup", "place cup"]),
        ]
        tasks = tasks or [
            {"task_index": 10, "task": "pick cup"},
            {"task_index": 20, "task": "place cup"},
        ]
        info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "total_episodes": len(rows),
            "total_frames": sum(row["length"] for row in rows),
            "total_tasks": len(tasks),
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                camera: {
                    "dtype": "video", "shape": [1300, 1600, 3],
                    "names": ["height", "width", "channels"],
                    "info": {"video.fps": 30},
                }
                for camera in CAMERAS
            },
        }
        (self.source / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        pq.write_table(pa.Table.from_pylist(tasks), self.source / "meta/tasks.parquet")
        groups = [[row] for row in rows] if separate_files else [rows]
        for index, group in enumerate(groups):
            pq.write_table(
                pa.Table.from_pylist(group),
                self.source / f"meta/episodes/chunk-000/file-{index:03d}.parquet",
            )
        # Deliberately invalid payloads: catalog construction must not open or
        # decode these. File existence checks are the only permitted operation.
        for row in rows:
            data_path = self.source / info["data_path"].format(
                chunk_index=row["data/chunk_index"], file_index=row["data/file_index"],
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.touch()
            for camera in CAMERAS:
                prefix = f"videos/{camera}/"
                video_path = self.source / info["video_path"].format(
                    video_key=camera,
                    chunk_index=row[prefix + "chunk_index"],
                    file_index=row[prefix + "file_index"],
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.touch()
        return rows

    def build(self, name="catalog", **kwargs):
        output = self.root / name
        report = build_catalog(self.source, output, **kwargs)
        tables = [pq.read_table(path) for path in sorted((output / "episode_manifest").glob("part-*.parquet"))]
        rows = pa.concat_tables(tables).to_pylist() if tables else []
        return output, report, rows

    def test_shared_payload_uses_episode_filter_not_global_row_offset(self):
        self.prepare()
        _, report, rows = self.build()
        self.assertEqual(report["total_episodes"], 2)
        self.assertEqual(report["total_frames"], 1020)
        episode = next(row for row in rows if row["episode_index"] == 42)
        self.assertEqual(episode["dataset_from_index"], 1000)
        self.assertEqual(episode["dataset_to_index"], 1020)
        self.assertEqual(episode["data_path"], "data/chunk-003/file-007.parquet")
        self.assertEqual(episode["data_locator"], "episode_index_filter")
        self.assertIsNone(episode["data_local_row_from"])
        self.assertIsNone(episode["data_local_row_to"])

    def test_independent_camera_files_offsets_and_counts_survive(self):
        original = self.prepare()
        _, _, rows = self.build()
        episode = next(row for row in rows if row["episode_index"] == 42)
        self.assertEqual([camera["camera"] for camera in episode["cameras"]], list(CAMERAS))
        for index, camera in enumerate(episode["cameras"]):
            prefix = f"videos/{CAMERAS[index]}/"
            self.assertEqual(camera["video_path"], f"videos/{CAMERAS[index]}/chunk-{2+index:03d}/file-{10+index:03d}.mp4")
            for field in ("from_timestamp", "to_timestamp", "source_frame_count"):
                self.assertEqual(camera[field], original[1][prefix + field])

    def test_sparse_task_ids_and_multi_task_original_text(self):
        self.prepare()
        output, _, rows = self.build()
        episode = next(row for row in rows if row["episode_index"] == 42)
        self.assertEqual(episode["task_ids"], [10, 20])
        self.assertEqual(episode["task_texts"], ["pick cup", "place cup"])
        tasks = pq.read_table(output / "task_catalog.parquet").to_pylist()
        self.assertEqual({row["task_index"] for row in tasks}, {10, 20})

    def test_missing_source_is_not_guessed_from_task_or_file_number(self):
        self.prepare([self.episode(90, 0, 20, ["pick cup"], source_mcap=None)])
        _, _, rows = self.build()
        episode = rows[0]
        self.assertIsNone(episode["source_mcap"])
        self.assertIsNone(episode["source_recording_key"])
        self.assertIsNone(episode["source_set_id"])
        self.assertIsNone(episode["source_trim_frame_start"])
        self.assertIsNone(episode["source_trim_frame_stop"])
        for name in ("scene", "session", "collector", "task_category"):
            self.assertIsNone(episode.get(name))

    def test_same_explicit_recording_shares_group_key(self):
        self.prepare()
        _, _, rows = self.build()
        self.assertEqual(rows[0]["source_mcap"], "/source/capture-a.mcap")
        self.assertTrue(rows[0]["source_recording_key"])
        self.assertEqual(rows[0]["source_recording_key"], rows[1]["source_recording_key"])

    def test_merge_mapping_preserves_task_ids_and_source_set(self):
        self.prepare(
            [self.episode(42, 1000, 20, ["place cup"])],
            tasks=[
                {"task_index": 10, "task": "place cup"},
                {"task_index": 20, "task": "place cup"},
            ],
        )
        merge = self.source / "meta/merge"
        merge.mkdir()
        pq.write_table(pa.Table.from_pylist([{
            "global_episode_index": 42,
            "set_id": "set-a",
            "source_mcap": "/source/capture-a.mcap",
            "global_task_indices": [20],
            "tasks": ["place cup"],
            "global_dataset_from_index": 1000,
            "global_dataset_to_index": 1020,
            "global_data_chunk_index": 3,
            "global_data_file_index": 7,
        }]), merge / "episode_mapping.parquet")
        _, _, rows = self.build()
        self.assertEqual(rows[0]["task_ids"], [20])
        self.assertEqual(rows[0]["task_texts"], ["place cup"])
        self.assertEqual(rows[0]["source_set_id"], "set-a")

    def test_duplicate_episode_ids_are_reported(self):
        self.prepare([
            self.episode(9, 0, 20),
            self.episode(9, 20, 20),
        ])
        _, report, rows = self.build(batch_size=1)
        self.assertTrue(report["issue_counts"])
        issues = json.dumps([row["issues"] for row in rows]).lower()
        self.assertIn("duplicate", issues)

    def test_bad_length_span_and_unknown_task_are_reported(self):
        row = self.episode(5, 0, 20, ["task without mapping"])
        row["dataset_to_index"] = 19
        self.prepare([row])
        _, report, rows = self.build()
        self.assertTrue(report["issue_counts"])
        issues = json.dumps(rows[0]["issues"]).lower()
        self.assertTrue("length" in issues or "span" in issues, issues)
        self.assertIn("task", issues)
        self.assertEqual(rows[0]["task_ids"], [])

    def test_limited_scan_never_claims_full_source_counts(self):
        self.prepare(separate_files=True)
        _, report, rows = self.build(max_meta_files=1, batch_size=1)
        self.assertFalse(report["scan_complete"])
        self.assertEqual(report["total_episodes"], 1)
        self.assertEqual(report["total_frames"], 1000)
        self.assertEqual(len(rows), 1)

    def test_shifted_complete_global_range_is_not_hidden_by_equal_frame_total(self):
        rows = [
            self.episode(41, 7, 1000),
            self.episode(42, 1007, 20, ["pick cup", "place cup"]),
        ]
        self.prepare(rows)
        _, report, _ = self.build()
        self.assertTrue(report["scan_complete"])
        self.assertEqual(report["total_frames"], 1020)
        self.assertGreater(report["issue_counts"].get("global_range_start_mismatch", 0), 0)
        self.assertGreater(report["issue_counts"].get("global_range_end_mismatch", 0), 0)

    def test_complete_scan_is_streamed_to_requested_parts(self):
        self.prepare(separate_files=True)
        output, report, rows = self.build(batch_size=1, part_rows=1)
        self.assertTrue(report["scan_complete"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(list((output / "episode_manifest").glob("part-*.parquet"))), 2)
        for name in ("catalog_config.json", "schema_report.json", "catalog_report.json"):
            self.assertIsInstance(json.loads((output / name).read_text()), dict)

    def test_output_cannot_be_within_source_or_overwrite_existing_directory(self):
        self.prepare()
        with self.assertRaises((ValueError, FileExistsError)):
            build_catalog(self.source, self.source / "generated")
        self.assertFalse((self.source / "generated").exists())
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            build_catalog(self.source, existing)
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_check_files_never_parses_empty_payloads(self):
        self.prepare()
        _, report, rows = self.build(check_files=True)
        self.assertTrue(report["scan_complete"])
        self.assertEqual(len(rows), 2)
        self.assertFalse(report["issue_counts"], report["issue_counts"])

    def test_missing_referenced_file_is_reported(self):
        self.prepare()
        missing = self.source / "videos/observation.images.wrist_right/chunk-005/file-013.mp4"
        missing.unlink()
        output, report, _ = self.build(check_files=True)
        self.assertEqual(report["issue_counts"]["missing_referenced_file"], 1)
        self.assertEqual(report["referenced_files"]["missing"], 1)
        inventory = pq.read_table(output / "file_inventory.parquet").to_pylist()
        absent = [row["path"] for row in inventory if row["present"] == 0]
        self.assertEqual(absent, [missing.relative_to(self.source).as_posix()])


if __name__ == "__main__":
    unittest.main()
