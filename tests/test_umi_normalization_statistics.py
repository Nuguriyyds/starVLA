"""Synthetic low-dimensional scan tests; no video, model, GPU or public data."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/umi_pretrain/tools"))
import compute_umi_normalization as tool
from starVLA.dataloader import umi_normalization as norm


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.source, self.index = self.root / "source", self.root / "index"
        self.source.mkdir()
        self.index.mkdir()
        self.intervals = {10: [(0, 3), (30, 32)], 20: [(2, 5)]}
        self.values = {ep: (np.arange(length * 16, dtype=np.float64).reshape(length, 16) / 997 + ep)
                       .astype(np.float32) for ep, length in [(10, 65), (20, 30)]}
        # Genuine holes contain invalid signals. They must never enter moments.
        for ep, values in self.values.items():
            keep = np.zeros(len(values), dtype=bool)
            for a, b in self.intervals[ep]:
                keep[a:b+16] = True
            values[~keep] = np.nan
        self.rows = {}
        for ep, values in self.values.items():
            rows = []
            for i, value in enumerate(values):
                rows.append({"episode_index": ep, "frame_index": ep * 100 + i,
                             "timestamp": i / 30, "task_index": ep // 10 - 1,
                             norm.SIGNALS[0]: value[:7].tolist(), norm.SIGNALS[1]: float(value[7]),
                             norm.SIGNALS[2]: value[8:15].tolist(), norm.SIGNALS[3]: float(value[15])})
            self.rows[ep] = rows
        unrelated = deepcopy(self.rows[10][20:24])
        for row in unrelated:
            row["episode_index"] = 99
        self.files = {"a.parquet": self.rows[10][:15] + unrelated,
                      "b.parquet": self.rows[20] + self.rows[10][15:]}
        fields = [("episode_index", pa.int64()), ("frame_index", pa.int64()),
                  ("timestamp", pa.float64()), ("task_index", pa.int64())]
        fields.extend((key, pa.list_(pa.float32()) if width > 1 else pa.float32())
                      for key, width in zip(norm.SIGNALS, norm.WIDTHS))
        self.schema = pa.schema(fields)
        for filename, rows in self.files.items():
            pq.write_table(pa.Table.from_pylist(rows, schema=self.schema), self.source / filename,
                           row_group_size=7)
        self.db = sqlite3.connect(self.index / "metadata.sqlite3")
        self.db.executescript("""
            CREATE TABLE episodes(episode_index INTEGER PRIMARY KEY,payload TEXT);
            CREATE TABLE segments(episode_index INTEGER,data_file TEXT,file_row_start INTEGER,
                file_row_end_exclusive INTEGER,episode_row_offset_start INTEGER,
                episode_row_offset_end_exclusive INTEGER,row_group_start INTEGER,row_group_end_inclusive INTEGER);
            CREATE TABLE tasks(task_index INTEGER PRIMARY KEY,task TEXT);
            CREATE TABLE data_files(data_file TEXT PRIMARY KEY,size_bytes INTEGER,mtime_ns INTEGER);
        """)
        self.db.executemany("INSERT INTO episodes VALUES (?,?)",
                            [(ep, json.dumps({"length": len(v)})) for ep, v in self.values.items()])
        self.db.executemany("INSERT INTO tasks VALUES (?,?)", [(0, "task zero"), (1, "task one")])
        self.db.executemany("INSERT INTO segments VALUES (?,?,?,?,?,?,?,?)", [
            (10, "a.parquet", 0, 15, 0, 15, 0, 2),
            (10, "b.parquet", 30, 80, 15, 65, 4, 11),
            (20, "b.parquet", 0, 30, 0, 30, 0, 4)])
        for name in self.files:
            stat = (self.source / name).stat()
            self.db.execute("INSERT INTO data_files VALUES (?,?,?)", (name, stat.st_size, stat.st_mtime_ns))
        self.db.commit()
        self.db.close()
        ranges = np.array([(ep, a, b) for ep, intervals in self.intervals.items()
                           for a, b in intervals], dtype=np.int64)
        cumulative = np.cumsum(ranges[:, 2] - ranges[:, 1])
        np.save(self.index / "ranges.npy", ranges)
        np.save(self.index / "cumulative.npy", cumulative)
        self.meta = {"version": "roban-umi-access-v1", "status": "completed", "horizon": 16,
                     "source_path": str(self.source), "fps": 30.0, "total_ranges": len(ranges),
                     "total_windows": int(cumulative[-1]), "signals": list(norm.SIGNALS),
                     "signal_widths": list(norm.WIDTHS), "representation": norm.RAW_REPRESENTATION,
                     "video_keys": ["observation.images." + n for n in
                                    ("head_left", "head_right", "wrist_left", "wrist_right")],
                     "rule_fingerprint": "1" * 64, "catalog_artifacts_sha256": {"manifest": "2" * 64},
                     "window_contract": {"source_info_sha256": "3" * 64}, "index_order": "episode,offset"}
        self.refresh_index_hashes()

    def refresh_index_hashes(self):
        self.meta["artifacts"] = {name: {"sha256": tool.file_hash(self.index / name),
                                         "size_bytes": (self.index / name).stat().st_size}
                                  for name in ("ranges.npy", "cumulative.npy", "metadata.sqlite3")}
        self.meta["view_fingerprint"] = tool.compute_view_fingerprint(self.meta)
        (self.index / "meta.json").write_text(json.dumps(self.meta), encoding="utf-8")

    def rewrite_file(self, filename):
        pq.write_table(pa.Table.from_pylist(self.files[filename], schema=self.schema),
                       self.source / filename, row_group_size=7)
        stat = (self.source / filename).stat()
        db = sqlite3.connect(self.index / "metadata.sqlite3")
        db.execute("UPDATE data_files SET size_bytes=?,mtime_ns=? WHERE data_file=?",
                   (stat.st_size, stat.st_mtime_ns, filename))
        db.commit()
        db.close()
        self.refresh_index_hashes()

    def reference(self):
        states, actions = [], []
        for ep, intervals in self.intervals.items():
            for a, b in intervals:
                for anchor in range(a, b):
                    states.append(self.values[ep][anchor])
                    actions.extend(self.values[ep][anchor+1:anchor+17])
        return np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fx = Fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def compute(self, name="stats.json", **kwargs):
        return tool.compute(self.fx.index, self.root / name, batch_rows=5, **kwargs)

    def test_compact_weights_equal_expanded_window_occurrences(self):
        weights = tool.AnchorWeights([(0, 3), (10, 12), (41, 42)])
        rows = np.arange(-5, 70)
        sw, aw = weights.weights(rows)
        anchors = [0, 1, 2, 10, 11, 41]
        np.testing.assert_array_equal(sw, [sum(r == a for a in anchors) for r in rows])
        np.testing.assert_array_equal(aw, [sum(a < r <= a+16 for a in anchors) for r in rows])
        self.assertEqual(int(sw.sum()), 6)
        self.assertEqual(int(aw.sum()), 96)

    def test_stats_match_per_window_reference_across_rows_files_and_episodes(self):
        result = self.compute()
        state, action = self.fx.reference()
        self.assertEqual(result["purpose"], "engineering")
        self.assertEqual(result["state"]["count"], 8)
        self.assertEqual(result["action"]["count"], 128)
        for name, values in (("state", state), ("action", action)):
            np.testing.assert_allclose(result[name]["mean"], values.mean(axis=0), rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(result[name]["variance"], values.var(axis=0), rtol=1e-14, atol=1e-14)
        self.assertLess(result["fit_details"]["selected_physical_rows"], len(state) + len(action))

    def test_completed_file_resume_is_exact(self):
        baseline = self.compute("baseline.json")
        def interrupt(_):
            raise KeyboardInterrupt("simulated stop after committed file")
        with self.assertRaises(KeyboardInterrupt):
            self.compute("resume.json", after_file=interrupt)
        self.assertFalse((self.root / "resume.json").exists())
        with patch.object(tool, "scan_file", wraps=tool.scan_file) as scanned:
            resumed = self.compute("resume.json", resume=True)
        self.assertEqual(scanned.call_count, 1)
        self.assertEqual(resumed, baseline)

    def test_failed_file_is_not_committed_or_counted_twice(self):
        original = tool.scan_file
        def fail_second(path, *args, **kwargs):
            if path.name == "b.parquet":
                raise RuntimeError("simulated partial file failure")
            return original(path, *args, **kwargs)
        with patch.object(tool, "scan_file", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "partial file"):
                self.compute("failed.json")
        self.assertEqual(len(list((self.root / "failed.json.work/completed").glob("*.json"))), 1)
        result = self.compute("failed.json", resume=True)
        self.assertEqual(result["state"]["count"], 8)
        self.assertEqual(result["action"]["count"], 128)

    def test_resume_rejects_changed_parameters(self):
        with self.assertRaises(KeyboardInterrupt):
            self.compute("resume.json", after_file=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.compute("resume.json", resume=True, min_std=0.1)

    def test_resume_rejects_changed_code(self):
        with self.assertRaises(KeyboardInterrupt):
            self.compute("resume.json", after_file=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
        hash_file = tool.file_hash
        def changed(path):
            return "f" * 64 if str(path) == str(tool.__file__) else hash_file(path)
        with patch.object(tool, "file_hash", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                self.compute("resume.json", resume=True)

    def test_lock_rejects_second_writer_and_is_released_on_failure(self):
        lock = self.root / "stats.json.lock"
        with tool.exclusive_lock(lock):
            with self.assertRaisesRegex(RuntimeError, "build lock exists"):
                self.compute()
        self.assertFalse(lock.exists())
        with self.assertRaises(KeyboardInterrupt):
            self.compute(after_file=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertFalse(lock.exists())
        self.compute(resume=True)

    def test_resume_rejects_corrupt_committed_moments(self):
        with self.assertRaises(KeyboardInterrupt):
            self.compute(after_file=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
        completed = next((self.root / "stats.json.work/completed").glob("*.json"))
        result = tool.read_json(completed)
        result["state"]["mean"][0] += 123
        completed.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content fingerprint mismatch"):
            self.compute(resume=True)

    def test_selected_nan_rejected_unselected_nan_allowed(self):
        # The fixture already has many unselected NaNs. One selected NaN must fail.
        self.fx.files["a.parquet"][2][norm.SIGNALS[1]] = float("nan")
        self.fx.rewrite_file("a.parquet")
        with self.assertRaisesRegex(ValueError, "Invalid selected FP32 signal"):
            self.compute()

    def test_cross_file_frame_boundary_must_be_consecutive(self):
        self.fx.files["b.parquet"][30]["frame_index"] += 100
        self.fx.rewrite_file("b.parquet")
        with self.assertRaisesRegex(ValueError, "continuity changed"):
            self.compute()

    def test_cross_file_task_boundary_must_match(self):
        # Change an entire covered component so the only mismatch is at file boundary.
        for row in self.fx.files["b.parquet"][30:34]:
            row["task_index"] = 1
        self.fx.rewrite_file("b.parquet")
        with self.assertRaisesRegex(ValueError, "cross-segment"):
            self.compute()

    def test_source_identity_mutation_rejected(self):
        with (self.fx.source / "a.parquet").open("ab") as stream:
            stream.write(b"extra")
        with self.assertRaisesRegex(ValueError, "Source identity changed"):
            self.compute()

    def test_view_bytes_mutation_rejected(self):
        ranges = np.load(self.fx.index / "ranges.npy")
        ranges[0, 1] += 1
        np.save(self.fx.index / "ranges.npy", ranges)
        with self.assertRaisesRegex(ValueError, "artifact bytes"):
            self.compute()

    def test_formal_fit_needs_explicit_approved_contract_before_scan(self):
        with patch.object(tool, "scan_file") as scanned:
            with self.assertRaisesRegex(ValueError, "explicit experiment contract"):
                self.compute(purpose="formal")
            self.assertFalse(scanned.called)

    def test_cannot_overwrite_source_or_view(self):
        with self.assertRaisesRegex(ValueError, "must not be inside"):
            tool.compute(self.fx.index, self.fx.source / "statistics.json")
        with self.assertRaisesRegex(ValueError, "must not be inside"):
            tool.compute(self.fx.index, self.fx.index / "statistics.json")


if __name__ == "__main__":
    unittest.main()
