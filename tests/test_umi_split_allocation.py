"""Property checks for source-group splitting; no models or datasets needed."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/umi_pretrain/tools/umi_split_allocation.py"
)
SPEC = importlib.util.spec_from_file_location("umi_split_allocation_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
allocate_groups = MODULE.allocate_groups


def example_groups(count=300):
    result = []
    for index in range(count):
        weight = 40 + (index * 11) % 71
        # A source group can contain two task labels and remain indivisible.
        result.append({
            "id": f"capture-{index:05d}",
            "windows": weight,
            "tasks": {index % 9: weight // 3, 100 + index % 5: weight - weight // 3},
            "sources": {f"source-{index % 3}": weight},
        })
    return result


class WholeGroupAllocationTest(unittest.TestCase):
    def test_seeded_determinism_is_independent_of_input_order(self):
        groups = example_groups()
        expected, report = allocate_groups(groups, seed=137)
        shuffled = list(groups)
        random.Random(777).shuffle(shuffled)
        actual, second_report = allocate_groups(shuffled, seed=137)
        self.assertEqual(actual, expected)
        self.assertEqual(second_report, report)
        other_seed, _ = allocate_groups(groups, seed=138)
        self.assertNotEqual(other_seed, expected)
        json.dumps(report, allow_nan=False)

    def test_whole_groups_exact_coverage_and_window_conservation(self):
        groups = example_groups()
        assigned, report = allocate_groups(groups)
        allowed = {"validation", *(f"stage_{i:02d}" for i in range(1, 6))}
        self.assertEqual(set(assigned), {group["id"] for group in groups})
        self.assertTrue(set(assigned.values()).issubset(allowed))
        # Each mixed-task capture has one label, not one decision per task.
        by_split = Counter()
        for group in groups:
            self.assertIsInstance(assigned[group["id"]], str)
            by_split[assigned[group["id"]]] += group["windows"]
        self.assertEqual(sum(by_split.values()), sum(group["windows"] for group in groups))
        for name, count in by_split.items():
            self.assertEqual(report["distributions"]["splits"][name]["windows"], count)
        for pass_name in ("validation_allocation", "training_stage_allocation"):
            allocation = report[pass_name]
            for split in allocation["splits"].values():
                deviation = split["deviation_from_target_windows"]
                self.assertLessEqual(deviation, allocation["maximum_over_target_bound_windows"] + 1e-6)
                self.assertGreaterEqual(deviation, -allocation["maximum_under_target_bound_windows"] - 1e-6)

    def test_huge_indivisible_group_and_singleton_task_are_not_duplicated(self):
        groups = [{"id": "huge", "windows": 10000, "tasks": {"rare": 10000}, "sources": {"S": 10000}}]
        groups += [
            {"id": f"small-{i}", "windows": 10, "tasks": {"common": 10}, "sources": {"S": 10}}
            for i in range(100)
        ]
        assigned, report = allocate_groups(groups, seed=7)
        self.assertEqual(len(assigned), 101)
        self.assertTrue(assigned["huge"].startswith("stage_"))
        rare_containing_splits = {assigned[group["id"]] for group in groups if "rare" in group["tasks"]}
        self.assertEqual(len(rare_containing_splits), 1)
        self.assertEqual(sum(split["windows"] for split in report["distributions"]["splits"].values()), 11000)
        json.dumps(report, allow_nan=False)

    def test_frequent_task_can_be_spread_without_forcing_singleton_coverage(self):
        groups = [
            {"id": f"source-{i:04d}", "windows": 100, "tasks": {7: 100}, "sources": {"S": 100}}
            for i in range(200)
        ]
        groups.append({"id": "singleton", "windows": 100, "tasks": {999: 100}, "sources": {"S": 100}})
        assigned, report = allocate_groups(groups)
        common_stages = {assigned[group["id"]] for group in groups[:-1] if assigned[group["id"]] != "validation"}
        self.assertEqual(common_stages, {f"stage_{i:02d}" for i in range(1, 6)})
        singleton_split = assigned["singleton"]
        self.assertEqual(report["distributions"]["splits"][singleton_split]["task_id_count"], 2)
        self.assertEqual(sum(info["task_id_count"] for info in report["distributions"]["splits"].values()), 7)

    def test_zero_windows_and_overlapping_label_weights_are_explicit(self):
        groups = example_groups(80)
        groups.append({"id": "empty", "windows": 0, "tasks": {}, "sources": {}})
        groups[0]["tasks"] = {1: groups[0]["windows"], 2: groups[0]["windows"]}
        assigned, report = allocate_groups(groups)
        self.assertTrue(assigned["empty"].startswith("stage_"))
        self.assertEqual(report["zero_window_group_count"], 1)
        self.assertEqual(report["distributions"]["global_task_label_weight"], sum(sum(g["tasks"].values()) for g in groups))

    def test_invalid_inputs_fail_instead_of_silently_allocating(self):
        good = {"id": "one", "windows": 10, "tasks": {0: 10}, "sources": {"S": 10}}
        invalid_groups = [
            [],
            [good, good],
            [{**good, "windows": -1}],
            [{**good, "windows": True}],
            [{**good, "windows": 1.5}],
            [{**good, "tasks": {0: -1}}],
            [{**good, "sources": {"S": float("nan")}}],
            [{**good, "id": ""}],
            [{**good, "windows": 0}],
            [{"id": "empty", "windows": 0}],
        ]
        for groups in invalid_groups:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                allocate_groups(groups)
        invalid_options = [
            {"seed": True}, {"stages": 0}, {"stages": 2.5},
            {"validation_fraction": 0}, {"validation_fraction": 1},
            {"validation_fraction": float("nan")},
            {"task_balance": -1}, {"source_balance": float("inf")},
        ]
        for kwargs in invalid_options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                allocate_groups([good], **kwargs)


if __name__ == "__main__":
    unittest.main()
