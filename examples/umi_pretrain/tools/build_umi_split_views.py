"""Compile/verify validation and stage access views from a frozen split manifest.

Existing completed views are verified before reuse. This compiles metadata only;
it neither copies public payload nor launches training. Failed builds keep logs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from build_umi_access_index import (
    PRIVATE, compute_view_fingerprint, fail, now, progress, read_json, sha256, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=PRIVATE / "roban_umi_splits_metadata_candidate_v1")
    parser.add_argument("--output", type=Path, default=PRIVATE / "roban_umi_candidate_views_v1")
    parser.add_argument("--allow-candidate-splits", action="store_true",
                        help="Compile metadata-only candidate views for interface checks; does not approve a final training split")
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2), help="Concurrent metadata compilation jobs, not training workers")
    args = parser.parse_args()
    splits, output = args.splits.resolve(), args.output.resolve()
    report, config = read_json(splits / "split_report.json"), read_json(splits / "split_config.json")
    if report.get("status") != "completed" or report["split_fingerprint"] != config["split_fingerprint"]:
        fail("Split manifests are not completed or fingerprints differ")
    if not report.get("semantic_balance_assessed", False) and not args.allow_candidate_splits:
        fail("Task/scene semantic balance is unassessed. These are candidate splits; use --allow-candidate-splits only for interface checks")
    if not report.get("checks") or not all(report["checks"].values()):
        fail("Split invariant checks have not passed")
    if output == PRIVATE.resolve() or not output.is_relative_to(PRIVATE.resolve()):
        fail("Views must be in a private data_preparation subdirectory")
    for protected in (splits, Path(config["catalog"]).resolve(), Path(config["window_index"]).resolve()):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            fail("View output overlaps input directories")
    names = ["validation"] + [f"stage_{i:02d}" for i in range(1, config["contract"]["stages"] + 1)]
    expected_hashes = {}
    for name in names:
        filename = name + "_episodes.parquet"
        expected_hashes[name] = report["artifacts"][filename]["sha256"]
        if sha256(splits / filename) != expected_hashes[name]:
            fail("Frozen episode list hash mismatch", split=name)
    builder = Path(__file__).with_name("build_umi_access_index.py")
    identity = {"split_fingerprint": config["split_fingerprint"],
                "split_config_sha256": sha256(splits / "split_config.json"),
                "split_report_sha256": sha256(splits / "split_report.json"),
                "episode_list_sha256": expected_hashes, "access_builder_sha256": sha256(builder)}
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "views.build.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        cfg_path = output / "view_build_config.json"
        if cfg_path.exists():
            if read_json(cfg_path)["identity"] != identity:
                fail("Existing views directory belongs to a different build; choose a new version")
        else:
            write_json(cfg_path, {"created_at": now(), "splits": str(splits), "identity": identity})
        (output / "logs").mkdir(exist_ok=True)

        def compile_one(name):
            target, selection = output / name, splits / (name + "_episodes.parquet")
            if not target.exists():
                progress("compile_view", name=name)
                with (output / "logs" / (name + ".log")).open("w") as log:
                    result = subprocess.run([
                        sys.executable, "-u", str(builder), "--catalog", config["catalog"],
                        "--window-index", config["window_index"], "--episode-list", str(selection),
                        "--output", str(target),
                    ], stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    fail("Access compilation failed; see per-view log", split=name, returncode=result.returncode)
            meta = read_json(target / "meta.json")
            expected = report["counts"][name]
            if meta["status"] != "completed" or meta["episode_list_sha256"] != expected_hashes[name]:
                fail("Existing view does not match frozen split", split=name)
            if meta["rule_fingerprint"] != report["rule_fingerprint"]:
                fail("View window rule differs from split", split=name)
            if meta["total_windows"] != expected["valid_windows"] or meta["total_episodes"] != expected["episodes"]:
                fail("View counts differ from manifest", split=name)
            if pq.ParquetFile(selection).metadata.num_rows != expected["episodes"]:
                fail("Manifest row count differs from split report", split=name)
            for filename, artifact in meta["artifacts"].items():
                if filename not in ("ranges.npy", "cumulative.npy", "metadata.sqlite3"):
                    fail("Unexpected view artifact", split=name, filename=filename)
                if sha256(target / filename) != artifact["sha256"]:
                    fail("Compiled view artifact hash mismatch", split=name, artifact=filename)
            fingerprint = compute_view_fingerprint(meta)
            if meta.get("view_fingerprint") != fingerprint:
                fail("Compiled view identity mismatch", split=name)
            progress("verified_view", name=name, windows=meta["total_windows"], episodes=meta["total_episodes"])
            return {"name": name, "path": str(target), "view_fingerprint": fingerprint,
                    "meta_sha256": sha256(target / "meta.json"),
                    "episode_list_sha256": meta["episode_list_sha256"],
                    "episodes": meta["total_episodes"], "valid_windows": meta["total_windows"]}

        completed = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(compile_one, name) for name in names]
            for future in as_completed(futures):
                completed.append(future.result())
        if sum(item["valid_windows"] for item in completed) != report["valid_windows"]:
            fail("Six views do not conserve eligible window total")
        if len({item["view_fingerprint"] for item in completed}) != len(names):
            fail("Distinct splits unexpectedly have identical view identities")
        if sha256(splits / "split_config.json") != identity["split_config_sha256"] or sha256(splits / "split_report.json") != identity["split_report_sha256"]:
            fail("Split inputs changed during compilation")
        write_json(output / "view_build_report.json", {
            "status": "completed", "created_at": now(), "identity": identity,
            "selection_status": report.get("selection_status", "candidate_only"),
            "semantic_balance_assessed": report.get("semantic_balance_assessed", False),
            "views": sorted(completed, key=lambda item: item["name"]),
            "checks": {"all_view_counts_match_manifests": True, "all_view_artifact_hashes_match": True,
                       "eligible_window_total_conserved": True, "view_fingerprints_distinct": True},
        })
        progress("completed", output=str(output), views=len(completed))
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
