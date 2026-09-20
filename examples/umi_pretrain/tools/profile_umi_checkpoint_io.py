"""Bounded, read-only checkpoint storage diagnostic. Does NOT validate payloads.

Compare sequential reads with read+SHA on the same prefixes, alternate order,
and record cache caveats. Never clears shared caches, loads a model or rewrites
an old manifest. Partial-read digests are not full-file integrity references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from starVLA.training.trainer_utils.umi_checkpoint import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes-per-file", type=int, default=64*1024**2,
                        help="0 reads whole files; default limits each pass to 64 MiB")
    args = parser.parse_args()
    root, output = args.checkpoint.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Write diagnostic output outside the checkpoint")
    if args.max_bytes_per_file < 0:
        raise ValueError("max-bytes-per-file must be nonnegative")
    marker = json.loads((root / "COMPLETED.json").read_text())
    if marker["manifest_sha256"] != sha256_file(root / "manifest.json"):
        raise ValueError("Manifest integrity mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    mount = subprocess.run(["findmnt", "-T", str(root), "-J", "-o", "TARGET,FSTYPE,SOURCE"],
                           capture_output=True, text=True)
    report = {"checkpoint": str(root), "manifest_version": manifest["version"],
              "integrity": manifest.get("integrity", "full"), "max_bytes_per_file": args.max_bytes_per_file,
              "mount": mount.stdout.strip(), "mount_error": mount.stderr.strip(), "files": [],
              "caveat": "Read-only prefixes, not a payload validation. Repeated reads warm caches; order alternates but does not establish cold-cache causality. No cache flush."}
    for name, metadata in sorted(manifest["files"].items()):
        if Path(name).name != name:
            raise ValueError("Invalid manifest filename")
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Missing or truncated checkpoint file: {name}")
        limit = min(metadata["bytes"], args.max_bytes_per_file or metadata["bytes"])
        record = {"file": name, "size_bytes": metadata["bytes"], "bytes_per_pass": limit,
                  "full_file_read": limit == metadata["bytes"], "passes": []}
        for method in ("read", "read_sha256", "read_sha256", "read"):
            before = time.perf_counter()
            consumed = 0
            digest = hashlib.sha256() if method == "read_sha256" else None
            with path.open("rb") as stream:
                while consumed < limit:
                    block = stream.read(min(8*1024**2, limit-consumed))
                    if not block:
                        raise ValueError("Short checkpoint read")
                    consumed += len(block)
                    if digest is not None:
                        digest.update(block)
            seconds = time.perf_counter()-before
            record["passes"].append({"method": method, "bytes": consumed, "seconds": seconds,
                                     "MiB_per_second": consumed/1024**2/seconds,
                                     "prefix_sha256": digest.hexdigest() if digest else None})
        report["files"].append(record)
        write_json(output, report)
        print(f"{name}: {metadata['bytes']} bytes; {limit} bytes/pass", flush=True)


if __name__ == "__main__":
    main()
