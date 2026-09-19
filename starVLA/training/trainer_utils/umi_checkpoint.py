"""Full local/shared-filesystem checkpoints with publish-after-completion semantics.

Only load trusted checkpoints produced by this training entry. Optimizer and RNG
files use PyTorch serialization. A weights export is deliberately not resumable.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .umi_training_state import fingerprint


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    # POSIX directory durability in addition to file fsync; shared filesystems
    # still require their documented server-side durability guarantees.
    sync_directory(path.parent)


def trim_uncommitted_log(path, committed_step):
    """Stream/truncate only the lost-update suffix, including a torn final line.

    Logs are auxiliary: the complete checkpoint owns progress. Do not load an
    entire long-run JSONL into memory just to recover its last committed offset.
    Interior corruption is an error; a partial final write is safely removed.
    """
    path = Path(path)
    if not path.exists():
        return {"truncated_bytes": 0, "torn_final_line": False}
    size = path.stat().st_size
    keep, previous, torn, add_newline = 0, -1, False, False
    with path.open("r+b") as stream:
        while line := stream.readline():
            try:
                step = json.loads(line)["global_update"]
            except (ValueError, KeyError, UnicodeDecodeError):
                if stream.tell() != size:
                    raise ValueError(f"Interior log corruption: {path}")
                torn = True
                break
            if type(step) is not int or step <= previous:
                raise ValueError(f"Non-monotonic committed log: {path}")
            if step > committed_step:
                break
            previous = step
            keep = stream.tell()
            add_newline = not line.endswith(b"\n")
        if keep != size:
            stream.truncate(keep)
            stream.flush()
            os.fsync(stream.fileno())
        elif keep and add_newline:
            # A complete JSON value with only its newline lost must not merge
            # with the next append into two values on one line.
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return {"truncated_bytes": size - keep, "torn_final_line": torn,
            "restored_final_newline": keep == size and add_newline}


def main_call(accelerator, function):
    """Propagate rank-zero validation/publication errors to every rank."""
    message = [None]
    if accelerator.is_main_process:
        try:
            message[0] = {"value": function()}
        except Exception as error:
            message[0] = {"error": f"{type(error).__name__}: {error}"}
    if dist.is_initialized():
        dist.broadcast_object_list(message, src=0)
    if "error" in message[0]:
        raise RuntimeError(message[0]["error"])
    return message[0]["value"]


def require_all(accelerator, condition, message):
    flag = torch.tensor(int(not condition), device=accelerator.device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise RuntimeError(message)


def capture_rng(generator=None):
    value = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch_cpu=torch.get_rng_state())
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        value["loader_generator"] = generator.get_state()
    return value


def restore_rng(value, generator=None):
    # Unlike Accelerate 1.12's best-effort RNG loading, errors here are fatal.
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])
    if generator is not None:
        generator.set_state(value["loader_generator"])


@contextmanager
def preserve_rng(generator=None):
    state = capture_rng(generator)
    try:
        yield
    finally:
        restore_rng(state, generator)


def inspect_checkpoint(path, identity):
    path = Path(path)
    marker_path = path / "COMPLETED.json"
    if not path.is_dir() or not marker_path.is_file():
        raise ValueError(f"Not a complete training checkpoint: {path}")
    marker = json.loads(marker_path.read_text())
    manifest_path = path / "manifest.json"
    if marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Checkpoint manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "umi-full-checkpoint-v1":
        raise ValueError("Unsupported checkpoint format")
    if manifest["identity_fingerprint"] != fingerprint(identity) or manifest["identity"] != identity:
        raise ValueError("Resume identity mismatch: plan/data/normalization/code/runtime must match")
    files = manifest["files"]
    required = {"pytorch_model.bin", "optimizer.bin", "custom_checkpoint_0.pkl",
                "custom_checkpoint_1.pkl", "training_state.json"}
    for rank in range(identity["runtime"]["world_size"]):
        required.update({f"random_states_{rank}.pkl", f"strict_rng_{rank}.pt"})
    if not required <= files.keys():
        raise ValueError(f"Missing required checkpoint files: {sorted(required - files.keys())}")
    for name, record in files.items():
        file = path / name
        if Path(name).name != name or not file.is_file():
            raise ValueError(f"Invalid checkpoint file: {name}")
        if file.stat().st_size != record["bytes"] or sha256_file(file) != record["sha256"]:
            raise ValueError(f"Checkpoint file corrupt or truncated: {name}")
    state = json.loads((path / "training_state.json").read_text())
    if state["global_update_step"] != manifest["step"]:
        raise ValueError("Checkpoint step mismatch")
    return state


class UMICheckpoints:
    def __init__(self, accelerator, run_dir, identity, progress):
        self.accelerator, self.run_dir = accelerator, Path(run_dir)
        self.identity, self.progress = identity, progress
        self.latest = None

    def resolve(self, requested):
        def resolve_main():
            path = Path(requested)
            if requested == "latest":
                pointer = json.loads((self.run_dir / "latest.json").read_text())
                name = pointer["checkpoint"]
                if Path(name).name != name:
                    raise ValueError("Invalid latest checkpoint name")
                path = self.run_dir / "checkpoints" / name
            state = inspect_checkpoint(path, self.identity)
            return dict(path=str(path.resolve()), state=state)
        resolved = main_call(self.accelerator, resolve_main)
        self.latest = resolved["path"]
        return resolved

    def save(self, generator=None):
        accelerator = self.accelerator
        step = self.progress.global_update_step
        name = f"update_{step:08d}"
        destination = self.run_dir / "checkpoints" / name

        def prepare():
            if destination.exists():
                saved = inspect_checkpoint(destination, self.identity)
                if saved != self.progress.state_dict():
                    raise ValueError("Refusing to overwrite a different checkpoint")
                return {"existing": True}
            # Partial directories remain unreferenced; retries never reuse them.
            import tempfile
            root = destination.parent
            root.mkdir(parents=True, exist_ok=True)
            return {"temporary": tempfile.mkdtemp(prefix=f".{name}-partial-", dir=root)}

        pending = main_call(accelerator, prepare)
        if not pending.get("existing"):
            temporary = Path(pending["temporary"])
            accelerator.wait_for_everyone()
            # Every rank must enter distributed saving. Only publication is rank 0.
            with preserve_rng(generator):
                accelerator.save_state(str(temporary), safe_serialization=False)
                torch.save(capture_rng(generator), temporary / f"strict_rng_{accelerator.process_index}.pt")
            accelerator.wait_for_everyone()

            def publish():
                write_json(temporary / "training_state.json", self.progress.state_dict())
                files = {}
                for file in sorted(temporary.iterdir()):
                    if not file.is_file():
                        raise ValueError("Unexpected sharded checkpoint layout for this backend")
                    with file.open("rb") as stream:
                        os.fsync(stream.fileno())
                    files[file.name] = {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
                manifest = dict(version="umi-full-checkpoint-v1", step=step,
                                identity=self.identity, identity_fingerprint=fingerprint(self.identity), files=files)
                write_json(temporary / "manifest.json", manifest)
                # Completion is published only after every required rank file exists.
                required = {"pytorch_model.bin", "optimizer.bin", "custom_checkpoint_0.pkl",
                            "custom_checkpoint_1.pkl", "training_state.json"}
                for rank in range(accelerator.num_processes):
                    required.update({f"random_states_{rank}.pkl", f"strict_rng_{rank}.pt"})
                if not required <= files.keys():
                    raise ValueError("Accelerate did not write all required training/rank states")
                write_json(temporary / "COMPLETED.json", {"manifest_sha256": sha256_file(temporary / "manifest.json")})
                os.rename(temporary, destination)
                sync_directory(destination.parent)
                write_json(self.run_dir / "latest.json", {"checkpoint": name, "step": step})
            main_call(accelerator, publish)
        else:
            main_call(accelerator, lambda: write_json(self.run_dir / "latest.json", {"checkpoint": name, "step": step}))
        accelerator.wait_for_everyone()
        self.latest = str(destination)
        return self.latest

    def load(self, resolved, generator=None):
        self.accelerator.load_state(resolved["path"])
        if self.progress.state_dict() != resolved["state"]:
            raise ValueError("Framework training state differs from checkpoint summary")
        state = torch.load(Path(resolved["path"]) / f"strict_rng_{self.accelerator.process_index}.pt",
                           map_location="cpu", weights_only=False)
        restore_rng(state, generator)
        if self.accelerator.step % self.accelerator.gradient_accumulation_steps:
            raise ValueError("Checkpoint lies inside gradient accumulation")
        self.accelerator.wait_for_everyone()
