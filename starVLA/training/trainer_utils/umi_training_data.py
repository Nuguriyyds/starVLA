"""Stage loader lifecycle and an explicit synthetic engineering backend."""
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.umi_sampler import UMIBlockShuffleSampler
from .umi_checkpoint import preserve_rng, sha256_file, write_json
from .umi_training_state import fingerprint


def list_collate(values):
    return values


class TinyDataset(Dataset):
    """Deterministic per-sample data; never selected implicitly for real training."""
    def __init__(self, stage):
        self.size, self.offset = int(stage["size"]), int(stage.get("offset", 0))
        self.view_fingerprint = fingerprint({"kind": "synthetic-engineering-v1",
                                             "size": self.size, "offset": self.offset})

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = np.sin(np.arange(16, dtype=np.float32) + (index + self.offset) / 10)[None, :]
        return {"state": x, "action": np.repeat((x * .3 + .1), 16, axis=0),
                "umi_metadata": {"dataset_index": index, "episode_index": self.offset,
                                 "frame_index": index, "view_fingerprint": self.view_fingerprint}}

    def provenance(self):
        return {"purpose": "engineering", "view_fingerprint": self.view_fingerprint,
                "num_windows": len(self), "normalization": "none", "synthetic": True}

    def close(self):
        pass


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(16, 24), torch.nn.Dropout(.15),
                                            torch.nn.GELU(), torch.nn.Linear(24, 16))

    def forward(self, examples):
        device = next(self.parameters()).device
        x = torch.from_numpy(np.stack([s["state"][0] for s in examples])).to(device)
        y = torch.from_numpy(np.stack([s["action"][0] for s in examples])).to(device)
        # Exercise every model-side RNG family in resume/evaluation tests.
        noise = torch.randn_like(x) * .03 + float(np.random.normal(0, .01)) + random.random() * .01
        return {"action_loss": (self.network(x + noise) - y).square().mean()}

    def predict_action(self, examples):
        device = next(self.parameters()).device
        x = torch.from_numpy(np.stack([s["state"][0] for s in examples])).to(device)
        value = self.network(x) + torch.randn_like(x) * .01
        return {"normalized_actions": value[:, None, :].expand(-1, 16, -1).detach().float().cpu().numpy()}


def make_loader(plan, stage, run_dir, *, evaluation=False, record=True):
    train = plan["training"]
    seed = int(train.get("seed", 42))
    output = Path(run_dir) / "data_access" / (("eval_" if evaluation else "stage_") + stage["name"])
    if plan["model_kind"] == "tiny":
        dataset = TinyDataset(stage)
        workers = train.get("num_workers", 0)
        options = {}
        if workers:
            options.update(multiprocessing_context="spawn", persistent_workers=True,
                           prefetch_factor=train.get("prefetch_factor", 2))
        return DataLoader(dataset, batch_size=train["batch_size"],
                          sampler=UMIBlockShuffleSampler(len(dataset), block_size=7, seed=seed,
                                                          shuffle=not evaluation),
                          collate_fn=list_collate, num_workers=workers, drop_last=True,
                          generator=torch.Generator().manual_seed(seed), **options)
    from omegaconf import OmegaConf
    from starVLA.dataloader import build_dataloader
    data = dict(plan["data"], index_dir=stage["index_dir"], dataset_py="umi_indexed",
                normalization_purpose=plan["purpose"], per_device_batch_size=train["batch_size"],
                num_workers=train.get("num_workers", 0), prefetch_factor=train.get("prefetch_factor", 2),
                drop_last=True, return_metadata=True, seed=seed, write_access_record=record,
                shuffle=False if evaluation else plan["data"].get("shuffle", True))
    cfg = OmegaConf.create({"framework": plan["framework"], "datasets": {"vla_data": data},
                            "output_dir": str(output)})
    return build_dataloader(cfg, dataset_py="umi_indexed")


def inspect_views(plan, run_dir):
    """Verify compact index bytes once on rank 0, never scan source frames/videos."""
    records, checked = [], set()
    views = list(plan["stages"]) + ([plan["evaluation"]] if plan.get("evaluation") else [])
    for i, stage in enumerate(views):
        loader = make_loader(plan, stage, run_dir, evaluation=i == len(plan["stages"]), record=False)
        try:
            dataset = loader.dataset
            record = dataset.provenance()
            if plan["model_kind"] != "tiny":
                raw = getattr(dataset, "raw_dataset", dataset)
                for name, artifact in raw.meta["artifacts"].items():
                    file = raw.index_dir / name
                    if file not in checked:
                        if sha256_file(file) != artifact["sha256"]:
                            raise ValueError(f"Access artifact changed: {file}")
                        checked.add(file)
                if plan["purpose"] == "formal" and not hasattr(dataset, "normalizer"):
                    raise ValueError("Formal training requires the approved normalization experiment contract")
                from starVLA.dataloader.umi_normalization import build_representation, parent_fingerprint
                record["parent_fingerprint"] = parent_fingerprint(raw.meta)
                record["representation"] = build_representation(raw.meta)
            records.append(record)
        finally:
            loader.dataset.close()
    if plan["model_kind"] != "tiny":
        if len({r["parent_fingerprint"] for r in records}) != 1:
            raise ValueError("All stages/evaluation must use the same parent data and rules")
        if len({fingerprint(r["representation"]) for r in records}) != 1:
            raise ValueError("Stage representations differ")
    return records


class StageLoader:
    """The model is wrapped once; only the active stage owns workers/caches."""
    def __init__(self, accelerator, loader, stage_index, seed):
        self.accelerator, self.raw = accelerator, loader
        self.sampler = loader.sampler
        self.generator = loader.generator
        self.stage_index, self.seed = stage_index, seed
        self.prepared = accelerator.prepare_data_loader(loader, device_placement=False)
        self.iterator = None
        self.first = None
        self.epoch = None

    def start(self, epoch, cursor):
        self.stop_iterator()
        self.sampler.set_epoch(epoch, start_index=cursor)
        self.prepared.set_epoch(epoch)
        # Worker base seeds are derived from stage/epoch. Current transforms are
        # deterministic. Iterator recreation must not consume model-side RNG or
        # advance an explicit generator relative to an uninterrupted process.
        with preserve_rng(self.generator):
            self.generator.manual_seed(self.seed + 1000003 * self.stage_index + epoch)
            self.iterator = iter(self.prepared)
            self.first = next(self.iterator)
        self.epoch = epoch

    def next(self):
        if self.first is not None:
            batch, self.first = self.first, None
            return batch
        return next(self.iterator)

    def stop_iterator(self):
        self.first = None
        if self.iterator is not None and hasattr(self.iterator, "close"):
            self.iterator.close()
        self.iterator = None
        # DataLoaderShard may have been abandoned before its final lookahead.
        self.prepared.end()
        base = self.prepared.base_dataloader
        worker_iterator = getattr(base, "_iterator", None)
        if worker_iterator is not None:
            worker_iterator._shutdown_workers()
            base._iterator = None

    def close(self):
        self.stop_iterator()
        self.prepared._is_accelerate_prepared = False
        self.accelerator._dataloaders = [x for x in self.accelerator._dataloaders if x is not self.prepared]
        self.raw.dataset.close()


def evaluate(accelerator, model, plan, run_dir):
    """Separate loader and RNG stream; engineering error, not policy success."""
    was_training = model.training
    loader = None
    with preserve_rng():
        try:
            seed = int(plan["evaluation"].get("seed", 9876))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            loader = make_loader(plan, plan["evaluation"], run_dir, evaluation=True)
            model.eval()
            with torch.no_grad():
                batch = next(iter(loader))
                truth = np.stack([s["action"] for s in batch]).astype(np.float64)
                inputs = [{k: v for k, v in s.items() if k != "action"} for s in batch]
                predicted = accelerator.unwrap_model(model).predict_action(examples=inputs)["normalized_actions"]
            predicted = np.asarray(predicted, dtype=np.float32)
            normalizer = getattr(loader.dataset, "normalizer", None)
            raw_pred = normalizer.inverse_action(predicted) if normalizer else predicted
            raw_true = normalizer.inverse_action(truth.astype(np.float32)) if normalizer else truth
            error = np.asarray(raw_pred, dtype=np.float64) - np.asarray(raw_true, dtype=np.float64)
            result = {"purpose": plan["purpose"], "view": loader.dataset.provenance()["view_fingerprint"],
                      "normalized_action_mse": float(np.mean((predicted - truth) ** 2)),
                      "raw_position_rmse": float(np.sqrt(np.mean(error[..., [0, 1, 2, 8, 9, 10]] ** 2))),
                      "raw_gripper_rmse": float(np.sqrt(np.mean(error[..., [7, 15]] ** 2))),
                      "raw_quaternion_component_rmse": float(np.sqrt(np.mean(error[..., [3, 4, 5, 6, 11, 12, 13, 14]] ** 2)))}
            if not all(np.isfinite(v) for v in result.values() if isinstance(v, float)):
                raise ValueError("Nonfinite evaluation output")
            return result
        finally:
            model.train(was_training)
            if loader is not None:
                worker_iterator = getattr(loader, "_iterator", None)
                if worker_iterator is not None:
                    worker_iterator._shutdown_workers()
                loader.dataset.close()
