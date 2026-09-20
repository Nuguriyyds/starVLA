"""Minimal reproduction of PPU process-group shutdown failure.

Run with the existing PPU runtime, without loading UMI or QwenPI:
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
        --standalone --nproc_per_node=2 <this-file>

This performs one real Linear/AdamW update. It deliberately keeps standard
distributed cleanup and propagates errors; it is not a workaround for them.
"""
from datetime import timedelta
import faulthandler
import os

import torch
import torch.distributed as dist


def main():
    faulthandler.enable(all_threads=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    model = torch.nn.parallel.DistributedDataParallel(
        torch.nn.Linear(16, 16).cuda(), device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), fused=False, foreach=False)
    model(torch.ones(2, 16, device=f"cuda:{rank}")).square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()
    print(rank, "before destroy", flush=True)
    dist.destroy_process_group()
    print(rank, "after destroy", flush=True)
    value = next(model.parameters()).detach().flatten()[[0, 1, 2]].cpu().tolist()
    print(rank, "post destroy tensor read", value, flush=True)


if __name__ == "__main__":
    main()
