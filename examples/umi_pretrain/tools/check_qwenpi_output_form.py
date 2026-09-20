"""Reproduce the bounded noise-plus-bias diagnostic for the fixed-window run.

Reads trusted local checkpoint tensors with mmap; no model loading or updates.
Assumes the tested eval-mode visual encoder consumes no random numbers before
the action head's first randn. Uses the seeds from check_qwenpi_learnability.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, default=200)
    args = parser.parse_args()
    pred = np.load(args.run_dir/f'predictions_{args.step:06d}.npz')['normalized_prediction']
    state = torch.load(args.run_dir/f'train/checkpoints/update_{args.step:08d}/pytorch_model.bin',
                       map_location='cpu', weights_only=True, mmap=True)
    key = 'action_model.action_decoder.layer2.bias'
    bias = state[key].numpy()
    noises = []
    for i in range(len(pred)):
        torch.manual_seed(19876+i)
        noises.append(torch.randn((1, *pred.shape[1:]), device='cuda', dtype=torch.float32)[0].cpu().numpy())
    noise = np.stack(noises)
    result = dict(bias_key=key, step=args.step, windows=len(pred),
        max_abs_difference_from_noise_plus_final_bias=float(np.max(np.abs(pred-(noise+bias)))),
        rms_difference_from_noise_plus_final_bias=float(np.sqrt(np.mean((pred.astype(np.float64)-noise-bias)**2))))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
