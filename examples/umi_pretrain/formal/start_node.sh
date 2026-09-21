#!/usr/bin/env bash
# Execute ONCE on each of four nodes, not once per PPU.
set -euo pipefail
: "${MASTER_ADDR:?Set the actual master address supplied by the cloud job}"
: "${NODE_RANK:?Set this node rank, 0..3}"
: "${PLAN:?Set an absolute, finalized formal plan path}"
: "${RUN_DIR:?Set an absolute private/shared persistent output path}"
: "${PYTHON_BIN:?Set the existing PPU environment python executable on this node}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ ! "$NODE_RANK" =~ ^[0-3]$ ]]; then
  printf '%s\n' 'NODE_RANK must be 0, 1, 2 or 3' >&2; exit 2
fi
if [[ "$PLAN" != /* || "$RUN_DIR" != /* || "$PYTHON_BIN" != /* ]]; then
  printf '%s\n' 'PLAN, RUN_DIR and PYTHON_BIN must be absolute paths' >&2; exit 2
fi
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"
if [[ -n "${PPU_ENV_SCRIPT:-}" ]]; then
  # Vendor environment applies only to this job process; no package changes.
  source "$PPU_ENV_SCRIPT"
fi
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
"$PYTHON_BIN" - "$PLAN" <<'PY'
import json,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from starVLA.training.trainer_utils.umi_training_state import validate_plan
p=OmegaConf.to_container(OmegaConf.load(sys.argv[1]),resolve=True)
validate_plan(p)
if (p['purpose']!='formal' or p['formal']['budget_status']!='fixed'
        or tuple(p['deployment'][key] for key in ('nodes','processes_per_node','world_size')) != (4,16,64)):
    raise ValueError('Require a finalized formal 64-process plan')
s=Path(p['data']['normalization_statistics'])
if not s.is_file():
    raise ValueError('Formal statistics are still pending: '+str(s))
stats=json.loads(s.read_text())
if stats.get('purpose')!='formal':
    raise ValueError('Engineering/pilot statistics cannot initialize the formal run')
visible = torch.cuda.device_count()
required = p['deployment']['processes_per_node']
if visible < required:
    raise RuntimeError(f'This node exposes {visible} devices but needs {required}; '
                       'check the cloud allocation and CUDA_VISIBLE_DEVICES. '
                       'Do not reuse the single-device debug setting; visibility is not changed automatically.')
print('Visible devices:',visible,'training processes on this node:',required,flush=True)
print('Plan budget:',p['formal']['total_updates'],'global batch:',p['training']['global_batch_size'],flush=True)
PY
OPTIONS=(--plan "$PLAN" --output-dir "$RUN_DIR")
if [[ -n "${RESUME:-}" ]]; then OPTIONS+=(--resume "$RESUME"); fi
if [[ -n "${PAUSE_AFTER_UPDATE:-}" ]]; then OPTIONS+=(--stop-after-update "$PAUSE_AFTER_UPDATE"); fi
exec "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes=4 --nproc-per-node=16 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0 \
  --module starVLA.training.train_umi_pretrain "${OPTIONS[@]}"
