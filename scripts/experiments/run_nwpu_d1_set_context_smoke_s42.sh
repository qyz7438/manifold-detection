#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
RUN_DIR="$ROOT/runs/nwpu_d1_set_context_smoke_s42"
cd "$ROOT"
free_mb=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
estimated_peak_mib=7168
if (( free_mb - estimated_peak_mib <= 8192 )); then
  echo "GPU2 memory.free=${free_mb} MiB; estimated peak ${estimated_peak_mib} MiB would violate the 8192 MiB reserve"
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "remote worktree is dirty"
  exit 4
fi
if [[ -f "$RUN_DIR/eval_metrics.json" ]]; then
  echo "D1 artifact already exists"
  exit 0
fi
export CUDA_VISIBLE_DEVICES=2
mkdir -p "$RUN_DIR"
/home/ps/anaconda3/envs/RLimage/bin/python scripts/train_nwpu_d1_set_context.py \
  --run-dir "$RUN_DIR" --require-clean-git 2>&1 | tee "$RUN_DIR/launcher.log"
