#!/usr/bin/env bash
set -euo pipefail

cd /home/ps/lzz/manifold-detection-energy-transport
RUN_NAME=nwpu_native_c1_contract_s42
RUN_DIR="runs/$RUN_NAME"

GPU_FREE=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [[ "$GPU_FREE" -le 8192 ]]; then
  echo "GPU2 has ${GPU_FREE} MiB free; strictly more than 8192 MiB is required"
  exit 3
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "remote worktree is dirty"
  exit 4
fi

export CUDA_VISIBLE_DEVICES=2
mkdir -p "$RUN_DIR"
/home/ps/anaconda3/envs/RLimage/bin/python scripts/verify_nwpu_native_c1_contract.py \
  --run-name "$RUN_NAME" 2>&1 | tee "$RUN_DIR/launcher.log"
test -s "$RUN_DIR/eval_metrics.json"
