#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
RUN_DIR="$ROOT/runs/nwpu_c2a_action_margin_smoke_s42"
cd "$ROOT"
free_mb=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( free_mb <= 8192 )); then
  echo "GPU2 memory.free=${free_mb} MiB; strictly more than 8192 MiB required"
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "remote worktree is dirty"
  exit 4
fi
if [[ -f "$RUN_DIR/eval_metrics.json" ]]; then
  echo "C2a artifact already exists"
  exit 0
fi
export CUDA_VISIBLE_DEVICES=2
mkdir -p "$RUN_DIR"
/home/ps/anaconda3/envs/RLimage/bin/python scripts/eval_nwpu_c2a_action_margin.py \
  --run-dir "$RUN_DIR" --require-clean-git 2>&1 | tee "$RUN_DIR/launcher.log"
