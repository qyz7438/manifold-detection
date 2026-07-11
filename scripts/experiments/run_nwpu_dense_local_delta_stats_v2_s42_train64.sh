#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ps/lzz/manifold-detection-energy-transport
RUN_DIR="$ROOT/runs/nwpu_dense_local_delta_stats_v2_s42_train64"
PYTHON=/home/ps/anaconda3/envs/RLimage/bin/python
ESTIMATED_PEAK_MIB=6144
REQUIRED_FREE_AFTER_MIB=8192

cd "$ROOT"
if [[ -f "$RUN_DIR/eval_metrics.json" ]]; then
  echo "dense local Delta-Q v2 stats already completed"
  exit 0
fi
if pgrep -af "scripts/analyze_dense_local_delta_stats.py.*dense_local_delta_stats_v2" >/dev/null; then
  echo "dense local Delta-Q v2 stats is already running"
  exit 0
fi
GPU2_FREE_MIB=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F, '$1 ~ /^[[:space:]]*2[[:space:]]*$/ {gsub(/[[:space:]]/, "", $2); print $2}')
if [[ -z "$GPU2_FREE_MIB" ]]; then
  echo "unable to read physical GPU2 memory.free" >&2
  exit 2
fi
if (( GPU2_FREE_MIB - ESTIMATED_PEAK_MIB <= REQUIRED_FREE_AFTER_MIB )); then
  echo "GPU2 reserve gate failed: free=$GPU2_FREE_MIB estimated_peak=$ESTIMATED_PEAK_MIB required_after>$REQUIRED_FREE_AFTER_MIB" >&2
  exit 3
fi
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "launching dense local Delta-Q v2 stats: gpu2_free_mib=$GPU2_FREE_MIB estimated_peak_mib=$ESTIMATED_PEAK_MIB"
exec "$PYTHON" scripts/analyze_dense_local_delta_stats.py \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --require-clean-git
