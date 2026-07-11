#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ps/lzz/manifold-detection-energy-transport
RUN_DIR="$ROOT/runs/nwpu_dense_local_delta_learner_s42_48_16"
PYTHON=/home/ps/anaconda3/envs/RLimage/bin/python
ESTIMATED_PEAK_MIB=6144
REQUIRED_FREE_AFTER_MIB=8192
cd "$ROOT"
if [[ -f "$RUN_DIR/eval_metrics.json" ]]; then echo "dense local learner already completed"; exit 0; fi
if pgrep -af "scripts/train_dense_local_delta_learner.py" >/dev/null; then echo "dense local learner already running"; exit 0; fi
GPU2_FREE_MIB=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$1 ~ /^[[:space:]]*2[[:space:]]*$/ {gsub(/[[:space:]]/, "", $2); print $2}')
if [[ -z "$GPU2_FREE_MIB" ]] || (( GPU2_FREE_MIB - ESTIMATED_PEAK_MIB <= REQUIRED_FREE_AFTER_MIB )); then echo "GPU2 reserve gate failed" >&2; exit 3; fi
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "launching dense local learner: gpu2_free_mib=$GPU2_FREE_MIB estimated_peak_mib=$ESTIMATED_PEAK_MIB"
exec "$PYTHON" scripts/train_dense_local_delta_learner.py --run-dir "$RUN_DIR" --device cuda --require-clean-git
