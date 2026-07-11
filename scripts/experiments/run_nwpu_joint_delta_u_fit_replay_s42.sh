#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ps/lzz/manifold-detection-energy-transport"
PYTHON="/home/ps/anaconda3/envs/RLimage/bin/python"
RUN_DIR="$REPO/runs/nwpu_joint_delta_u_probe_v2_s42_traincal_cleanval180"

cd "$REPO"

FREE_MB="$(
  nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits |
    tr -d '[:space:]'
)"
if [[ ! "$FREE_MB" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU2 free memory: $FREE_MB" >&2
  exit 2
fi
if (( FREE_MB <= 8192 )); then
  echo "GPU2 has ${FREE_MB} MiB free; strictly more than 8192 MiB is required." >&2
  exit 75
fi

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" scripts/analyze_nwpu_joint_delta_u_fit_replay.py \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --require-clean-git
