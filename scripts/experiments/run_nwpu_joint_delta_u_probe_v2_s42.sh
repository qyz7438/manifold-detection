#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ps/lzz/manifold-detection-energy-transport
RUN_DIR="$ROOT/runs/nwpu_joint_delta_u_probe_v2_s42_traincal_cleanval180"
PYTHON=/home/ps/anaconda3/envs/RLimage/bin/python
RESULT="$RUN_DIR/eval_metrics.json"
CONFIG_SHA=0b327d1f60e3fac222d1233a7fa35ed60c81abcf44d987a653f71b71189f6938

test "$(git -C "$ROOT" rev-parse --is-inside-work-tree)" = true
CURRENT_HEAD=$(git -C "$ROOT" rev-parse HEAD)

if [ -f "$RESULT" ] && "$PYTHON" -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
inputs = payload.get("inputs", {})
environment = inputs.get("environment", {})
same = (
    payload.get("completed") is True
    and inputs.get("code_commit") == sys.argv[2]
    and environment.get("config_file_hash") == sys.argv[3]
)
raise SystemExit(0 if same else 1)
' "$RESULT" "$CURRENT_HEAD" "$CONFIG_SHA"; then
  printf 'joint Delta-U v2 already completed for commit %s and locked config\n' "$CURRENT_HEAD"
  exit 0
fi

GPU_FREE=$(nvidia-smi --id=2 --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [ "$GPU_FREE" -le 8192 ]; then
  printf 'GPU2 memory.free=%s MiB; strict launch gate requires >8192 MiB\n' "$GPU_FREE" >&2
  exit 75
fi

mkdir -p "$RUN_DIR"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" scripts/probe_nwpu_joint_delta_u_v2.py \
  --config spectral_detection_posttrain/configs/versions/det.energy.joint_probe.002.json \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --require-clean-git \
  --require-full-validation
