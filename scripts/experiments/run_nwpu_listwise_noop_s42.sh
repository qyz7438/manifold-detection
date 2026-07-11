#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ps/lzz/manifold-detection-energy-transport
RUN_DIR="$ROOT/runs/nwpu_listwise_noop_s42_reused_outer"
PYTHON=/home/ps/anaconda3/envs/RLimage/bin/python
RESULT="$RUN_DIR/eval_metrics.json"
CONFIG_SHA=3c29eaa334ef88a7ae0e9b04936ac4259c96d021055d46d0daf93c734495718a

CURRENT_HEAD=$(git -C "$ROOT" rev-parse HEAD)
if [ -f "$RESULT" ] && "$PYTHON" -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
same = payload.get("completed") is True and payload.get("inputs", {}).get("code_commit") == sys.argv[2] and payload.get("inputs", {}).get("config_sha256") == sys.argv[3]
raise SystemExit(0 if same else 1)
' "$RESULT" "$CURRENT_HEAD" "$CONFIG_SHA"; then
  printf 'listwise no-op probe already completed for commit %s\n' "$CURRENT_HEAD"
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

exec "$PYTHON" scripts/probe_nwpu_listwise_noop.py \
  --config spectral_detection_posttrain/configs/versions/det.energy.listwise_noop.001.json \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --require-clean-git
