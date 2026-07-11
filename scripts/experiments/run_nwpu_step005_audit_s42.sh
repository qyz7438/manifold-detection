#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ps/lzz/manifold-detection-energy-transport
RUN_DIR="$ROOT/runs/nwpu_step005_strata_s42_adaptive"
PYTHON=/home/ps/anaconda3/envs/RLimage/bin/python
RESULT="$RUN_DIR/eval_metrics.json"
CONFIG_SHA=c3979fe727f8c3a45981b8ca91310e39ae4ad305ccd2064ff64e633e87717e02

CURRENT_HEAD=$(git -C "$ROOT" rev-parse HEAD)
if [ -f "$RESULT" ] && "$PYTHON" -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
same = (
    payload.get("completed") is True
    and payload.get("inputs", {}).get("code_commit") == sys.argv[2]
    and payload.get("inputs", {}).get("config_sha256") == sys.argv[3]
)
raise SystemExit(0 if same else 1)
' "$RESULT" "$CURRENT_HEAD" "$CONFIG_SHA"; then
  printf 'step-0.05 audit already completed for commit %s\n' "$CURRENT_HEAD"
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
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" scripts/analyze_nwpu_step005_strata.py \
  --config spectral_detection_posttrain/configs/versions/det.energy.step005_audit.001.json \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --require-clean-git
