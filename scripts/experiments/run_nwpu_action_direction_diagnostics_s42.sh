#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
PYTHON="/home/ps/anaconda3/envs/RLimage/bin/python"
RUN_NAME="diag_action_direction_strong_s42_fullval"
RESULT="${ROOT}/runs/${RUN_NAME}/diagnostics.json"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

result_complete() {
  "${PYTHON}" - "$RESULT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("completed") is True else 1)
PY
}

wait_for_gpu2() {
  while true; do
    free_mb="$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb > 8192 )); then
      log "GPU2 free=${free_mb}MB; starting diagnostic"
      return
    fi
    log "GPU2 free=${free_mb:-unknown}MB; waiting for >8192MB"
    sleep 60
  done
}

if result_complete; then
  log "${RUN_NAME} already complete"
  exit 0
fi

wait_for_gpu2
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" scripts/analyze_action_transport_directions.py \
  --run-name "${RUN_NAME}" \
  --detector-checkpoint runs/nwpu_mob_strong_cosine_s42_bs8_36ep/checkpoint_best.pth \
  --action-run boxonly=runs/action_native_boxonly_strong_s42_bs8_ep4 \
  --action-run preserve2=runs/action_native_preserve2_strong_s42_bs8_ep4 \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --batch-size 8 \
  --num-workers 0 \
  --seed 42 \
  --data-seed 42 \
  --scales 0,0.05,0.1,0.25,0.5,1.0 \
  --permutation-seed 1701 \
  --oracle-min-iou 0.5 \
  --score-threshold 0.05 \
  --nms-threshold 0.5 \
  --detections-per-img 100 \
  --require-clean-git

result_complete
log "${RUN_NAME} complete"
