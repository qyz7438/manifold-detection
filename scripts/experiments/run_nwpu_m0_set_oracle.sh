#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
PYTHON="/home/ps/anaconda3/envs/RLimage/bin/python"
RUN_NAME="nwpu_m0_set_oracle_s42_fullval"
RUN_DIR="${ROOT}/runs/${RUN_NAME}"
RESULT="${RUN_DIR}/eval_metrics.json"
CONFIG="spectral_detection_posttrain/configs/versions/det.energy.set_oracle.m0.001.json"
CHECKPOINT="runs/nwpu_mob_strong_cosine_s42_bs8_36ep/checkpoint_best.pth"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

result_complete() {
  "${PYTHON}" - "${RESULT}" <<'PY'
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
    free_mb="$(
      nvidia-smi -i 2 \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits \
        | head -n 1 \
        | tr -d '[:space:]'
    )"
    if [[ ! "${free_mb}" =~ ^[0-9]+$ ]]; then
      log "GPU2 free memory is unavailable; retrying"
      sleep 60
      continue
    fi
    if (( free_mb <= 8192 )); then
      log "GPU2 free=${free_mb}MB; waiting for strictly more than 8192MB"
      sleep 60
      continue
    fi
    log "GPU2 free=${free_mb}MB; starting locked M0 full-val"
    return
  done
}

if result_complete; then
  log "${RUN_NAME} already complete"
  exit 0
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  log "missing strong checkpoint: ${CHECKPOINT}"
  exit 3
fi

mkdir -p "${RUN_DIR}"
wait_for_gpu2
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" scripts/eval_nwpu_m0_set_oracle.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --annotation "data/NWPU_VHR10_coco.json" \
  --data-root "data/NWPU VHR-10 dataset" \
  --output "${RESULT}" \
  --device cuda \
  --require-full-validation \
  --require-clean-git

result_complete
log "${RUN_NAME} complete"
