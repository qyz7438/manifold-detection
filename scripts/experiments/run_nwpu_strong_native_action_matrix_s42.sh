#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ps/lzz/manifold-detection-energy-transport}"
PYTHON_BIN="${PYTHON_BIN:-/home/ps/anaconda3/envs/RLimage/bin/python}"
GPU_ID="${GPU_ID:-2}"
MIN_FREE_MB="${MIN_FREE_MB:-8192}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-0.0003}"
STRONG_RUN="nwpu_mob_strong_cosine_s42_bs8_36ep"
PARITY_RUN="det_action_zero_parity_nativefix_fullft18best_s42"

if [[ "${GPU_ID}" != "2" ]]; then
  echo "This experiment is restricted to GPU2; got GPU_ID=${GPU_ID}." >&2
  exit 2
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    echo "$(date -Is) waiting for ${path}"
    sleep "${POLL_SECONDS}"
  done
}

require_completed() {
  local path="$1"
  wait_for_file "${path}"
  if ! grep -q '"completed": true' "${path}"; then
    echo "Required run is incomplete: ${path}" >&2
    exit 4
  fi
}

require_strict_parity() {
  local path="$1"
  require_completed "${path}"
  if ! "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
strict = data.get("strict_zero_action_parity") or {}
aggregate = data.get("aggregate_zero_action_parity") or {}
passed = (
    strict.get("passed") is True
    and strict.get("mismatched_images") == 0
    and strict.get("actions_are_exact_zero") is True
    and aggregate.get("passed") is True
)
raise SystemExit(0 if passed else 1)
PY
  then
    echo "Native strict zero-action parity failed: ${path}" >&2
    exit 5
  fi
}

wait_for_gpu_memory() {
  while true; do
    local free_mb
    free_mb="$(
      nvidia-smi -i "${GPU_ID}" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits \
        | head -n 1 \
        | tr -d '[:space:]'
    )"
    if [[ -n "${free_mb}" && "${free_mb}" -gt "${MIN_FREE_MB}" ]]; then
      return 0
    fi
    echo "$(date -Is) GPU${GPU_ID} free=${free_mb:-unknown}MB; waiting for >${MIN_FREE_MB}MB"
    sleep "${POLL_SECONDS}"
  done
}

skip_or_fail_incomplete() {
  local metrics="$1"
  if [[ ! -f "${metrics}" ]]; then
    return 1
  fi
  if grep -q '"completed": true' "${metrics}"; then
    return 0
  fi
  echo "Incomplete matrix run requires review: ${metrics}" >&2
  exit 6
}

run_full_control() {
  local checkpoint="$1"
  local run_name="ctrl_strong_full_ft4_nwpu_s42_bs${BATCH_SIZE}_ep${EPOCHS}_lr3e4"
  if skip_or_fail_incomplete "runs/${run_name}/eval_metrics.json"; then
    echo "$(date -Is) ${run_name} already complete; skipping"
    return 0
  fi
  wait_for_gpu_memory
  "${PYTHON_BIN}" scripts/round28_train_eval.py \
    --run-name "${run_name}" \
    --dataset nwpu \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --checkpoint "${checkpoint}" \
    --trainable-mode full \
    --selection-metric ap75 \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --data-seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    --per-size-ap \
    --require-clean-git
}

run_action() {
  local checkpoint="$1"
  local label="$2"
  local preserve_weight="$3"
  local run_name="action_native_${label}_strong_s42_bs${BATCH_SIZE}_ep${EPOCHS}"
  if skip_or_fail_incomplete "runs/${run_name}/eval_metrics.json"; then
    echo "$(date -Is) ${run_name} already complete; skipping"
    return 0
  fi
  wait_for_gpu_memory
  "${PYTHON_BIN}" scripts/train_energy_transport_action.py \
    --run-name "${run_name}" \
    --dataset nwpu \
    --checkpoint "${checkpoint}" \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --no-pretrained \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --data-seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --lr 0.001 \
    --residual-scale 0.0 \
    --max-score-delta 0.05 \
    --max-box-delta 0.20 \
    --box-base decoded \
    --match-mode class_aware \
    --box-weight 1.0 \
    --high-iou-preserve-weight "${preserve_weight}" \
    --energy-weight 0.0 \
    --hwm-weight 0.0 \
    --score-threshold 0.05 \
    --action-score-threshold 0.05 \
    --nms-threshold 0.50 \
    --detections-per-img 100 \
    --postprocess-mode native \
    --per-class \
    --per-size \
    --require-clean-git
}

strong_metrics="runs/${STRONG_RUN}/eval_metrics.json"
parity_metrics="runs/${PARITY_RUN}/eval_metrics.json"
require_completed "${strong_metrics}"
require_strict_parity "${parity_metrics}"
strong_checkpoint="runs/${STRONG_RUN}/checkpoint_best.pth"
if [[ ! -f "${strong_checkpoint}" ]]; then
  echo "Missing strong checkpoint: ${strong_checkpoint}" >&2
  exit 3
fi

run_full_control "${strong_checkpoint}"
run_action "${strong_checkpoint}" "boxonly" 0.0
run_action "${strong_checkpoint}" "preserve2" 2.0

echo "$(date -Is) strong native action matrix complete"
