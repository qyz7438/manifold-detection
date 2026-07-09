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
EPOCHS="${EPOCHS:-8}"
LR="${LR:-0.0003}"
C0_RUN="ctrl_full_ft18_from12best_fullnw0_nwpu_s42_bs8_ep18"
PARITY_RUN="det_action_zero_parity_fullft18best_s42"
RUN_NAME="ctrl_full_refine8_from_ft18last_fullnw0_nwpu_s42_bs${BATCH_SIZE}_ep${EPOCHS}_lr3e4"

if [[ "${GPU_ID}" != "2" ]]; then
  echo "This experiment is restricted to GPU2; got GPU_ID=${GPU_ID}." >&2
  exit 2
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

wait_for_completed_parity() {
  local metrics="runs/${PARITY_RUN}/eval_metrics.json"
  while [[ ! -f "${metrics}" ]] || ! grep -q '"completed": true' "${metrics}"; do
    echo "$(date -Is) waiting for completed parity diagnostics: ${metrics}"
    sleep "${POLL_SECONDS}"
  done
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

checkpoint="runs/${C0_RUN}/checkpoint_last.pth"
metrics="runs/${RUN_NAME}/eval_metrics.json"

wait_for_completed_parity
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing stage-2 source checkpoint: ${checkpoint}" >&2
  exit 3
fi
if [[ -f "${metrics}" ]]; then
  if grep -q '"completed": true' "${metrics}"; then
    echo "$(date -Is) ${RUN_NAME} already complete; skipping"
    exit 0
  fi
  echo "Incomplete stage-2 run requires review: ${metrics}" >&2
  exit 4
fi

wait_for_gpu_memory
echo "$(date -Is) starting ${RUN_NAME} from ${checkpoint}"
"${PYTHON_BIN}" scripts/round28_train_eval.py \
  --run-name "${RUN_NAME}" \
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

echo "$(date -Is) NWPU stage-2 full-model refinement complete"
