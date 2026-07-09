#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ps/lzz/manifold-detection-energy-transport}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/home/ps/lzz/RLimage/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/ps/anaconda3/envs/RLimage/bin/python}"
GPU_ID="${GPU_ID:-2}"
SEEDS="${SEEDS:-42 2024 999}"
EPOCHS="${EPOCHS:-18}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-0.001}"
MIN_FREE_MB="${MIN_FREE_MB:-8192}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ "${GPU_ID}" != "2" ]]; then
  echo "This experiment is restricted to GPU2; got GPU_ID=${GPU_ID}." >&2
  exit 2
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

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

run_one() {
  local seed="$1"
  local checkpoint="${SOURCE_RUN_ROOT}/nwpu_mob_baseline_s${seed}_12ep/checkpoint_best.pth"
  local run_name="ctrl_full_ft18_from12best_fullnw0_nwpu_s${seed}_bs${BATCH_SIZE}_ep${EPOCHS}"
  local metrics_path="runs/${run_name}/eval_metrics.json"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing source checkpoint: ${checkpoint}" >&2
    exit 3
  fi
  if [[ -f "${metrics_path}" ]]; then
    if grep -q '"completed": true' "${metrics_path}"; then
      echo "$(date -Is) ${run_name} already complete; skipping"
      return 0
    fi
    echo "Incomplete or failed run requires review: ${metrics_path}" >&2
    exit 4
  fi

  wait_for_gpu_memory
  echo "$(date -Is) starting ${run_name} from ${checkpoint}"
  "${PYTHON_BIN}" scripts/round28_train_eval.py \
    --run-name "${run_name}" \
    --dataset nwpu \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --checkpoint "${checkpoint}" \
    --trainable-mode full \
    --selection-metric ap75 \
    --epochs "${EPOCHS}" \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    --per-size-ap \
    --require-clean-git
}

for seed in ${SEEDS}; do
  run_one "${seed}"
done

echo "$(date -Is) NWPU full-model convergence control complete"
