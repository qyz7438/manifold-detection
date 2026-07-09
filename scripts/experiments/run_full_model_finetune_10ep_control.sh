#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ps/lzz/RLimage}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-2}"
SEEDS="${SEEDS:-42 2024 999}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-0.001}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
WAIT_FOR_GPU_MEMORY="${WAIT_FOR_GPU_MEMORY:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

wait_for_training_idle() {
  if [[ "${WAIT_FOR_IDLE}" != "1" ]]; then
    return 0
  fi

  while pgrep -f "scripts/round28_train_eval.py" >/dev/null \
    || pgrep -f "scripts/train_energy_transport_action.py" >/dev/null; do
    echo "==== $(date -Is) waiting for existing training jobs on GPU${GPU_ID} ===="
    pgrep -af "scripts/round28_train_eval.py" || true
    pgrep -af "scripts/train_energy_transport_action.py" || true
    sleep 300
  done
}

wait_for_gpu_memory() {
  if [[ "${WAIT_FOR_GPU_MEMORY}" != "1" ]]; then
    return 0
  fi

  while true; do
    local free_mb
    free_mb="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
    if [[ -n "${free_mb}" && "${free_mb}" -ge "${MIN_FREE_MB}" ]]; then
      return 0
    fi
    echo "==== $(date -Is) waiting for GPU${GPU_ID} free memory >= ${MIN_FREE_MB} MB; current=${free_mb:-unknown} ===="
    nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
    sleep 300
  done
}

run_one() {
  local seed="$1"
  local checkpoint="runs/nwpu_mob_baseline_s${seed}_12ep/checkpoint_best.pth"
  local run_name="ctrl_full_ft10_fullnw0_nwpu_s${seed}_bs${BATCH_SIZE}_ep${EPOCHS}"

  wait_for_training_idle
  wait_for_gpu_memory
  echo "==== $(date -Is) ${run_name} ===="
  "${PYTHON_BIN}" scripts/round28_train_eval.py \
    --run-name "${run_name}" \
    --dataset nwpu \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --checkpoint "${checkpoint}" \
    --trainable-mode full \
    --epochs "${EPOCHS}" \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}"
}

for seed in ${SEEDS}; do
  run_one "${seed}"
done

echo "==== $(date -Is) full-model 10ep fine-tune control complete ===="
