#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ps/lzz/RLimage}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-2}"
SEEDS="${SEEDS:-42 2024}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-0.001}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"

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

run_one() {
  local seed="$1"
  local checkpoint="runs/nwpu_mob_baseline_s${seed}_12ep/checkpoint_best.pth"
  local run_name="ctrl_boxhead_ft_fullnw0_nwpu_s${seed}_bs${BATCH_SIZE}_ep${EPOCHS}"

  wait_for_training_idle
  echo "==== $(date -Is) ${run_name} ===="
  "${PYTHON_BIN}" scripts/round28_train_eval.py \
    --run-name "${run_name}" \
    --dataset nwpu \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --checkpoint "${checkpoint}" \
    --trainable-mode box_head_only \
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

echo "==== $(date -Is) box-head fine-tune control complete ===="
