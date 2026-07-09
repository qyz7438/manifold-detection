#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ps/lzz/RLimage}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-2}"
SEEDS="${SEEDS:-42 2024}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MATCH_MODE="${MATCH_MODE:-class_aware}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_one() {
  local seed="$1"
  local tag="$2"
  local checkpoint="runs/nwpu_mob_baseline_s${seed}_12ep/checkpoint_best.pth"
  shift 2

  local run_name="abl_match_${MATCH_MODE}_${tag}_fullnw0_nwpu_s${seed}_bs${BATCH_SIZE}_ep${EPOCHS}"
  echo "==== $(date -Is) ${run_name} ===="
  "${PYTHON_BIN}" scripts/train_energy_transport_action.py \
    --run-name "${run_name}" \
    --dataset nwpu \
    --checkpoint "${checkpoint}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --box-base decoded \
    --match-mode "${MATCH_MODE}" \
    "$@"
}

for seed in ${SEEDS}; do
  run_one "${seed}" "boxonly_e0" \
    --energy-weight 0.0 \
    --hwm-weight 0.0 \
    --high-iou-preserve-weight 0.0

  run_one "${seed}" "box_preserve2_e0" \
    --energy-weight 0.0 \
    --hwm-weight 0.0 \
    --high-iou-preserve-weight 2.0
done

echo "==== $(date -Is) match-mode box refinement ablation complete ===="
