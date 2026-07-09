#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ps/lzz/manifold-detection-energy-transport}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/home/ps/lzz/RLimage/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/ps/anaconda3/envs/RLimage/bin/python}"
GPU_ID="${GPU_ID:-2}"
MIN_FREE_MB="${MIN_FREE_MB:-8192}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RUN_TAG="${RUN_TAG:-nativefix}"
C0_RUN="ctrl_full_ft18_from12best_fullnw0_nwpu_s42_bs8_ep18"

if [[ "${GPU_ID}" != "2" ]]; then
  echo "This experiment is restricted to GPU2; got GPU_ID=${GPU_ID}." >&2
  exit 2
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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

wait_for_c0() {
  local metrics="runs/${C0_RUN}/eval_metrics.json"
  while [[ ! -f "${metrics}" ]] || ! grep -q '"completed": true' "${metrics}"; do
    echo "$(date -Is) waiting for completed C0 metrics: ${metrics}"
    sleep "${POLL_SECONDS}"
  done
}

run_parity() {
  local run_name="$1"
  local checkpoint="$2"
  local metrics="runs/${run_name}/eval_metrics.json"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing parity checkpoint: ${checkpoint}" >&2
    exit 3
  fi
  if [[ -f "${metrics}" ]]; then
    if grep -q '"completed": true' "${metrics}"; then
      echo "$(date -Is) ${run_name} already complete; skipping"
      return 0
    fi
    echo "Incomplete parity run requires review: ${metrics}" >&2
    exit 4
  fi

  wait_for_gpu_memory
  echo "$(date -Is) starting ${run_name} from ${checkpoint}"
  "${PYTHON_BIN}" scripts/train_energy_transport_action.py \
    --run-name "${run_name}" \
    --dataset nwpu \
    --checkpoint "${checkpoint}" \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --no-pretrained \
    --epochs 0 \
    --seed "${SEED}" \
    --data-seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 0 \
    --min-size 480 \
    --max-size 480 \
    --residual-scale 0.0 \
    --max-score-delta 0.05 \
    --max-box-delta 0.20 \
    --box-base decoded \
    --match-mode class_agnostic \
    --score-threshold 0.05 \
    --action-score-threshold 0.05 \
    --nms-threshold 0.50 \
    --detections-per-img 100 \
    --postprocess-mode native \
    --per-class \
    --per-size \
    --require-clean-git
}

wait_for_c0
run_parity \
  "det_action_zero_parity_${RUN_TAG}_baseline_s${SEED}" \
  "${SOURCE_RUN_ROOT}/nwpu_mob_baseline_s${SEED}_12ep/checkpoint_best.pth"
run_parity \
  "det_action_zero_parity_${RUN_TAG}_fullft18best_s${SEED}" \
  "runs/${C0_RUN}/checkpoint_best.pth"

echo "$(date -Is) action zero-parity controls complete"
