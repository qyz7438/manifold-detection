#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ps/lzz/manifold-detection-energy-transport}"
PYTHON_BIN="${PYTHON_BIN:-/home/ps/anaconda3/envs/RLimage/bin/python}"
GPU_ID="${GPU_ID:-2}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data/NWPU VHR-10 dataset}"
ANNOTATION="${ANNOTATION:-${ROOT_DIR}/data/NWPU_VHR10_coco.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${ROOT_DIR}/spectral_detection_posttrain/configs/splits/nwpu_re_roi_counterfactual_s42_nested.json}"
CACHE="${CACHE:?Set CACHE to the committed fit-only re-ROI cache artifact}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the locked source detector checkpoint}"
GROUP_NAME="${GROUP_NAME:-nwpu_oracle_utility_boxhead_v001}"

if [[ "${GPU_ID}" != "2" ]]; then
  echo "ERROR: this protocol permits GPU2 only" >&2
  exit 64
fi

require_gpu2_memory() {
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d '[:space:]')"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib <= 8192 )); then
    echo "ERROR: GPU2 memory.free=${free_mib:-unknown} MiB; require strictly >8192 MiB" >&2
    exit 75
  fi
}

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=2

for required in "${ANNOTATION}" "${SPLIT_MANIFEST}" "${CACHE}" "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required artifact missing: ${required}" >&2
    exit 66
  fi
done

for seed in 42 2024 999; do
  for arm in Z U W F S; do
    run_dir="runs/${GROUP_NAME}_s${seed}_${arm}"
    if [[ -f "${run_dir}/eval_metrics.json" ]]; then
      echo "SKIP completed ${run_dir}"
      continue
    fi
    if [[ -e "${run_dir}" ]]; then
      echo "ERROR: refusing incomplete or reused run directory ${run_dir}" >&2
      exit 73
    fi
    require_gpu2_memory
    "${PYTHON_BIN}" scripts/experiments/awr_weighted_boxhead/train.py \
      --arm "${arm}" \
      --seed "${seed}" \
      --data-root "${DATA_ROOT}" \
      --annotation "${ANNOTATION}" \
      --split-manifest "${SPLIT_MANIFEST}" \
      --cache "${CACHE}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${run_dir}" \
      --device cuda
  done
done

analysis_args=()
for seed in 42 2024 999; do
  for arm in U W F S; do
    analysis_args+=(--run-dir "runs/${GROUP_NAME}_s${seed}_${arm}")
  done
done

"${PYTHON_BIN}" scripts/experiments/awr_weighted_boxhead/analyze_group.py \
  "${analysis_args[@]}" \
  --output "runs/${GROUP_NAME}_group_metrics.json" \
  --resamples 10000
