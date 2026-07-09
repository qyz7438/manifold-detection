#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
PYTHON="/home/ps/anaconda3/envs/RLimage/bin/python"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

wait_for_gpu2() {
  while true; do
    free_mb="$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb > 8192 )); then
      log "GPU2 free=${free_mb}MB; starting dense-gain candidate run"
      return
    fi
    log "GPU2 free=${free_mb:-unknown}MB; waiting for >8192MB"
    sleep 60
  done
}

run_one() {
  local feature_source="$1"
  local run_name="action_candidate_dense_gain_${feature_source}_s42_k73_bs8_ep4"
  local result="runs/${run_name}/eval_metrics.json"
  if [[ -f "${result}" ]] && "${PYTHON}" -c \
    "import json; raise SystemExit(0 if json.load(open('${result}')).get('completed') is True else 1)"; then
    log "${run_name} already complete"
    return
  fi

  wait_for_gpu2
  CUDA_VISIBLE_DEVICES=2 "${PYTHON}" scripts/train_candidate_action_energy.py \
    --run-name "${run_name}" \
    --detector-checkpoint runs/nwpu_mob_strong_cosine_s42_bs8_36ep/checkpoint_best.pth \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --feature-source "${feature_source}" \
    --loss-mode dense_gain \
    --epochs 4 \
    --batch-size 8 \
    --num-workers 0 \
    --seed 42 \
    --data-seed 42 \
    --hidden-dim 256 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --step-sizes 0.05,0.1,0.2 \
    --min-iou-gain 0.002 \
    --gain-beta 0.02 \
    --gain-impact-boost 10.0 \
    --gain-boundary-boost 2.0 \
    --energy-weight 0.001 \
    --min-energy-drop 0.002 \
    --min-score 0.05 \
    --require-foreground-dominant \
    --max-actions-per-image 32 \
    --min-oracle-ap75 0.36 \
    --score-threshold 0.05 \
    --nms-threshold 0.5 \
    --detections-per-img 100 \
    --shuffle-seed 31415 \
    --require-clean-git
}

run_one box
run_one spatial
log "dense-gain candidate matrix complete"
