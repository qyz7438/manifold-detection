#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
PYTHON="/home/ps/anaconda3/envs/RLimage/bin/python"
DETECTOR="runs/nwpu_mob_strong_cosine_s42_bs8_36ep/checkpoint_best.pth"
ENERGY="runs/action_candidate_energy_strong_s42_k73_bs8_ep4/candidate_energy_best_ap75.pth"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

wait_for_gpu2() {
  while true; do
    free_mb="$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb > 8192 )); then
      log "GPU2 free=${free_mb}MB; starting next candidate policy eval"
      return
    fi
    log "GPU2 free=${free_mb:-unknown}MB; waiting for >8192MB"
    sleep 60
  done
}

run_eval() {
  local budget="$1"
  local drop="$2"
  local drop_tag="${drop//./p}"
  local run_name="action_candidate_policy_ep3_s42_b${budget}_d${drop_tag}"
  local result="runs/${run_name}/eval_metrics.json"
  if [[ -f "${result}" ]] && "${PYTHON}" -c \
    "import json; raise SystemExit(0 if json.load(open('${result}')).get('completed') is True else 1)"; then
    log "${run_name} already complete"
    return
  fi

  wait_for_gpu2
  CUDA_VISIBLE_DEVICES=2 "${PYTHON}" scripts/train_candidate_action_energy.py \
    --run-name "${run_name}" \
    --detector-checkpoint "${DETECTOR}" \
    --energy-checkpoint "${ENERGY}" \
    --eval-only \
    --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs 0 \
    --batch-size 8 \
    --num-workers 0 \
    --seed 42 \
    --data-seed 42 \
    --hidden-dim 256 \
    --step-sizes 0.05,0.1,0.2 \
    --min-iou-gain 0.002 \
    --min-energy-drop "${drop}" \
    --min-score 0.05 \
    --require-foreground-dominant \
    --max-actions-per-image "${budget}" \
    --min-oracle-ap75 0.0 \
    --score-threshold 0.05 \
    --nms-threshold 0.5 \
    --detections-per-img 100 \
    --shuffle-seed 31415 \
    --require-clean-git
}

for budget in 1 2 4 8 16 32; do
  run_eval "${budget}" 0.0
done

for drop in 0.02 0.04 0.06 0.08 0.10; do
  run_eval 32 "${drop}"
done

log "candidate policy sweep complete"
