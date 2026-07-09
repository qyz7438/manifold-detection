#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
PYTHON="/home/ps/anaconda3/envs/RLimage/bin/python"
DIAGNOSTIC="${ROOT}/runs/diag_action_direction_gatebound_strong_s42_fullval/diagnostics.json"
RUN_NAME="action_benefit_energy_preserve2_strong_s42_bs8_ep4"
RESULT="${ROOT}/runs/${RUN_NAME}/eval_metrics.json"
MIN_ORACLE_ACCEPT_AP75="0.36"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

json_complete() {
  "${PYTHON}" - "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("completed") is True else 1)
PY
}

gate_has_headroom() {
  "${PYTHON}" - "$DIAGNOSTIC" "$MIN_ORACLE_ACCEPT_AP75" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = payload["results"]["preserve2"]["decision"]
best = decision["best_oracle_accept"]
value = float(best["oracle_accept_ap75"])
threshold = float(sys.argv[2])
print(f"oracle_accept_ap75={value:.6f} scale={best['scale']} required={threshold:.6f}")
raise SystemExit(0 if value >= threshold else 1)
PY
}

wait_for_gpu2() {
  while true; do
    free_mb="$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb > 8192 )); then
      log "GPU2 free=${free_mb}MB; starting benefit energy training"
      return
    fi
    log "GPU2 free=${free_mb:-unknown}MB; waiting for >8192MB"
    sleep 60
  done
}

if json_complete "$RESULT"; then
  log "${RUN_NAME} already complete"
  exit 0
fi

while ! json_complete "$DIAGNOSTIC"; do
  log "waiting for oracle-accept diagnostic"
  sleep 60
done

if ! gate_has_headroom; then
  log "oracle-accept upper bound below ${MIN_ORACLE_ACCEPT_AP75}; benefit gate is no-go"
  exit 2
fi

wait_for_gpu2
CUDA_VISIBLE_DEVICES=2 "${PYTHON}" scripts/train_action_benefit_energy.py \
  --run-name "${RUN_NAME}" \
  --detector-checkpoint runs/nwpu_mob_strong_cosine_s42_bs8_36ep/checkpoint_best.pth \
  --action-run-dir runs/action_native_preserve2_strong_s42_bs8_ep4 \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 4 \
  --batch-size 8 \
  --num-workers 0 \
  --seed 42 \
  --data-seed 42 \
  --hidden-dim 256 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --min-positive-gain 0.005 \
  --temperature 0.05 \
  --ranking-weight 1.0 \
  --regression-weight 1.0 \
  --boundary-boost 2.0 \
  --foreground-boost 1.0 \
  --gate-threshold 0.0 \
  --eval-gate-thresholds=-0.01,-0.005,0,0.0025,0.005,0.01 \
  --min-score 0.05 \
  --require-foreground-dominant \
  --max-actions-per-image 32 \
  --score-threshold 0.05 \
  --nms-threshold 0.5 \
  --detections-per-img 100 \
  --shuffle-seed 2718 \
  --require-clean-git

json_complete "$RESULT"
log "${RUN_NAME} complete"
