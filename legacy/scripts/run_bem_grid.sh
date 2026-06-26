#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONPATH=E:/CLIproject/RLimage
PY=E:/anaconda/01/envs/RLimage/python
SCRIPT=scripts/round28_train_eval.py

SEEDS=(42 123 2024)
EPOCHS_MAIN=3

run() {
  name=$1
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $name"
  $PY $SCRIPT --run-name "$name" "$@" || echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $name"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $name"
}

for seed in "${SEEDS[@]}"; do
  run "grid_baseline_full_s${seed}_e${EPOCHS_MAIN}"      --afm-type none --trainable-mode full --epochs $EPOCHS_MAIN --seed $seed
  run "grid_baseline_boxhead_s${seed}_e${EPOCHS_MAIN}"   --afm-type none --trainable-mode box_head_only --epochs $EPOCHS_MAIN --seed $seed
  run "grid_phase_boxhead_s${seed}_e${EPOCHS_MAIN}"      --afm-type mplseg_phase_only --trainable-mode box_head_only --epochs $EPOCHS_MAIN --seed $seed
  run "grid_bem_boxhead_s${seed}_e${EPOCHS_MAIN}"        --afm-type none --use-bem --trainable-mode box_head_only --epochs $EPOCHS_MAIN --seed $seed
  run "grid_bem_phase_boxhead_s${seed}_e${EPOCHS_MAIN}"  --afm-type mplseg_phase_only --use-bem --trainable-mode box_head_only --epochs $EPOCHS_MAIN --seed $seed
done

# Extra 5-epoch BEM runs on two seeds
for seed in 42 123; do
  run "grid_bem_boxhead_s${seed}_e5" --afm-type none --use-bem --trainable-mode box_head_only --epochs 5 --seed $seed
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE"
