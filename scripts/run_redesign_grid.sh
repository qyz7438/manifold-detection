#!/usr/bin/env bash
# Ablation grid for the detection structure redesign.
# Runs on Penn-Fudan, 3 seeds, 3 epochs, box_head_only training.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=E:/CLIproject/RLimage
PY=E:/anaconda/01/envs/RLimage/python
SCRIPT=scripts/round28_train_eval.py
SEEDS=(42 123 2024)
EPOCHS=3

run() {
  name=$1; shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $name"
  $PY $SCRIPT --run-name "$name" "$@" || echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $name"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $name"
}

for seed in "${SEEDS[@]}"; do
  run "redesign_afm_s${seed}"        --afm-type mplseg_phase_only --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pbg_s${seed}"    --afm-type mplseg_phase_only --use-pbg --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_tam_s${seed}"    --afm-type mplseg_phase_only --use-tam --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pah_s${seed}"    --afm-type mplseg_phase_only --use-pah --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pbg_tam_s${seed}" --afm-type mplseg_phase_only --use-pbg --use-tam --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_pbg_pah_s${seed}" --afm-type mplseg_phase_only --use-pbg --use-pah --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
  run "redesign_afm_all_s${seed}"    --afm-type mplseg_phase_only --use-pbg --use-tam --use-pah --trainable-mode box_head_only --epochs $EPOCHS --seed $seed
done
