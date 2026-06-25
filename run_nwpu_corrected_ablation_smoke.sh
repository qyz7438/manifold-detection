#!/bin/bash
set -e

cd /e/CLIproject/RLimage
# Use the isolated worktree source while keeping data/runs in the main RLimage tree.
export PYTHONPATH=/e/CLIproject/manifold/.worktrees/feat-nwpu-etf-rs-redesign:$PYTHONPATH
PYTHON="/e/anaconda/01/envs/RLimage/python.exe"
CONFIG="spectral_detection_posttrain/configs/manifold_nwpu.yaml"
COMMON="--config $CONFIG --baseline runs/nwpu_baseline_best.pth --lr 1e-5 --lr-manifold 1e-4 --lambda-tr 0.01 --lambda-en 0.001 --class-reweight inv_sqrt --lambda-proj-intra 0.1 --lambda-proj-inter 0.1 --projection-inter-margin 0.5 --epochs 2 --limit-train 50 --limit-val 50 --early-stopping-patience 5"

echo "=== Arm A: baseline ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_baseline_smoke

echo "=== Arm B: +RS bank ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_rs_smoke --rs-orient-bins 4 --rs-scale-bins 3

echo "=== Arm C: +ETF (fixed) ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_etf_smoke --use-etf-classifier --etf-use-projector --lambda-logit-preserve 1.0 --lambda-bbox-preserve 1.0

echo "=== Arm D: +RS+ETF ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_rs_etf_smoke --rs-orient-bins 4 --rs-scale-bins 3 --use-etf-classifier --etf-use-projector --lambda-logit-preserve 1.0 --lambda-bbox-preserve 1.0

echo "All corrected ablation smoke runs completed."
