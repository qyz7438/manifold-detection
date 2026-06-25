#!/bin/bash
set -e

# Remote full-dataset ablation on GPU 2.
# Run from the repo root on the Linux server.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=2
PYTHON="/home/ps/anaconda3/envs/manifold/bin/python"
CONFIG="spectral_detection_posttrain/configs/manifold_nwpu.yaml"
COMMON="--config $CONFIG --baseline runs/nwpu_baseline_best.pth --lr 1e-5 --lr-manifold 1e-4 --lambda-tr 0.01 --lambda-en 0.001 --class-reweight inv_sqrt --lambda-proj-intra 0.1 --lambda-proj-inter 0.1 --projection-inter-margin 0.5 --batch-size 8 --epochs 10 --eval-every 1 --early-stopping-patience 5"

echo "=== Arm A: baseline ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_baseline_remote

echo "=== Arm B: +RS bank ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_rs_remote --rs-orient-bins 4 --rs-scale-bins 3

echo "=== Arm C: +ETF (fixed) ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_etf_remote --use-etf-classifier --etf-use-projector --lambda-logit-preserve 1.0 --lambda-bbox-preserve 1.0

echo "=== Arm D: +RS+ETF ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
  $COMMON --run-name nwpu_corrected_rs_etf_remote --rs-orient-bins 4 --rs-scale-bins 3 --use-etf-classifier --etf-use-projector --lambda-logit-preserve 1.0 --lambda-bbox-preserve 1.0

echo "All remote ablation runs completed."
