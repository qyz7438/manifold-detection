#!/bin/bash
set -e
PYTHON=E:/anaconda/01/envs/RLimage/python.exe
CONFIG=spectral_detection_posttrain/configs/manifold_nwpu.yaml
BASELINE=runs/round2100_nwpu_baseline/checkpoint_best.pth
COMMON="--config $CONFIG --baseline $BASELINE --epochs 10 --num-prototypes 4 --lambda-tr 0.01 --lr 1e-5 --lr-manifold 1e-4 --geometry-every 0 --eval-every 1 --early-stopping-patience 10 --active-manifold-correction --active-correction-mode residual"

echo "=== Running g015_en0000 (resume interrupted) ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
    $COMMON --run-name nwpu_active_sweep_g015_en0000 --active-correction-gamma 0.15 --lambda-en 0.0

echo "=== Running g015_en0010 ==="
$PYTHON -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
    $COMMON --run-name nwpu_active_sweep_g015_en0010 --active-correction-gamma 0.15 --lambda-en 0.01

echo "=== Sweep resume complete ==="
