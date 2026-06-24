#!/bin/bash
# Preservation-loss ablation on cloud
COMMON="--config spectral_detection_posttrain/configs/manifold_nwpu.yaml --baseline runs/nwpu_baseline_best.pth --lr 1e-5 --lr-manifold 1e-4 --lambda-tr 0.01 --lambda-en 0.001 --lambda-logit-preserve 1.0 --lambda-bbox-preserve 1.0 --epochs 5 --early-stopping-patience 5"

CUDA_VISIBLE_DEVICES=0 python -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain $COMMON --run-name nwpu_head_bottleneck_r128_preserve_e5 --box-head-type bottleneck --box-head-rank 128 > cloud_logs/nwpu_head_bottleneck_r128_preserve_e5.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain $COMMON --run-name nwpu_head_convlowdim_c128_preserve_e5 --box-head-type conv_lowdim --box-head-conv-channels 128 > cloud_logs/nwpu_head_convlowdim_c128_preserve_e5.log 2>&1 &

echo "Launched preservation ablation jobs: bottleneck (GPU 0), conv_lowdim (GPU 1)"
