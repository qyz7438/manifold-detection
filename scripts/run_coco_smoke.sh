#!/bin/bash
set -e
source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

# Smoke test on a small COCO subset (1 epoch, 500 train / 200 val images)
python scripts/round28_train_eval.py \
  --dataset coco --coco-root ./data/coco \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 1 --seed 42 --batch-size 8 --lr 0.024 \
  --limit-train 500 --limit-val 200 \
  --run-name coco_mob_baseline_smoke

# Optional: spectral manifold smoke
python scripts/round28_train_eval.py \
  --dataset coco --coco-root ./data/coco \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 1 --seed 42 --batch-size 8 --lr 0.024 \
  --limit-train 500 --limit-val 200 \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name coco_mob_fpn_sm_smoke
