#!/bin/bash
set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

# ============================================================
# M3: COCO 17 smoke run (500 train / 200 val, seed 42)
# ============================================================
echo "===== M3: COCO smoke baseline ====="
python scripts/round28_train_eval.py \
  --dataset coco --coco-root ./data/coco \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 1 --seed 42 --batch-size 8 --lr 0.004 \
  --limit-train 500 --limit-val 200 \
  --run-name coco_mob_baseline_smoke

echo "===== M3: COCO smoke FPN-SM ====="
python scripts/round28_train_eval.py \
  --dataset coco --coco-root ./data/coco \
  --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs 1 --seed 42 --batch-size 8 --lr 0.004 \
  --limit-train 500 --limit-val 200 \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name coco_mob_fpn_sm_smoke

# ============================================================
# M4: NWPU VHR-10 seed 999 (third seed)
# ============================================================
DATASET=nwpu
EPOCHS=12
SEED=999

echo "===== M4: nwpu_mob_baseline seed=$SEED ====="
python scripts/round28_train_eval.py \
  --dataset $DATASET --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs $EPOCHS --seed $SEED --batch-size 16 \
  --run-name nwpu_mob_baseline_s${SEED}_12ep

echo "===== M4: nwpu_mob_fpn_sm seed=$SEED ====="
python scripts/round28_train_eval.py \
  --dataset $DATASET --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
  --epochs $EPOCHS --seed $SEED --batch-size 16 \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name nwpu_mob_fpn_sm_s${SEED}_12ep

echo "===== M4: nwpu_resnet_baseline seed=$SEED ====="
python scripts/round28_train_eval.py \
  --dataset $DATASET --model-name fasterrcnn_resnet50_fpn \
  --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
  --min-size 800 --max-size 1333 \
  --run-name nwpu_resnet_baseline_s${SEED}_12ep

echo "===== M4: nwpu_resnet_fpn_sm seed=$SEED ====="
python scripts/round28_train_eval.py \
  --dataset $DATASET --model-name fasterrcnn_resnet50_fpn \
  --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
  --min-size 800 --max-size 1333 \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name nwpu_resnet_fpn_sm_s${SEED}_12ep

echo "===== M3 + M4 (seed 999) complete ====="
