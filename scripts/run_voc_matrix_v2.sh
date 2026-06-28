#!/bin/bash
set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

DATASET=voc
EPOCHS=12

for SEED in 42 2024 999; do
  echo "===== voc_mob_baseline seed=$SEED (bs32 lr0.024) ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --run-name voc_mob_baseline_s${SEED}_12ep

  echo "===== voc_mob_fpn_sm seed=$SEED (bs32 lr0.024) ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
    --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
    --run-name voc_mob_fpn_sm_s${SEED}_12ep

  echo "===== voc_resnet_baseline seed=$SEED (bs4 lr0.005) ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --run-name voc_resnet_baseline_s${SEED}_12ep

  echo "===== voc_resnet_fpn_sm seed=$SEED (bs4 lr0.005) ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
    --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
    --run-name voc_resnet_fpn_sm_s${SEED}_12ep
done
