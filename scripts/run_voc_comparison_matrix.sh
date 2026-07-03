#!/bin/bash
set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

DATASET=voc
EPOCHS=12

for SEED in 42 2024 999; do
  echo "===== voc_mob_fcanet seed=$SEED ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --fpn-attention-type fcanet --fpn-attention-reduction 16 \
    --run-name voc_mob_fcanet_s${SEED}_12ep

  echo "===== voc_mob_eca seed=$SEED ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --fpn-attention-type eca \
    --run-name voc_mob_eca_s${SEED}_12ep

  echo "===== voc_resnet_fcanet seed=$SEED (bs4 lr0.005) ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --fpn-attention-type fcanet --fpn-attention-reduction 16 \
    --run-name voc_resnet_fcanet_s${SEED}_12ep

  echo "===== voc_resnet_eca seed=$SEED (bs4 lr0.005) ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --fpn-attention-type eca \
    --run-name voc_resnet_eca_s${SEED}_12ep
done
