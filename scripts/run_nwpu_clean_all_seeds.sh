#!/bin/bash
set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

DATASET=nwpu
EPOCHS=12

FPN_SM_FLAGS="--fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01"

for SEED in 42 2024 999; do
  echo "===== NWPU clean seed=$SEED ====="

  echo "--- nwpu_mob_baseline_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 16 \
    --require-clean-git --per-size-ap --model-cost \
    --run-name nwpu_mob_baseline_clean_s${SEED}_12ep

  echo "--- nwpu_mob_fpn_sm_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 16 \
    --require-clean-git --per-size-ap --model-cost \
    $FPN_SM_FLAGS \
    --run-name nwpu_mob_fpn_sm_clean_s${SEED}_12ep

  echo "--- nwpu_resnet_baseline_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --require-clean-git --per-size-ap --model-cost \
    --run-name nwpu_resnet_baseline_clean_s${SEED}_12ep

  echo "--- nwpu_resnet_fpn_sm_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --require-clean-git --per-size-ap --model-cost \
    $FPN_SM_FLAGS \
    --run-name nwpu_resnet_fpn_sm_clean_s${SEED}_12ep
done

echo "===== NWPU clean all seeds complete ====="
