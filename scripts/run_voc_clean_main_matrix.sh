#!/bin/bash
set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

DATASET=voc
EPOCHS=12

# Shared FPN-SM configuration used in M1 and ablations.
FPN_SM_FLAGS="--fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01"

# Shared FPN-real-adapter configuration (real-valued control baseline).
FPN_REAL_FLAGS="--fpn-real-adapter \
  --fpn-real-latent-dim 64 --fpn-real-init-alpha 0.01"

for SEED in 42 2024 999; do
  echo "===== VOC clean main matrix seed=$SEED ====="

  # ---------------- MobileNet V3 Large 320 FPN ----------------
  echo "--- voc_mob_baseline_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --require-clean-git --per-size-ap --model-cost \
    --run-name voc_mob_baseline_clean_s${SEED}_12ep

  echo "--- voc_mob_fpn_sm_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --require-clean-git --per-size-ap --model-cost \
    $FPN_SM_FLAGS \
    --run-name voc_mob_fpn_sm_clean_s${SEED}_12ep

  echo "--- voc_mob_fcanet_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --require-clean-git --per-size-ap --model-cost \
    --fpn-attention-type fcanet --fpn-attention-reduction 16 \
    --run-name voc_mob_fcanet_clean_s${SEED}_12ep

  echo "--- voc_mob_eca_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --require-clean-git --per-size-ap --model-cost \
    --fpn-attention-type eca \
    --run-name voc_mob_eca_clean_s${SEED}_12ep

  echo "--- voc_mob_fpn_real_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_mobilenet_v3_large_320_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 32 --lr 0.024 \
    --require-clean-git --per-size-ap --model-cost \
    $FPN_REAL_FLAGS \
    --run-name voc_mob_fpn_real_clean_s${SEED}_12ep

  # ---------------- ResNet50 FPN ----------------
  echo "--- voc_resnet_baseline_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --require-clean-git --per-size-ap --model-cost \
    --run-name voc_resnet_baseline_clean_s${SEED}_12ep

  echo "--- voc_resnet_fpn_sm_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --require-clean-git --per-size-ap --model-cost \
    $FPN_SM_FLAGS \
    --run-name voc_resnet_fpn_sm_clean_s${SEED}_12ep

  echo "--- voc_resnet_fcanet_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --require-clean-git --per-size-ap --model-cost \
    --fpn-attention-type fcanet --fpn-attention-reduction 16 \
    --run-name voc_resnet_fcanet_clean_s${SEED}_12ep

  echo "--- voc_resnet_eca_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --require-clean-git --per-size-ap --model-cost \
    --fpn-attention-type eca \
    --run-name voc_resnet_eca_clean_s${SEED}_12ep

  echo "--- voc_resnet_fpn_real_clean_s${SEED}_12ep ---"
  python scripts/round28_train_eval.py \
    --dataset $DATASET --voc-full --model-name fasterrcnn_resnet50_fpn \
    --epochs $EPOCHS --seed $SEED --batch-size 4 --lr 0.005 \
    --min-size 800 --max-size 1333 \
    --require-clean-git --per-size-ap --model-cost \
    $FPN_REAL_FLAGS \
    --run-name voc_resnet_fpn_real_clean_s${SEED}_12ep
done

echo "===== VOC clean main matrix complete ====="
