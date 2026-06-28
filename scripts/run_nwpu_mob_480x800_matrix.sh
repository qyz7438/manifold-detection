#!/bin/bash
set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

DATASET=nwpu
MODEL=fasterrcnn_mobilenet_v3_large_320_fpn
EPOCHS=12
BS=8
MIN=480
MAX=800

for SEED in 42 2024 999; do
  echo "===== nwpu_mob_480x800_baseline seed=$SEED ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --model-name $MODEL --epochs $EPOCHS --seed $SEED \
    --batch-size $BS --min-size $MIN --max-size $MAX \
    --run-name nwpu_mob_480x800_baseline_s${SEED}_12ep

  echo "===== nwpu_mob_480x800_fpn_sm seed=$SEED ====="
  python scripts/round28_train_eval.py \
    --dataset $DATASET --model-name $MODEL --epochs $EPOCHS --seed $SEED \
    --batch-size $BS --min-size $MIN --max-size $MAX \
    --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
    --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
    --run-name nwpu_mob_480x800_fpn_sm_s${SEED}_12ep

done
