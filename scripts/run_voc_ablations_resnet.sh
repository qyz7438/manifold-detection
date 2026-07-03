#!/bin/bash
set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

DATASET=voc
MODEL=fasterrcnn_resnet50_fpn
EPOCHS=12
SEED=42
BS=4
LR=0.005
RESIZE="--min-size 800 --max-size 1333"
COMMON="--dataset $DATASET --voc-full --model-name $MODEL --epochs $EPOCHS --seed $SEED --batch-size $BS --lr $LR $RESIZE --require-clean-git --per-size-ap --model-cost"

# Baseline
echo "--- baseline ---"
python scripts/round28_train_eval.py $COMMON --run-name voc_resnet_abl_baseline_s42_12ep

# FPN-SM default (matches main matrix)
echo "--- fpn_sm default ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name voc_resnet_abl_fpn_sm_default_s42_12ep

# FPN-real-adapter
echo "--- fpn_real ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-real-adapter --fpn-real-latent-dim 64 --fpn-real-init-alpha 0.01 \
  --run-name voc_resnet_abl_fpn_real_s42_12ep

# Latent dim
echo "--- fpn_sm latent_dim 32 ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 32 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name voc_resnet_abl_fpn_sm_lat32_s42_12ep

echo "--- fpn_sm latent_dim 128 ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 128 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name voc_resnet_abl_fpn_sm_lat128_s42_12ep

# Init alpha
echo "--- fpn_sm init_alpha 0.001 ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.001 \
  --run-name voc_resnet_abl_fpn_sm_alpha1e-3_s42_12ep

echo "--- fpn_sm init_alpha 0.1 ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.1 \
  --run-name voc_resnet_abl_fpn_sm_alpha1e-1_s42_12ep

# Freq coords
echo "--- fpn_sm freq_coords on ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name voc_resnet_abl_fpn_sm_freq_on_s42_12ep

# Level coords
echo "--- fpn_sm level_coords off ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords --no-fpn-sm-use-level-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --run-name voc_resnet_abl_fpn_sm_no_level_s42_12ep

# Gate activation
echo "--- fpn_sm gate sigmoid2 ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --fpn-sm-gate-activation sigmoid2 \
  --run-name voc_resnet_abl_fpn_sm_gate_sigmoid2_s42_12ep

echo "--- fpn_sm gate tanh ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --fpn-sm-gate-activation tanh \
  --run-name voc_resnet_abl_fpn_sm_gate_tanh_s42_12ep

# Suppress DC
echo "--- fpn_sm suppress_dc ---"
python scripts/round28_train_eval.py $COMMON \
  --fpn-spectral-manifold --no-fpn-sm-use-freq-coords \
  --fpn-sm-latent-dim 64 --fpn-sm-hidden-dim 128 --fpn-sm-init-alpha 0.01 \
  --fpn-sm-suppress-dc \
  --run-name voc_resnet_abl_fpn_sm_suppress_dc_s42_12ep

echo "===== VOC ResNet50 ablations complete ====="
