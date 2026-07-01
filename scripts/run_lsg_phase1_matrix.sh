#!/bin/bash
# LSG v1 Phase 1 matrix on NWPU VHR-10.
# Runs only missing experiments (skips runs that already have eval_metrics.json).
# Usage: nohup bash scripts/run_lsg_phase1_matrix.sh > /tmp/lsg_phase1_matrix.log 2>&1 &

set -e

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate RLimage
export PYTHONPATH=/home/ps/lzz/RLimage:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

DATASET=nwpu
MODEL=fasterrcnn_mobilenet_v3_large_320_fpn
EPOCHS=50
BS=16
LR=0.024
TRAINABLE=box_head_only

SEEDS=(42 123 2024)

run_if_missing() {
    local run_name=$1
    shift
    local run_dir="runs/${run_name}"
    if [ -f "${run_dir}/eval_metrics.json" ]; then
        echo "SKIP ${run_name}: eval_metrics.json exists"
        return 0
    fi
    echo "===== RUN ${run_name} ====="
    python scripts/round28_train_eval.py \
        --dataset $DATASET --model-name $MODEL \
        --epochs $EPOCHS --seed $SEED --batch-size $BS --lr $LR \
        --trainable-mode $TRAINABLE \
        --run-name $run_name "$@"
}

for SEED in "${SEEDS[@]}"; do
    # Group A: Baseline v3
    run_if_missing "nwpu_baseline_v3_50ep_bs16_s${SEED}"

    # Group B: AFM hard-coded (phase-only)
    run_if_missing "nwpu_afm_phase_only_50ep_bs16_s${SEED}" \
        --afm-type mplseg_phase_only

    # Group C: LSG mag-only without radius
    run_if_missing "nwpu_lsg_v1_noradius_50ep_bs16_s${SEED}" \
        --use-lsg --lsg-alpha-init 0.1 --no-lsg-use-radius

    # Group D: LSG mag-only with radius
    run_if_missing "nwpu_lsg_v1_radius_50ep_bs16_s${SEED}" \
        --use-lsg --lsg-alpha-init 0.1 --lsg-use-radius
done

echo "LSG Phase 1 matrix completed."
