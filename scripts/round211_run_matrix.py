"""Run Round 2.11 VOC 6-group matrix."""
from __future__ import annotations

import subprocess
import sys

PYTHON = sys.executable
CONFIG = "spectral_detection_posttrain/configs/round211_voc.yaml"
BASELINE_CKPT = "runs/round211_voc_baseline/checkpoint_last.pth"

COMMANDS = [
    # Step 1: Train baseline
    [PYTHON, "scripts/round211_train_baseline.py",
     "--config", CONFIG, "--run-name", "round211_voc_baseline",
     "--limit-train", "300", "--limit-val", "150", "--epochs", "1"],

    # Step 2: V1 baseline eval
    [PYTHON, "scripts/round211_posttrain.py",
     "--config", CONFIG, "--checkpoint", BASELINE_CKPT,
     "--run-name", "round211_voc_v1_baseline_eval", "--mode", "eval_only",
     "--limit-val", "150"],

    # Step 3: V2 detection-only post-train
    [PYTHON, "scripts/round211_posttrain.py",
     "--config", CONFIG, "--checkpoint", BASELINE_CKPT,
     "--run-name", "round211_voc_v2_posttrain_detection_only", "--mode", "detection_only",
     "--limit-train", "300", "--limit-val", "150", "--epochs", "1"],

    # Step 4: V3 spatial post-train
    [PYTHON, "scripts/round211_posttrain.py",
     "--config", CONFIG, "--checkpoint", BASELINE_CKPT,
     "--run-name", "round211_voc_v3_posttrain_spatial", "--mode", "spatial",
     "--limit-train", "300", "--limit-val", "150", "--epochs", "1"],

    # Step 5: V4 spatial+spectral loggate
    [PYTHON, "scripts/round211_posttrain.py",
     "--config", CONFIG, "--checkpoint", BASELINE_CKPT,
     "--run-name", "round211_voc_v4_posttrain_spatial_spectral_loggate", "--mode", "spatial_spectral_loggate",
     "--limit-train", "300", "--limit-val", "150", "--epochs", "1"],

    # Step 6: V5 spatial+shuffled spectral
    [PYTHON, "scripts/round211_posttrain.py",
     "--config", CONFIG, "--checkpoint", BASELINE_CKPT,
     "--run-name", "round211_voc_v5_posttrain_spatial_shuffled_spectral", "--mode", "spatial_shuffled_spectral",
     "--limit-train", "300", "--limit-val", "150", "--epochs", "1"],

    # Step 7: V6 stress eval
    [PYTHON, "scripts/round211_eval_stress.py",
     "--config", CONFIG, "--limit-val", "150"],
]


def main():
    for cmd in COMMANDS:
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
