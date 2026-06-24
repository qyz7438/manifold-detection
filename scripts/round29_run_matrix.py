"""Run Plan 2.10 corrected matrix: B1/B2 post-training + G8/G9 RPN edge-mix."""
from __future__ import annotations

import subprocess
import sys

PYTHON = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
G1_CKPT = "runs/round29_g1_baseline_full/checkpoint_last.pth"

GROUPS = [
    ("round210_b1_ckpt_eval", "none", "current", "full", 0, False, G1_CKPT),
    ("round210_b2_posttrain", "none", "current", "box_head_only", 1, False, G1_CKPT),
    ("round210_g8_rpn_clean", "none", "current", "rpn_box_head", 1, False, None),
    ("round210_g9_rpn_mixed", "none", "current", "rpn_box_head", 1, True, None),
]

for run_name, afm_type, residual_mode, trainable_mode, epochs, edge_mix, checkpoint in GROUPS:
    cmd = [PYTHON, SCRIPT, "--run-name", run_name, "--afm-type", afm_type,
           "--afm-residual-mode", residual_mode, "--trainable-mode", trainable_mode,
           "--epochs", str(epochs), "--seed", "42"]
    if edge_mix:
        cmd.append("--edge-mix")
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
