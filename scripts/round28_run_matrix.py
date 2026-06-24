"""Run Round 2.8 9-group diagnostic matrix."""
from __future__ import annotations

import subprocess
import sys

GROUPS = [
    ("round28_g01_baseline_full", "none", "current", "full"),
    ("round28_g02_old_afm_full", "old", "current", "full"),
    ("round28_g03_identity_current_full", "identity", "current", "full"),
    ("round28_g04_identity_delta_full", "identity", "delta", "full"),
    ("round28_g05_identity_norm_delta_full", "identity", "norm_delta", "full"),
    ("round28_g06_baseline_box_head_only", "none", "current", "box_head_only"),
    ("round28_g07_identity_current_afm_only", "identity", "current", "afm_only"),
    ("round28_g08_identity_current_afm_box_head", "identity", "current", "afm_box_head"),
    ("round28_g09_identity_delta_afm_box_head", "identity", "delta", "afm_box_head"),
]


def main() -> None:
    for run_name, afm_type, residual_mode, trainable_mode in GROUPS:
        cmd = [
            sys.executable, "scripts/round28_train_eval.py",
            "--run-name", run_name, "--afm-type", afm_type,
            "--afm-residual-mode", residual_mode, "--trainable-mode", trainable_mode,
            "--epochs", "1", "--seed", "42",
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
