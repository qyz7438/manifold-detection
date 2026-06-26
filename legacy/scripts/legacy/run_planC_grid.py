"""Run an overnight Plan C experiment grid on Penn-Fudan segmentation.

All experiments use the RLimage conda environment with CUDA.  The grid is
executed sequentially on a single GPU to avoid contention.

Usage:
    E:/anaconda/01/envs/RLimage/python.exe scripts/run_planC_grid.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time

import torch


ROOT = Path("E:/CLIproject/RLimage")
CHECKPOINT = ROOT / "runs/round41_baseline_s42/checkpoint_last.pth"
PYTHON = "E:/anaconda/01/envs/RLimage/python.exe"


@dataclass
class Exp:
    name: str
    action: str
    reward: str
    objective: str
    epochs: int = 1
    sigma: float | None = None
    lr: float | None = None
    unfreeze: bool = False
    num_samples: int = 8
    flatness_interval: int = 0


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
GRID: list[Exp] = []

# Baselines: adapter-only supervised with different capacities
GRID.append(Exp("sup_logit_1ep", "logit", "ce_delta", "supervised", epochs=1, lr=1e-3))
GRID.append(Exp("sup_logit_3ep", "logit", "ce_delta", "supervised", epochs=3, lr=1e-3))
GRID.append(Exp("sup_feature_1ep", "feature", "ce_delta", "supervised", epochs=1, lr=1e-3))
GRID.append(Exp("sup_calibration_1ep", "calibration", "ce_delta", "supervised", epochs=1, lr=1e-2))

# End-to-end negative controls
GRID.append(Exp("e2e_sup_logit_1ep", "logit", "ce_delta", "supervised", epochs=1, lr=1e-4, unfreeze=True))
GRID.append(Exp("e2e_sup_logit_3ep", "logit", "ce_delta", "supervised", epochs=3, lr=1e-4, unfreeze=True))

# RLVR: logit action, scan reward and sigma
for sigma in [0.5, 1.0, 2.0]:
    GRID.append(Exp(f"rlvr_logit_ce_s{int(sigma*10):02d}", "logit", "ce_delta", "rlvr", epochs=1, sigma=sigma))
for sigma in [0.5, 1.0, 2.0]:
    GRID.append(Exp(f"rlvr_logit_iou_s{int(sigma*10):02d}", "logit", "iou_delta", "rlvr", epochs=1, sigma=sigma))
for sigma in [1.0, 2.0]:
    GRID.append(Exp(f"rlvr_logit_boundary_s{int(sigma*10):02d}", "logit", "boundary_delta", "rlvr", epochs=1, sigma=sigma))

# RLVR: feature action
GRID.append(Exp("rlvr_feature_ce_s10", "feature", "ce_delta", "rlvr", epochs=1, sigma=1.0))
GRID.append(Exp("rlvr_feature_iou_s10", "feature", "iou_delta", "rlvr", epochs=1, sigma=1.0))
GRID.append(Exp("rlvr_feature_boundary_s10", "feature", "boundary_delta", "rlvr", epochs=1, sigma=1.0))

# RLVR: calibration action
GRID.append(Exp("rlvr_calibration_ce_s10", "calibration", "ce_delta", "rlvr", epochs=1, sigma=1.0))
GRID.append(Exp("rlvr_calibration_iou_s10", "calibration", "iou_delta", "rlvr", epochs=1, sigma=1.0))

# DPO
GRID.append(Exp("dpo_logit_ce", "logit", "ce_delta", "dpo", epochs=1, lr=1e-3))
GRID.append(Exp("dpo_logit_iou", "logit", "iou_delta", "dpo", epochs=1, lr=1e-3))
GRID.append(Exp("dpo_feature_ce", "feature", "ce_delta", "dpo", epochs=1, lr=1e-3))
GRID.append(Exp("dpo_calibration_ce", "calibration", "ce_delta", "dpo", epochs=1, lr=1e-2))

# Negative control: random reward
GRID.append(Exp("rlvr_logit_random_s10", "logit", "random_delta", "rlvr", epochs=1, sigma=1.0))


def build_cmd(exp: Exp) -> list[str]:
    cmd = [
        PYTHON, "-m",
        "spectral_detection_posttrain.trainers.segmentation.action_local_seg_rlvr",
        "--run-name", f"planC_grid_{exp.name}",
        "--checkpoint", str(CHECKPOINT),
        "--action", exp.action,
        "--reward", exp.reward,
        "--objective", exp.objective,
        "--epochs", str(exp.epochs),
        "--num-samples", str(exp.num_samples),
        "--flatness-interval", str(exp.flatness_interval),
    ]
    if exp.sigma is not None:
        cmd.extend(["--sigma", str(exp.sigma)])
    if exp.lr is not None:
        cmd.extend(["--lr", str(exp.lr)])
    if exp.unfreeze:
        cmd.append("--unfreeze-baseline")
    return cmd


def run_exp(exp: Exp) -> dict:
    print(f"\n{'='*60}")
    print(f"Running: {exp.name}")
    print(f"{'='*60}")
    cmd = build_cmd(exp)
    t0 = time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    elapsed = time() - t0

    summary_path = ROOT / "runs" / f"planC_grid_{exp.name}" / "summary.json"
    result = {
        "name": exp.name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "elapsed_sec": elapsed,
        "baseline_mIoU": None,
        "best_mIoU": None,
        "delta_mIoU": None,
    }
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        baseline = summary.get("baseline_metrics", {}).get("mIoU")
        best = summary.get("best_miou")
        result["baseline_mIoU"] = baseline
        result["best_mIoU"] = best
        result["delta_mIoU"] = best - baseline if baseline is not None and best is not None else None
    return result


def main():
    # Sanity check: CUDA must be available.
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. Refusing to run on CPU.", file=sys.stderr)
        sys.exit(1)
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"Total experiments: {len(GRID)}")

    results: list[dict] = []
    for i, exp in enumerate(GRID, 1):
        print(f"\n[{i}/{len(GRID)}] {exp.name}")
        res = run_exp(exp)
        results.append(res)
        # Save incremental results
        with open(ROOT / "runs" / "planC_grid_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Final summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'name':<40} {'baseline':>10} {'best':>10} {'delta':>10} {'status':>8}")
    print("-"*80)
    for r in sorted(results, key=lambda x: x.get("delta_mIoU") or -1, reverse=True):
        print(
            f"{r['name']:<40} "
            f"{r['baseline_mIoU'] or -1:>10.4f} "
            f"{r['best_mIoU'] or -1:>10.4f} "
            f"{r['delta_mIoU'] or -1:>10.4f} "
            f"{r['status']:>8}"
        )


if __name__ == "__main__":
    main()
