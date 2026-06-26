"""Plan C adapter post-training grid (supervised only).

This grid reflects the ALAPT central idea: compare action spaces and
feature-adapter residual scales on a frozen Penn-Fudan FCN baseline.
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
    epochs: int = 3
    lr: float = 1e-3
    scale: float = 1.0
    unfreeze: bool = False
    flatness_interval: int = 0


GRID: list[Exp] = []

# Reference baselines
GRID.append(Exp("sup_logit_3ep", "logit", epochs=3, lr=1e-3, scale=1.0))
GRID.append(Exp("sup_calibration_3ep", "calibration", epochs=3, lr=1e-3, scale=1.0))

# Feature adapter: residual scale sweep
for scale in [0.1, 0.3, 0.6, 1.0, 2.0]:
    GRID.append(Exp(f"sup_feature_scale{str(scale).replace('.', '')}_3ep", "feature",
                    epochs=3, lr=1e-3, scale=scale))

# End-to-end backbone variants (action=logit leaves classifier head frozen)
GRID.append(Exp("e2e_sup_logit_3ep", "logit", epochs=3, lr=1e-4, scale=1.0, unfreeze=True))
GRID.append(Exp("e2e_sup_feature_scale10_3ep", "feature", epochs=3, lr=1e-4, scale=1.0, unfreeze=True))


def build_cmd(exp: Exp) -> list[str]:
    cmd = [
        PYTHON, "-m",
        "spectral_detection_posttrain.trainers.segmentation.action_local_adapter",
        "--run-name", f"planC_adapter_{exp.name}",
        "--checkpoint", str(CHECKPOINT),
        "--action", exp.action,
        "--epochs", str(exp.epochs),
        "--lr", str(exp.lr),
        "--scale", str(exp.scale),
        "--flatness-interval", str(exp.flatness_interval),
    ]
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

    summary_path = ROOT / "runs" / f"planC_adapter_{exp.name}" / "summary.json"
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
        with open(ROOT / "runs" / "planC_adapter_grid_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'name':<40} {'baseline':>10} {'best':>10} {'delta':>10} {'status':>8}")
    print("-"*80)
    for r in sorted(results, key=lambda x: x.get("delta_mIoU") if x.get("delta_mIoU") is not None else -1, reverse=True):
        print(
            f"{r['name']:<40} "
            f"{r['baseline_mIoU'] or -1:>10.4f} "
            f"{r['best_mIoU'] or -1:>10.4f} "
            f"{r['delta_mIoU'] if r['delta_mIoU'] is not None else -1:>10.4f} "
            f"{r['status']:>8}"
        )


if __name__ == "__main__":
    main()
