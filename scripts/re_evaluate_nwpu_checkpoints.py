#!/usr/bin/env python3
"""
Re-evaluate NWPU_VHR10 post-training checkpoints with clean detector settings.

After the eval pollution fix (commit 8c455be), rounds 2129-2207 were trained with
rollout settings leaking into evaluation (score_threshold=0.001 instead of 0.05).
This script re-evaluates checkpoints from those rounds using clean settings:
  - score_threshold=0.05 (standard eval threshold)
  - detections_per_img=100 (standard eval limit)

Usage:
    python scripts/re_evaluate_nwpu_checkpoints.py --run-dir runs/round2202_hd_fusion_tp06_pca96_logistic_15ep
    python scripts/re_evaluate_nwpu_checkpoints.py --run-dir runs/round2203_raw_ifft_A3_maxprop100_15ep --save-clean

Batch mode:
    python scripts/re_evaluate_nwpu_checkpoints.py --batch --prefix round220
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.models.bbox_adapter import install_residual_bbox_adapter

from scripts.round2129_nwpu_posttrain_smoke import (
    DATA,
    ANNOT,
    NWPUDataset,
    build_loaders,
    build_nwpu_model,
    collate,
    configure_detector_eval,
    evaluate,
    evaluate_clean_detector,
)

DETECTIONS_PER_IMG = 100
SCORE_THRESHOLD = 0.05


def re_evaluate(
    run_dir: Path,
    checkpoint_name: str = "checkpoint_best.pth",
    save: bool = False,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint and re-evaluate with clean detector settings.

    Returns the clean evaluation metrics dict.
    """
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_nwpu_model(device)

    # Read round_config.json to determine if adapters need to be installed
    round_config_path = run_dir / "round_config.json"
    if round_config_path.exists():
        with open(round_config_path, encoding="utf-8") as f:
            round_config = json.load(f)
    else:
        round_config = {}

    # Check if the checkpoint contains adapter keys (rescue-mode models)
    ckpt = torch.load(checkpoint_path, map_location=device)
    has_adapter_keys = any("adapter" in k for k in ckpt["model"])
    rescue_mode = round_config.get("rescue_mode", False)

    if has_adapter_keys or rescue_mode:
        install_residual_bbox_adapter(
            model,
            hidden_dim=128,
            scale=1.0,
            enable_cls_adapter=True,
            cls_scale=float(round_config.get("cls_adapter_scale", 1.0)),
        )

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Build NWPU validation loader (same as training script)
    _, val_loader = build_loaders(
        argparse.Namespace(
            limit_train=100000,
            limit_val=100000,
            batch_size=2,
        )
    )

    # Run clean evaluation
    metrics = evaluate_clean_detector(
        model,
        val_loader,
        device,
        score_threshold=SCORE_THRESHOLD,
        detections_per_img=DETECTIONS_PER_IMG,
    )

    if save:
        output_path = run_dir / "clean_eval_metrics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved clean metrics to {output_path}")
    else:
        print(json.dumps(metrics, indent=2))

    return metrics


def find_checkpoint_dirs(base_dir: str = "runs", prefix: str = "round22") -> list[Path]:
    """Find run directories that contain checkpoint_best.pth."""
    runs_path = Path(base_dir)
    if not runs_path.is_dir():
        return []
    dirs = sorted(runs_path.iterdir())
    if prefix:
        dirs = [d for d in dirs if d.name.startswith(prefix) and d.is_dir()]
    return [d for d in dirs if (d / "checkpoint_best.pth").exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate NWPU checkpoints with clean eval settings.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Single run directory to re-evaluate")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth", help="Checkpoint filename (default: checkpoint_best.pth)")
    parser.add_argument("--save-clean", action="store_true", help="Save clean_eval_metrics.json in the run directory")
    parser.add_argument("--batch", action="store_true", help="Batch mode: re-evaluate all matching run dirs")
    parser.add_argument("--prefix", default="round22", help="Prefix filter for batch mode (default: round22)")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Eval settings: score_threshold={SCORE_THRESHOLD}, detections_per_img={DETECTIONS_PER_IMG}")

    results: list[dict] = []

    if args.run_dir is not None:
        # Single-run mode
        run_dirs = [args.run_dir]
    elif args.batch:
        run_dirs = find_checkpoint_dirs(prefix=args.prefix)
        print(f"Found {len(run_dirs)} run directories with prefix '{args.prefix}'")
        for d in run_dirs:
            print(f"  {d.name}")
    else:
        parser.error("Specify --run-dir or --batch")

    for run_dir in run_dirs:
        print(f"\n{'='*60}")
        print(f"Re-evaluating: {run_dir.name}")
        print(f"{'='*60}")
        try:
            metrics = re_evaluate(run_dir, checkpoint_name=args.checkpoint, save=args.save_clean, device=device)
            row = {
                "run_name": run_dir.name,
                "ap50": metrics.get("ap50", None),
                "ap75": metrics.get("ap75", None),
                "ece": metrics.get("ece", None),
                "num_predictions": metrics.get("num_predictions", None),
                "fp_rate": metrics.get("false_positive_rate", None),
            }
            results.append(row)
            fp_str = f"{row['fp_rate']:.4f}" if row['fp_rate'] is not None else "N/A"
            print(f"  AP50={row['ap50']:.4f}  AP75={row['ap75']:.4f}  ECE={row['ece']:.4f}  Pred={row['num_predictions']}  FP={fp_str}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"run_name": run_dir.name, "error": str(e)})

    # Print summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Run':40s} {'AP50':>8s} {'AP75':>8s} {'ECE':>8s} {'Pred':>6s} {'FP':>8s}")
        print("-" * 80)
        for r in results:
            run = r.get("run_name", "?")
            ap50 = f"{r['ap50']:.4f}" if r.get("ap50") is not None else "N/A"
            ap75 = f"{r['ap75']:.4f}" if r.get("ap75") is not None else "N/A"
            ece = f"{r['ece']:.4f}" if r.get("ece") is not None else "N/A"
            pred = str(r.get("num_predictions", "N/A"))
            fp_rate = r.get("fp_rate", None)
            fp = f"{fp_rate:.4f}" if fp_rate is not None else "N/A"
            print(f"{run:40s} {ap50:>8s} {ap75:>8s} {ece:>8s} {pred:>6s} {fp:>8s}")

    # Save aggregate results
    summary_path = Path(f"runs/re_evaluate_nwpu_clean_{args.prefix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAggregate results saved to {summary_path}")


if __name__ == "__main__":
    main()
