r"""Gamma × lambda_en sweep for active manifold correction on NWPU VHR-10.

This script sequentially runs a small grid of MGL-OPT active-correction
configurations on the full NWPU train/val split and writes a summary CSV with
AP50/AP75/ECE and per-class AP.

Example:
    python run_nwpu_active_sweep.py --epochs 10 --out runs/nwpu_active_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SWEEP_GRID = [
    # (gamma, lambda_en, run_name_suffix)
    (0.05, 0.001, "g005_en0001"),
    (0.15, 0.001, "g015_en0001"),
    (0.30, 0.001, "g030_en0001"),
    (0.15, 0.000, "g015_en0000"),
    (0.15, 0.010, "g015_en0010"),
]

FC1_PRESERVE_GRID = [
    # (lambda_fc1_rank, lambda_fc1_compact, fc1_rank_target, logit_keep, bbox_keep, suffix)
    (0.05, 0.02, 16, 0.2, 1.0, "mild_keep"),
    (0.50, 0.05, 8, 0.2, 1.0, "strong_keep"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["active", "fc1-preserve"], default="active")
    parser.add_argument("--config", default="spectral_detection_posttrain/configs/manifold_nwpu.yaml")
    parser.add_argument("--baseline", default="runs/round2100_nwpu_baseline/checkpoint_best.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="runs/nwpu_active_sweep_summary.csv")
    parser.add_argument("--num-prototypes", type=int, default=4)
    parser.add_argument("--lambda-tr", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-manifold", type=float, default=1e-4)
    parser.add_argument("--active-correction-mode", default="residual",
                        choices=["residual", "endpoint", "gated_endpoint", "gated-endpoint"])
    parser.add_argument("--active-endpoint-gate-init", type=float, default=0.25)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=None)
    parser.add_argument("--preserve-temperature", type=float, default=2.0)
    parser.add_argument("--lambda-proj-intra", type=float, default=0.0)
    parser.add_argument("--lambda-proto-div", type=float, default=0.0)
    parser.add_argument("--lambda-proj-inter", type=float, default=0.0)
    parser.add_argument("--projection-inter-margin", type=float, default=0.5)
    parser.add_argument("--proto-div-temperature", type=float, default=0.1)
    return parser.parse_args()


def _append_optional_int(cmd: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_training_command(
    *,
    config: str,
    baseline: str,
    run_name: str,
    epochs: int,
    num_prototypes: int,
    lambda_tr: float,
    lambda_en: float,
    lr: float,
    lr_manifold: float,
    active_gamma: float | None,
    active_correction_mode: str | None = None,
    active_endpoint_gate_init: float | None = None,
    fc1_rank: float | None = None,
    fc1_compact: float | None = None,
    fc1_rank_target: int | None = None,
    logit_preserve: float | None = None,
    bbox_preserve: float | None = None,
    preserve_temperature: float | None = None,
    proj_intra: float | None = None,
    proto_div: float | None = None,
    proj_inter: float | None = None,
    projection_inter_margin: float | None = None,
    proto_div_temperature: float | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    warmup_batches: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "spectral_detection_posttrain.trainers.detection.train_manifold_posttrain",
        "--config", config,
        "--baseline", baseline,
        "--run-name", run_name,
        "--epochs", str(epochs),
        "--num-prototypes", str(num_prototypes),
        "--lambda-tr", str(lambda_tr),
        "--lambda-en", str(lambda_en),
        "--lr", str(lr),
        "--lr-manifold", str(lr_manifold),
    ]

    if active_gamma is not None:
        cmd.extend(["--active-manifold-correction", "--active-correction-gamma", str(active_gamma)])
        if active_correction_mode is not None:
            cmd.extend(["--active-correction-mode", active_correction_mode])
        if active_endpoint_gate_init is not None:
            cmd.extend(["--active-endpoint-gate-init", str(active_endpoint_gate_init)])

    if fc1_rank is not None:
        if fc1_compact is None or fc1_rank_target is None:
            raise ValueError("fc1_compact and fc1_rank_target are required with fc1_rank")
        cmd.extend(
            [
                "--lambda-fc1-rank", str(fc1_rank),
                "--lambda-fc1-compact", str(fc1_compact),
                "--fc1-rank-target", str(fc1_rank_target),
            ]
        )

    if logit_preserve is not None or bbox_preserve is not None:
        cmd.extend(
            [
                "--lambda-logit-preserve", str(0.0 if logit_preserve is None else logit_preserve),
                "--lambda-bbox-preserve", str(0.0 if bbox_preserve is None else bbox_preserve),
            ]
        )
        if preserve_temperature is not None:
            cmd.extend(["--preserve-temperature", str(preserve_temperature)])

    if proj_intra is not None:
        cmd.extend(["--lambda-proj-intra", str(proj_intra)])
    if proto_div is not None:
        cmd.extend(["--lambda-proto-div", str(proto_div)])
    if proj_inter is not None:
        cmd.extend(["--lambda-proj-inter", str(proj_inter)])
    if projection_inter_margin is not None:
        cmd.extend(["--projection-inter-margin", str(projection_inter_margin)])
    if proto_div_temperature is not None:
        cmd.extend(["--proto-div-temperature", str(proto_div_temperature)])

    _append_optional_int(cmd, "--limit-train", limit_train)
    _append_optional_int(cmd, "--limit-val", limit_val)
    _append_optional_int(cmd, "--warmup-batches", warmup_batches)

    cmd.extend(
        [
            "--geometry-every", "0",
            "--eval-every", "1",
            "--early-stopping-patience", str(epochs),
        ]
    )
    return cmd


def run_config(
    config: str,
    baseline: str,
    gamma: float | None,
    lambda_en: float,
    run_name: str,
    epochs: int,
    num_prototypes: int,
    lambda_tr: float,
    lr: float,
    lr_manifold: float,
    mode: str,
    active_correction_mode: str | None = None,
    active_endpoint_gate_init: float | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    warmup_batches: int | None = None,
    fc1_rank: float | None = None,
    fc1_compact: float | None = None,
    fc1_rank_target: int | None = None,
    logit_preserve: float | None = None,
    bbox_preserve: float | None = None,
    preserve_temperature: float | None = None,
    proj_intra: float | None = None,
    proto_div: float | None = None,
    proj_inter: float | None = None,
    projection_inter_margin: float | None = None,
    proto_div_temperature: float | None = None,
) -> dict | None:
    cmd = build_training_command(
        config=config,
        baseline=baseline,
        run_name=run_name,
        epochs=epochs,
        num_prototypes=num_prototypes,
        lambda_tr=lambda_tr,
        lambda_en=lambda_en,
        lr=lr,
        lr_manifold=lr_manifold,
        active_gamma=gamma,
        active_correction_mode=active_correction_mode,
        active_endpoint_gate_init=active_endpoint_gate_init,
        fc1_rank=fc1_rank,
        fc1_compact=fc1_compact,
        fc1_rank_target=fc1_rank_target,
        logit_preserve=logit_preserve,
        bbox_preserve=bbox_preserve,
        preserve_temperature=preserve_temperature,
        proj_intra=proj_intra,
        proto_div=proto_div,
        proj_inter=proj_inter,
        projection_inter_margin=projection_inter_margin,
        proto_div_temperature=proto_div_temperature,
        limit_train=limit_train,
        limit_val=limit_val,
        warmup_batches=warmup_batches,
    )
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"mode={mode}, gamma={gamma}, lambda_en={lambda_en}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"FAILED: {run_name} (exit {result.returncode})")
        return None

    result_path = Path("runs") / run_name / "manifold_result.json"
    if not result_path.exists():
        print(f"Missing result file: {result_path}")
        return None

    data = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "mode": mode,
        "run_name": run_name,
        "gamma": gamma,
        "active_correction_mode": active_correction_mode,
        "active_endpoint_gate_init": active_endpoint_gate_init,
        "lambda_en": lambda_en,
        "lambda_tr": lambda_tr,
        "lambda_fc1_rank": fc1_rank,
        "lambda_fc1_compact": fc1_compact,
        "fc1_rank_target": fc1_rank_target,
        "lambda_logit_preserve": logit_preserve,
        "lambda_bbox_preserve": bbox_preserve,
        "lambda_proj_intra": proj_intra,
        "lambda_proto_div": proto_div,
        "lambda_proj_inter": proj_inter,
        "lr": lr,
        "lr_manifold": lr_manifold,
        "epochs": epochs,
        "best_epoch": data.get("best_epoch"),
        "initial_ap50": data.get("initial_val_ap50"),
        "initial_ap75": data.get("initial_val_ap75"),
        "best_ap50": data.get("best_val_ap50"),
        "best_ap75": data.get("best_val_ap75"),
        "best_ece": data.get("best_metrics", {}).get("ece"),
        "per_class_ap50": json.dumps(data.get("best_metrics", {}).get("per_class_ap50", {})),
        "per_class_ap75": json.dumps(data.get("best_metrics", {}).get("per_class_ap75", {})),
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode", "run_name", "gamma", "active_correction_mode", "active_endpoint_gate_init",
        "lambda_en", "lambda_tr",
        "lambda_fc1_rank", "lambda_fc1_compact", "fc1_rank_target",
        "lambda_logit_preserve", "lambda_bbox_preserve",
        "lambda_proj_intra", "lambda_proto_div", "lambda_proj_inter",
        "lr", "lr_manifold",
        "epochs", "best_epoch", "initial_ap50", "initial_ap75",
        "best_ap50", "best_ap75", "best_ece", "per_class_ap50", "per_class_ap75",
    ]

    rows = []
    if args.mode == "active":
        configs = [
            {
                "run_name": f"nwpu_active_sweep_{suffix}",
                "gamma": gamma,
                "lambda_en": lambda_en,
                "lambda_tr": args.lambda_tr,
                "active_correction_mode": args.active_correction_mode,
                "active_endpoint_gate_init": args.active_endpoint_gate_init,
            }
            for gamma, lambda_en, suffix in SWEEP_GRID
        ]
    else:
        fc1_warmup_batches = 0 if args.warmup_batches is None else args.warmup_batches
        configs = [
            {
                "run_name": f"nwpu_fc1_preserve_{suffix}",
                "gamma": None,
                "lambda_en": 0.0,
                "lambda_tr": 0.0,
                "fc1_rank": fc1_rank,
                "fc1_compact": fc1_compact,
                "fc1_rank_target": fc1_rank_target,
                "logit_preserve": logit_keep,
                "bbox_preserve": bbox_keep,
                "preserve_temperature": args.preserve_temperature,
                "proj_intra": args.lambda_proj_intra,
                "proto_div": args.lambda_proto_div,
                "proj_inter": args.lambda_proj_inter,
                "projection_inter_margin": args.projection_inter_margin,
                "proto_div_temperature": args.proto_div_temperature,
                "warmup_batches": fc1_warmup_batches,
            }
            for fc1_rank, fc1_compact, fc1_rank_target, logit_keep, bbox_keep, suffix in FC1_PRESERVE_GRID
        ]

    for sweep_config in configs:
        row = run_config(
            args.config,
            args.baseline,
            sweep_config["gamma"],
            sweep_config["lambda_en"],
            sweep_config["run_name"],
            args.epochs,
            args.num_prototypes,
            sweep_config["lambda_tr"],
            args.lr,
            args.lr_manifold,
            args.mode,
            active_correction_mode=sweep_config.get("active_correction_mode"),
            active_endpoint_gate_init=sweep_config.get("active_endpoint_gate_init"),
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            warmup_batches=sweep_config.get("warmup_batches", args.warmup_batches),
            fc1_rank=sweep_config.get("fc1_rank"),
            fc1_compact=sweep_config.get("fc1_compact"),
            fc1_rank_target=sweep_config.get("fc1_rank_target"),
            logit_preserve=sweep_config.get("logit_preserve"),
            bbox_preserve=sweep_config.get("bbox_preserve"),
            preserve_temperature=sweep_config.get("preserve_temperature"),
            proj_intra=sweep_config.get("proj_intra"),
            proto_div=sweep_config.get("proto_div"),
            proj_inter=sweep_config.get("proj_inter"),
            projection_inter_margin=sweep_config.get("projection_inter_margin"),
            proto_div_temperature=sweep_config.get("proto_div_temperature"),
        )
        if row is not None:
            rows.append(row)
            # Write incremental summary so partial results are visible.
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nIncremental summary written to {out_path}")

    print("\n=== SWEEP COMPLETE ===")
    for row in rows:
        print(
            f"{row['run_name']:25s}  "
            f"AP50 {row['initial_ap50']:.4f} -> {row['best_ap50']:.4f}  "
            f"AP75 {row['initial_ap75']:.4f} -> {row['best_ap75']:.4f}  "
            f"ECE {row['best_ece']:.4f}"
        )


if __name__ == "__main__":
    main()
