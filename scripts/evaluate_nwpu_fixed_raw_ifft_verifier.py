from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.analysis.dimensionality import binary_ranking_metrics
from spectral_detection_posttrain.analysis.raw_ifft_verifier import (
    apply_selection_policy,
    calibrate_precision_threshold,
    fit_train_effect_scorer,
    threshold_metrics,
)


LEGACY_FEATURE_NAMES = [
    "raw_edge",
    "phase_edge",
    "hp015_edge",
    "fft_edge_truncation",
    "low_edge",
    "high_edge",
    "high_minus_low_edge",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "high_low_energy_ratio",
    "phase_abs_low",
    "phase_abs_mid",
    "phase_abs_high",
    "negative_phase_abs_low",
    "negative_phase_abs_high",
    "energy_times_negative_phase_high",
    "entropy",
    "center_surround",
    "laplacian",
    "autocorr_peak",
    "phase_std",
    "phase_abs_diff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen raw-iFFT verifier with train-calibrated thresholds.")
    parser.add_argument(
        "--cache",
        default="runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz",
    )
    parser.add_argument("--run-name", default="round2152_fixed_raw_ifft_verifier_ap75")
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "fft_edge_truncation@64",
            "phase_edge@64",
            "phase_abs_high@11",
        ],
        help="Frozen feature set. Format: legacy_feature_name@crop_size.",
    )
    parser.add_argument("--target-precisions", type=float, nargs="+", default=[0.7, 0.8, 0.85, 0.9])
    parser.add_argument("--margin-std-fracs", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--methods", nargs="+", default=["train_effect_sum", "rank_sum"])
    parser.add_argument("--primary-feature", default="fft_edge_truncation@64")
    parser.add_argument("--primary-target-precisions", type=float, nargs="+", default=[0.0, 0.8, 0.9])
    parser.add_argument("--top-k-per-image", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def load_feature_matrix(data: np.lib.npyio.NpzFile, split: str, feature_specs: list[str]) -> np.ndarray:
    columns = []
    for spec in feature_specs:
        try:
            feature_name, crop_text = spec.split("@", maxsplit=1)
            crop_size = int(crop_text)
        except ValueError as exc:
            raise ValueError(f"Invalid feature spec '{spec}', expected name@crop_size") from exc
        if feature_name not in LEGACY_FEATURE_NAMES:
            raise ValueError(f"Unknown legacy feature '{feature_name}' in '{spec}'")
        key = f"{split}_legacy_ifft_{crop_size}"
        if key not in data.files:
            raise ValueError(f"Missing cached feature bank '{key}' for '{spec}'")
        columns.append(np.asarray(data[key][:, LEGACY_FEATURE_NAMES.index(feature_name)], dtype=np.float64))
    return np.stack(columns, axis=1)


def evaluate_selected(selected: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = threshold_metrics(selected, labels)
    metrics["threshold"] = float(threshold)
    return metrics


def run_single_policy(
    *,
    data: np.lib.npyio.NpzFile,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    method: str,
    target_precision: float,
    margin_std_frac: float,
    primary_feature: str,
    primary_target_precision: float,
    top_k_per_image: int,
) -> dict[str, object]:
    scorer = fit_train_effect_scorer(train_features, train_labels, method=method)
    train_scores = scorer.score(train_features)
    val_scores = scorer.score(val_features)
    margin = float(np.std(train_scores) * float(margin_std_frac))
    calibration = calibrate_precision_threshold(
        train_scores,
        train_labels,
        target_precision=target_precision,
        margin=margin,
    )

    primary_train_scores = None
    primary_val_scores = None
    primary_threshold = None
    primary_calibration = None
    if primary_target_precision > 0.0:
        primary_train = load_feature_matrix(data, "train", [primary_feature])
        primary_val = load_feature_matrix(data, "val", [primary_feature])
        primary_scorer = fit_train_effect_scorer(primary_train, train_labels, method="train_effect_sum")
        primary_train_scores = primary_scorer.score(primary_train)
        primary_val_scores = primary_scorer.score(primary_val)
        primary_calibration = calibrate_precision_threshold(
            primary_train_scores,
            train_labels,
            target_precision=primary_target_precision,
            margin=0.0,
        )
        primary_threshold = primary_calibration.threshold

    train_selected = apply_selection_policy(
        train_scores,
        threshold=calibration.threshold,
        primary_scores=primary_train_scores,
        primary_threshold=primary_threshold,
        image_ids=data["train_image_ids"],
        top_k_per_image=top_k_per_image or None,
    )
    val_selected = apply_selection_policy(
        val_scores,
        threshold=calibration.threshold,
        primary_scores=primary_val_scores,
        primary_threshold=primary_threshold,
        image_ids=data["val_image_ids"],
        top_k_per_image=top_k_per_image or None,
    )

    return {
        "method": method,
        "target_precision": float(target_precision),
        "margin_std_frac": float(margin_std_frac),
        "margin": margin,
        "primary_feature": primary_feature if primary_target_precision > 0.0 else None,
        "primary_target_precision": float(primary_target_precision),
        "primary_calibration": asdict(primary_calibration) if primary_calibration is not None else None,
        "top_k_per_image": int(top_k_per_image),
        "score_weights": scorer.weights.astype(float).tolist(),
        "calibration": asdict(calibration),
        "train": {
            "ranking": binary_ranking_metrics(train_scores, train_labels),
            "fixed_threshold": evaluate_selected(train_selected, train_labels, calibration.threshold),
        },
        "val": {
            "ranking": binary_ranking_metrics(val_scores, val_labels),
            "fixed_threshold": evaluate_selected(val_selected, val_labels, calibration.threshold),
        },
    }


def policy_sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    val = row["val"]["fixed_threshold"]  # type: ignore[index]
    precision = float(val["precision"])  # type: ignore[index]
    recall = float(val["recall"])  # type: ignore[index]
    selected = float(val["selected"])  # type: ignore[index]
    fp = float(val["fp"])  # type: ignore[index]
    meets_precision = 1.0 if precision >= 0.7 else 0.0
    return (meets_precision, recall, precision, selected - fp * 0.01)


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.cache)
    train_labels = data["train_labels"].astype(bool)
    val_labels = data["val_labels"].astype(bool)
    train_features = load_feature_matrix(data, "train", args.features)
    val_features = load_feature_matrix(data, "val", args.features)

    policies = []
    for method in args.methods:
        for target_precision in args.target_precisions:
            for margin_std_frac in args.margin_std_fracs:
                for primary_target_precision in args.primary_target_precisions:
                    for top_k_per_image in args.top_k_per_image:
                        policies.append(
                            run_single_policy(
                                data=data,
                                train_features=train_features,
                                val_features=val_features,
                                train_labels=train_labels,
                                val_labels=val_labels,
                                method=method,
                                target_precision=target_precision,
                                margin_std_frac=margin_std_frac,
                                primary_feature=args.primary_feature,
                                primary_target_precision=primary_target_precision,
                                top_k_per_image=top_k_per_image,
                            )
                        )
    policies = sorted(policies, key=policy_sort_key, reverse=True)
    report: dict[str, object] = {
        "cache": str(args.cache),
        "features": list(args.features),
        "args": vars(args),
        "train": {
            "candidate_count": int(train_labels.shape[0]),
            "positive_count": int(train_labels.sum()),
            "negative_count": int((~train_labels).sum()),
        },
        "val": {
            "candidate_count": int(val_labels.shape[0]),
            "positive_count": int(val_labels.sum()),
            "negative_count": int((~val_labels).sum()),
        },
        "policies": policies,
        "leaderboard_top": policies[:40],
    }

    output_path = run_dir / "fixed_verifier_report.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    for index, row in enumerate(policies[:25], start=1):
        val = row["val"]["fixed_threshold"]  # type: ignore[index]
        train = row["train"]["fixed_threshold"]  # type: ignore[index]
        print(
            f"#{index:02d} method={row['method']} target={row['target_precision']} "
            f"margin={row['margin_std_frac']} primary={row['primary_target_precision']} "
            f"topk={row['top_k_per_image']} "
            f"val_p={val['precision']:.4f} val_r={val['recall']:.4f} "
            f"val_sel={val['selected']} val_tp={val['tp']} val_fp={val['fp']} "
            f"train_p={train['precision']:.4f} train_r={train['recall']:.4f}"
        )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
