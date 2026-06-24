from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from large_scale_verifier_fusion_sweep import (  # noqa: E402
    build_transform,
    fit_center,
    fit_classwise,
    fit_knn,
    fit_logistic,
    fit_raw_scorer,
    fusion_scores,
    high_labels,
    high_low_mask,
    metrics_fixed_thresholds,
    score_lc_with_scorer,
)
from spectral_detection_posttrain.analysis.dimensionality import binary_ranking_metrics  # noqa: E402
from spectral_detection_posttrain.analysis.raw_ifft_verifier import (  # noqa: E402
    apply_selection_policy,
    calibrate_precision_threshold,
    threshold_metrics,
)
from spectral_detection_posttrain.utils.io import save_json  # noqa: E402


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    regime: str | None = None
    transform: str | None = None
    components: int | None = None
    scorer: str | None = None
    classwise: bool = False
    fusion: str = "raw_only"
    fusion_method: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy-level sweep for raw-iFFT/high-dimensional verifier fusion.")
    parser.add_argument(
        "--full-cache",
        default="runs/round2199_box_feature_classwise_iou_bucket_manifold/iou_bucket_box_features.npz",
    )
    parser.add_argument(
        "--raw-cache",
        default="runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz",
    )
    parser.add_argument("--run-name", default="round2201_large_scale_fusion_policy_sweep")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--low-conf-max", type=float, default=0.5)
    parser.add_argument("--high-iou-min", type=float, default=0.75)
    parser.add_argument("--low-iou-max", type=float, default=0.3)
    parser.add_argument("--classwise-min-pos", type=int, default=5)
    parser.add_argument("--classwise-min-neg", type=int, default=5)
    parser.add_argument(
        "--raw-features",
        nargs="+",
        default=["fft_edge_truncation@64", "phase_edge@64", "phase_abs_high@11"],
    )
    return parser.parse_args()


def policy_selected(
    scores: np.ndarray,
    *,
    threshold: float,
    primary_scores: np.ndarray | None,
    primary_threshold: float | None,
    image_ids: np.ndarray | None,
    top_k_per_image: int | None,
) -> np.ndarray:
    return apply_selection_policy(
        scores,
        threshold=threshold,
        primary_scores=primary_scores,
        primary_threshold=primary_threshold,
        image_ids=image_ids,
        top_k_per_image=top_k_per_image,
    )


def calibrate_policy_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float,
    margin: float,
    primary_scores: np.ndarray | None,
    primary_threshold: float | None,
    image_ids: np.ndarray | None,
    top_k_per_image: int | None,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    candidates = np.unique(scores)
    candidates = np.sort(candidates)[::-1]
    best: dict[str, Any] | None = None
    positive_count = max(1, int(labels.sum()))
    for base_threshold in candidates:
        threshold = float(base_threshold + margin)
        selected = policy_selected(
            scores,
            threshold=threshold,
            primary_scores=primary_scores,
            primary_threshold=primary_threshold,
            image_ids=image_ids,
            top_k_per_image=top_k_per_image,
        )
        metrics = threshold_metrics(selected, labels)
        if float(metrics["precision"]) < float(target_precision):
            continue
        recall = float(metrics["tp"]) / positive_count
        candidate = {
            "threshold": threshold,
            "base_threshold": float(base_threshold),
            "margin": float(margin),
            "reason": "ok",
            **metrics,
        }
        if best is None:
            best = candidate
            continue
        if (recall, int(metrics["tp"]), -int(metrics["fp"]), int(metrics["selected"])) > (
            float(best["recall"]),
            int(best["tp"]),
            -int(best["fp"]),
            int(best["selected"]),
        ):
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": float("inf"),
        "base_threshold": float("inf"),
        "margin": float(margin),
        "selected": 0,
        "tp": 0,
        "fp": 0,
        "precision": 0.0,
        "recall": 0.0,
        "false_positive_rate": 0.0,
        "reason": "no_policy_selection_reaches_target_precision",
    }


def evaluate_policy(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    *,
    final_target_precision: float,
    margin_fraction: float,
    train_image_ids: np.ndarray,
    val_image_ids: np.ndarray,
    primary_name: str,
    primary_train_scores: np.ndarray | None,
    primary_val_scores: np.ndarray | None,
    primary_target_precision: float | None,
    top_k_per_image: int | None,
) -> dict[str, Any]:
    primary_threshold = None
    primary_calibration: dict[str, Any] | None = None
    if primary_train_scores is not None and primary_target_precision is not None:
        raw_cal = calibrate_precision_threshold(
            primary_train_scores,
            train_labels,
            target_precision=float(primary_target_precision),
        )
        primary_threshold = float(raw_cal.threshold)
        primary_calibration = {
            "threshold": float(raw_cal.threshold),
            "target_precision": float(raw_cal.target_precision),
            "selected_prefix": int(raw_cal.selected_prefix),
            "tp_prefix": int(raw_cal.tp_prefix),
            "fp_prefix": int(raw_cal.fp_prefix),
            "precision_prefix": float(raw_cal.precision_prefix),
            "recall_prefix": float(raw_cal.recall_prefix),
            "reason": str(raw_cal.reason),
        }

    margin = float(np.std(train_scores) * float(margin_fraction))
    calibration = calibrate_policy_threshold(
        train_scores,
        train_labels,
        target_precision=float(final_target_precision),
        margin=margin,
        primary_scores=primary_train_scores,
        primary_threshold=primary_threshold,
        image_ids=train_image_ids,
        top_k_per_image=top_k_per_image,
    )
    train_selected = policy_selected(
        train_scores,
        threshold=float(calibration["threshold"]),
        primary_scores=primary_train_scores,
        primary_threshold=primary_threshold,
        image_ids=train_image_ids,
        top_k_per_image=top_k_per_image,
    )
    val_selected = policy_selected(
        val_scores,
        threshold=float(calibration["threshold"]),
        primary_scores=primary_val_scores,
        primary_threshold=primary_threshold,
        image_ids=val_image_ids,
        top_k_per_image=top_k_per_image,
    )
    train_metrics = threshold_metrics(train_selected, train_labels)
    val_metrics = threshold_metrics(val_selected, val_labels)
    return {
        "final_target_precision": float(final_target_precision),
        "margin_fraction": float(margin_fraction),
        "primary_name": primary_name,
        "primary_target_precision": None if primary_target_precision is None else float(primary_target_precision),
        "top_k_per_image": top_k_per_image,
        "calibration": calibration,
        "primary_calibration": primary_calibration,
        "train": train_metrics,
        "val": val_metrics,
    }


def build_regimes(
    train_ious: np.ndarray,
    val_ious: np.ndarray,
    train_classes: np.ndarray,
    val_classes: np.ndarray,
    train_probs: np.ndarray,
    val_probs: np.ndarray,
    *,
    high_iou_min: float,
    low_iou_max: float,
    low_conf_max: float,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "full_allneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=high_iou_min, low=low_iou_max, matched_only=False),
            "val_mask": high_low_mask(val_ious, val_classes, high=high_iou_min, low=low_iou_max, matched_only=False),
        },
        "full_matchedneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=high_iou_min, low=low_iou_max, matched_only=True),
            "val_mask": high_low_mask(val_ious, val_classes, high=high_iou_min, low=low_iou_max, matched_only=True),
        },
        "lowconf_allneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=high_iou_min, low=low_iou_max, matched_only=False)
            & (train_probs <= float(low_conf_max)),
            "val_mask": high_low_mask(val_ious, val_classes, high=high_iou_min, low=low_iou_max, matched_only=False)
            & (val_probs <= float(low_conf_max)),
        },
        "lowconf_matchedneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=high_iou_min, low=low_iou_max, matched_only=True)
            & (train_probs <= float(low_conf_max)),
            "val_mask": high_low_mask(val_ious, val_classes, high=high_iou_min, low=low_iou_max, matched_only=True)
            & (val_probs <= float(low_conf_max)),
        },
    }


def score_spec(
    spec: ScoreSpec,
    *,
    full: np.lib.npyio.NpzFile,
    raw: np.lib.npyio.NpzFile,
    regimes: dict[str, dict[str, np.ndarray]],
    raw_train_scores: np.ndarray,
    raw_val_scores: np.ndarray,
    train_labels: np.ndarray,
    high_iou_min: float,
    knn_k: int,
    classwise_min_pos: int,
    classwise_min_neg: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if spec.fusion == "raw_only":
        return raw_train_scores, raw_val_scores, {"source": "raw_only"}

    if spec.regime is None or spec.transform is None or spec.scorer is None:
        raise ValueError(f"Incomplete score spec: {spec}")

    train_features_all = np.asarray(full["train_features_l2"], dtype=np.float64)
    val_features_all = np.asarray(full["val_features_l2"], dtype=np.float64)
    train_ious = np.asarray(full["train_best_iou"], dtype=np.float64)
    val_ious = np.asarray(full["val_best_iou"], dtype=np.float64)
    train_classes = np.asarray(full["train_class_id"], dtype=np.int64)
    val_classes = np.asarray(full["val_class_id"], dtype=np.int64)

    regime = regimes[spec.regime]
    train_mask = np.asarray(regime["train_mask"], dtype=bool)
    val_mask = np.asarray(regime["val_mask"], dtype=bool)
    regime_train_labels = high_labels(train_ious[train_mask], high=high_iou_min)
    regime_val_labels = high_labels(val_ious[val_mask], high=high_iou_min)
    train_features = train_features_all[train_mask]
    val_features = val_features_all[val_mask]
    train_classes_regime = train_classes[train_mask]
    val_classes_regime = val_classes[val_mask]

    transform = build_transform("l2" if spec.transform == "l2" else "l2_z", train_features, spec.components)
    train_x = transform.apply(train_features)
    val_x = transform.apply(val_features)
    fitters = {
        "center": fit_center,
        "knn": lambda x, y: fit_knn(x, y, k=int(knn_k)),
        "logistic": fit_logistic,
    }
    fitter = fitters[spec.scorer]
    global_scorer = fitter(train_x, regime_train_labels)
    diagnostics: dict[str, Any] = {
        "regime": spec.regime,
        "transform": transform.name,
        "scorer": spec.scorer,
        "classwise": bool(spec.classwise),
        "regime_train_count": int(regime_train_labels.shape[0]),
        "regime_train_positive": int(regime_train_labels.sum()),
        "regime_val_count": int(regime_val_labels.shape[0]),
        "regime_val_positive": int(regime_val_labels.sum()),
    }

    scorer = global_scorer
    if spec.classwise:
        if spec.scorer not in {"center", "knn"}:
            raise ValueError(f"Classwise scorer does not support {spec.scorer}")
        scorer, class_diagnostics = fit_classwise(
            fitter,
            global_scorer,
            train_x,
            regime_train_labels,
            train_classes_regime,
            min_pos=int(classwise_min_pos),
            min_neg=int(classwise_min_neg),
        )
        diagnostics["classwise_diagnostics"] = class_diagnostics

    regime_train_scores = scorer.score(train_x, train_classes_regime)
    regime_val_scores = scorer.score(val_x, val_classes_regime)
    diagnostics["regime_ranking"] = binary_ranking_metrics(regime_val_scores, regime_val_labels)
    lc_train_hd, lc_val_hd = score_lc_with_scorer(transform, scorer, raw)
    diagnostics["lc_hd_ranking"] = binary_ranking_metrics(lc_val_hd, np.asarray(raw["val_labels"], dtype=bool))

    if spec.fusion == "hd_only":
        return lc_train_hd, lc_val_hd, diagnostics
    if spec.fusion == "raw_hd":
        train_scores, val_scores = fusion_scores(
            [raw_train_scores, lc_train_hd],
            [raw_val_scores, lc_val_hd],
            train_labels,
            method=str(spec.fusion_method),
        )
        return train_scores, val_scores, diagnostics
    if spec.fusion == "raw_hd_prob":
        train_scores, val_scores = fusion_scores(
            [raw_train_scores, lc_train_hd, np.asarray(raw["train_label_probs"], dtype=np.float64)],
            [raw_val_scores, lc_val_hd, np.asarray(raw["val_label_probs"], dtype=np.float64)],
            train_labels,
            method=str(spec.fusion_method),
        )
        return train_scores, val_scores, diagnostics
    raise ValueError(f"Unknown fusion mode: {spec.fusion}")


def default_specs() -> list[ScoreSpec]:
    return [
        ScoreSpec(name="raw_only"),
        ScoreSpec(
            name="fusion_raw_hd_prob/full_matchedneg/l2_z_pca128/center/train_effect",
            regime="full_matchedneg",
            transform="l2_z",
            components=128,
            scorer="center",
            fusion="raw_hd_prob",
            fusion_method="train_effect",
        ),
        ScoreSpec(
            name="fusion_raw_hd_prob/full_allneg/l2_z_pca96/center/train_effect",
            regime="full_allneg",
            transform="l2_z",
            components=96,
            scorer="center",
            fusion="raw_hd_prob",
            fusion_method="train_effect",
        ),
        ScoreSpec(
            name="fusion_raw_hd_prob/full_allneg/l2_z_pca128/center/train_effect",
            regime="full_allneg",
            transform="l2_z",
            components=128,
            scorer="center",
            fusion="raw_hd_prob",
            fusion_method="train_effect",
        ),
        ScoreSpec(
            name="fusion_raw_hd_prob/full_allneg/l2_z_pca32/logistic/logistic",
            regime="full_allneg",
            transform="l2_z",
            components=32,
            scorer="logistic",
            fusion="raw_hd_prob",
            fusion_method="logistic",
        ),
        ScoreSpec(
            name="fusion_raw_hd_prob/lowconf_allneg/l2_z_pca96/logistic/train_effect",
            regime="lowconf_allneg",
            transform="l2_z",
            components=96,
            scorer="logistic",
            fusion="raw_hd_prob",
            fusion_method="train_effect",
        ),
        ScoreSpec(
            name="fusion_raw_hd/full_allneg/l2/classwise_center/train_effect",
            regime="full_allneg",
            transform="l2",
            scorer="center",
            classwise=True,
            fusion="raw_hd",
            fusion_method="train_effect",
        ),
        ScoreSpec(
            name="fusion_raw_hd/full_allneg/l2/classwise_center/logistic",
            regime="full_allneg",
            transform="l2",
            scorer="center",
            classwise=True,
            fusion="raw_hd",
            fusion_method="logistic",
        ),
        ScoreSpec(
            name="fusion_raw_hd/full_allneg/l2/classwise_knn/train_effect",
            regime="full_allneg",
            transform="l2",
            scorer="knn",
            classwise=True,
            fusion="raw_hd",
            fusion_method="train_effect",
        ),
        ScoreSpec(
            name="fusion_raw_hd_prob/full_allneg/l2/classwise_knn/train_effect",
            regime="full_allneg",
            transform="l2",
            scorer="knn",
            classwise=True,
            fusion="raw_hd_prob",
            fusion_method="train_effect",
        ),
        ScoreSpec(
            name="fusion_raw_hd_prob/full_allneg/l2_z/classwise_knn/train_effect",
            regime="full_allneg",
            transform="l2_z",
            scorer="knn",
            classwise=True,
            fusion="raw_hd_prob",
            fusion_method="train_effect",
        ),
    ]


def compact_policy_row(score_name: str, policy: dict[str, Any], ranking: dict[str, Any]) -> dict[str, Any]:
    val = policy["val"]
    train = policy["train"]
    return {
        "score_name": score_name,
        "final_target_precision": policy["final_target_precision"],
        "primary_name": policy["primary_name"],
        "primary_target_precision": policy["primary_target_precision"],
        "top_k_per_image": policy["top_k_per_image"],
        "margin_fraction": policy["margin_fraction"],
        "val_precision": float(val["precision"]),
        "val_recall": float(val["recall"]),
        "val_selected": int(val["selected"]),
        "val_tp": int(val["tp"]),
        "val_fp": int(val["fp"]),
        "train_precision": float(train["precision"]),
        "train_recall": float(train["recall"]),
        "train_selected": int(train["selected"]),
        "train_tp": int(train["tp"]),
        "train_fp": int(train["fp"]),
        "auc": float(ranking["auc"]),
        "average_precision": float(ranking["average_precision"]),
        "rank_recall_at_precision_0.7": float(ranking["recall_at_precision_0.7"]),
        "rank_recall_at_precision_0.8": float(ranking["recall_at_precision_0.8"]),
        "rank_recall_at_precision_0.9": float(ranking["recall_at_precision_0.9"]),
        "threshold": float(policy["calibration"]["threshold"]),
        "calibration_reason": str(policy["calibration"]["reason"]),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), run_dir / "config.json")

    full = np.load(args.full_cache)
    raw = np.load(args.raw_cache)

    train_labels = np.asarray(raw["train_labels"], dtype=bool)
    val_labels = np.asarray(raw["val_labels"], dtype=bool)
    train_image_ids = np.asarray(raw["train_image_ids"])
    val_image_ids = np.asarray(raw["val_image_ids"])
    raw_train_scores, raw_val_scores, raw_scorer = fit_raw_scorer(raw, list(args.raw_features))

    regimes = build_regimes(
        np.asarray(full["train_best_iou"], dtype=np.float64),
        np.asarray(full["val_best_iou"], dtype=np.float64),
        np.asarray(full["train_class_id"], dtype=np.int64),
        np.asarray(full["val_class_id"], dtype=np.int64),
        np.asarray(full["train_matched_prob"], dtype=np.float64),
        np.asarray(full["val_matched_prob"], dtype=np.float64),
        high_iou_min=float(args.high_iou_min),
        low_iou_max=float(args.low_iou_max),
        low_conf_max=float(args.low_conf_max),
    )

    report: dict[str, Any] = {
        "full_cache": str(args.full_cache),
        "raw_cache": str(args.raw_cache),
        "raw_features": list(args.raw_features),
        "raw_scorer": {
            "weights": raw_scorer.weights.astype(float).tolist(),
            "mean": raw_scorer.scaler.mean_.astype(float).tolist(),
            "scale": raw_scorer.scaler.scale_.astype(float).tolist(),
        },
        "candidate_counts": {
            "train": int(train_labels.shape[0]),
            "train_positive": int(train_labels.sum()),
            "val": int(val_labels.shape[0]),
            "val_positive": int(val_labels.sum()),
        },
        "score_specs": {},
        "policies": [],
        "leaderboards": {},
    }

    final_targets = [0.7, 0.8, 0.9]
    primary_targets: list[float | None] = [None, 0.8, 0.9]
    top_k_values: list[int | None] = [None, 1, 2, 3]
    margin_fractions = [0.0, 0.05, 0.1, 0.2]

    rows: list[dict[str, Any]] = []
    for spec in default_specs():
        train_scores, val_scores, diagnostics = score_spec(
            spec,
            full=full,
            raw=raw,
            regimes=regimes,
            raw_train_scores=raw_train_scores,
            raw_val_scores=raw_val_scores,
            train_labels=train_labels,
            high_iou_min=float(args.high_iou_min),
            knn_k=int(args.knn_k),
            classwise_min_pos=int(args.classwise_min_pos),
            classwise_min_neg=int(args.classwise_min_neg),
        )
        ranking = binary_ranking_metrics(val_scores, val_labels)
        fixed = metrics_fixed_thresholds(train_scores, train_labels, val_scores, val_labels)
        report["score_specs"][spec.name] = {
            "diagnostics": diagnostics,
            "ranking": ranking,
            "fixed_thresholds": fixed["fixed_thresholds"],
        }

        for final_target in final_targets:
            for primary_target in primary_targets:
                for top_k in top_k_values:
                    for margin_fraction in margin_fractions:
                        policy = evaluate_policy(
                            train_scores,
                            val_scores,
                            train_labels,
                            val_labels,
                            final_target_precision=final_target,
                            margin_fraction=margin_fraction,
                            train_image_ids=train_image_ids,
                            val_image_ids=val_image_ids,
                            primary_name="none" if primary_target is None else "raw_iFFT",
                            primary_train_scores=None if primary_target is None else raw_train_scores,
                            primary_val_scores=None if primary_target is None else raw_val_scores,
                            primary_target_precision=primary_target,
                            top_k_per_image=top_k,
                        )
                        row = compact_policy_row(spec.name, policy, ranking)
                        rows.append(row)
                        report["policies"].append({  # type: ignore[union-attr]
                            "score_name": spec.name,
                            "policy": policy,
                            "ranking": ranking,
                        })

    report["leaderboards"] = {
        "val_precision_ge_0.7": sorted(
            [row for row in rows if row["val_precision"] >= 0.7],
            key=lambda row: (row["val_recall"], row["val_precision"], row["average_precision"], -row["val_fp"]),
            reverse=True,
        )[:100],
        "val_precision_ge_0.8": sorted(
            [row for row in rows if row["val_precision"] >= 0.8],
            key=lambda row: (row["val_recall"], row["val_precision"], row["average_precision"], -row["val_fp"]),
            reverse=True,
        )[:100],
        "val_precision_ge_0.9": sorted(
            [row for row in rows if row["val_precision"] >= 0.9],
            key=lambda row: (row["val_recall"], row["val_precision"], row["average_precision"], -row["val_fp"]),
            reverse=True,
        )[:100],
        "all_by_val_recall": sorted(
            rows,
            key=lambda row: (row["val_recall"], row["val_precision"], row["average_precision"], -row["val_fp"]),
            reverse=True,
        )[:100],
    }
    save_json(report, run_dir / "large_scale_fusion_policy_report.json")
    print(json.dumps(report["leaderboards"], ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
