"""Phase 0 v2: offline screen for signals that complement detector confidence.

Instead of predicting raw IoU (where cached label_prob is already a strong
predictor), this script tests whether the 7 interpretable signals can predict:

  A) IoU rank minus confidence rank (underestimation / overestimation)
  B) IoU residual: best_iou - label_prob
  C) High-confidence false positives: conf >= 0.7, IoU < 0.5

These tasks measure whether the signals can correct detector calibration errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.diagnose_interpretable_reward_signals as diag
from spectral_detection_posttrain.signals.fft.raw_ifft_verifier import (
    fit_train_effect_scorer,
)


DEFAULT_CACHE = Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz")
DEFAULT_OUT_DIR = Path("runs/phase0_nwpu_iou_verifier_screen")
DATA_ROOT = Path("data/NWPU VHR-10 dataset")
COCO_JSON = Path("data/NWPU_VHR10_coco.json")
MAX_SIZE = 480

RAW_IFFT_REFERENCE_FEATURES = [
    "fft_edge_truncation@64",
    "phase_edge@64",
    "phase_abs_high@11",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 v2: detector confidence complementarity screen.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--coco-json", type=Path, default=COCO_JSON)
    parser.add_argument("--max-size", type=int, default=MAX_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--high-conf-threshold", type=float, default=0.7)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_impute(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isfinite(X), X, np.nan)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def recall_at_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float = 0.70) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    thresholds = np.append(thresholds, 1.0)
    valid = precision >= target_precision
    if not valid.any():
        return 0.0, 0.0
    best_idx = int(np.argmax(recall[valid]))
    return float(recall[valid][best_idx]), float(thresholds[valid][best_idx])


def build_signal_matrices(
    data: np.lib.npyio.NpzFile,
    data_root: Path,
    coco_json: Path,
    max_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_arrays = diag.extract_split_arrays(data, "train", 0)
    val_arrays = diag.extract_split_arrays(data, "val", 0)
    train_indices = np.arange(train_arrays["labels"].shape[0])
    val_indices = np.arange(val_arrays["labels"].shape[0])

    coco = diag.load_coco(coco_json)
    image_infos = {int(info["id"]): info for info in coco["images"]}
    train_image_ids = set(int(v) for v in np.unique(train_arrays["image_ids"]))
    aspect_stats = diag.compute_gt_aspect_stats(coco, train_image_ids)

    print(f"Computing image-based interpretable signals for train={len(train_indices)} val={len(val_indices)} ...")
    train_signals = diag.compute_image_signals(
        arrays=train_arrays,
        coco_infos=image_infos,
        data_root=data_root,
        max_size=max_size,
        device=device,
    )
    val_signals = diag.compute_image_signals(
        arrays=val_arrays,
        coco_infos=image_infos,
        data_root=data_root,
        max_size=max_size,
        device=device,
    )

    train_signals["aspect_ratio_plausibility"] = diag.aspect_plausibility(
        train_arrays["proposal_boxes"], train_arrays["class_ids"], aspect_stats
    )
    val_signals["aspect_ratio_plausibility"] = diag.aspect_plausibility(
        val_arrays["proposal_boxes"], val_arrays["class_ids"], aspect_stats
    )
    train_signals["nms_survivor_density"] = diag.nms_support_density(
        train_arrays["proposal_boxes"], train_arrays["class_ids"], train_arrays["image_ids"]
    )
    val_signals["nms_survivor_density"] = diag.nms_support_density(
        val_arrays["proposal_boxes"], val_arrays["class_ids"], val_arrays["image_ids"]
    )

    train_raw = diag.load_legacy_feature_matrix(data, "train", RAW_IFFT_REFERENCE_FEATURES, train_indices)
    val_raw = diag.load_legacy_feature_matrix(data, "val", RAW_IFFT_REFERENCE_FEATURES, val_indices)
    raw_scorer = fit_train_effect_scorer(train_raw, train_arrays["labels"].astype(bool), method="train_effect_sum")
    train_signals["reference_raw_ifft_recipe"] = raw_scorer.score(train_raw)
    val_signals["reference_raw_ifft_recipe"] = raw_scorer.score(val_raw)

    for spec in RAW_IFFT_REFERENCE_FEATURES:
        name = spec.replace("@", "_")
        train_signals[name] = diag.load_legacy_feature_matrix(data, "train", [spec], train_indices).ravel()
        val_signals[name] = diag.load_legacy_feature_matrix(data, "val", [spec], val_indices).ravel()

    signal_names = [
        "boundary_phase_coherence",
        "interior_exterior_texture_contrast",
        "aspect_ratio_plausibility",
        "multi_scale_saliency_consistency",
        "score_edge_alignment",
        "nms_survivor_density",
        "activation_centroid_consistency",
        "reference_raw_ifft_recipe",
        "fft_edge_truncation_64",
        "phase_edge_64",
        "phase_abs_high_11",
    ]
    return train_signals, val_signals, signal_names, train_arrays, val_arrays


def evaluate_task(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    label: str,
    task_type: str,
    task_name: str = "",
    seed: int = 42,
) -> dict[str, Any]:
    task_name = task_name or task_type
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z = scaler.transform(X_val)

    if task_type in ("regression", "rank_diff"):
        reg = Ridge(alpha=1.0, random_state=seed)
        reg.fit(X_train_z, y_train)
        val_pred = reg.predict(X_val_z)
        spear = float(spearmanr(y_val, val_pred, nan_policy="omit")[0])
        mse = float(np.mean((y_val - val_pred) ** 2))
        return {"label": label, "task": task_name, "val_mse": mse, "val_spearman": spear}

    if task_type == "classification":
        clf = LogisticRegression(C=0.25, class_weight="balanced", solver="liblinear", random_state=seed, max_iter=1000)
        clf.fit(X_train_z, y_train)
        proba = clf.predict_proba(X_val_z)[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        ap = float(average_precision_score(y_val, proba))
        r70, threshold70 = recall_at_precision(y_val, proba, target_precision=0.70)
        return {
            "label": label,
            "task": task_name,
            "val_auc": auc,
            "val_ap": ap,
            "val_r_at_p70": r70,
            "val_threshold_p70": threshold70,
        }

    raise ValueError(f"Unknown task_type: {task_type}")


def run_ablations(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_taskA_train: np.ndarray,
    y_taskA_val: np.ndarray,
    y_taskB_train: np.ndarray,
    y_taskB_val: np.ndarray,
    classification_tasks: list[tuple[str, np.ndarray, np.ndarray]],
    names: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for i, name in enumerate(names):
        results.append(evaluate_task(X_train[:, [i]], X_val[:, [i]], y_taskA_train, y_taskA_val, f"single:{name}", "rank_diff"))
        results.append(evaluate_task(X_train[:, [i]], X_val[:, [i]], y_taskB_train, y_taskB_val, f"single:{name}", "regression"))
        for task_label, y_train, y_val in classification_tasks:
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            results.append(evaluate_task(X_train[:, [i]], X_val[:, [i]], y_train, y_val, f"single:{name}", "classification", task_name=task_label))

    new7 = [
        "boundary_phase_coherence",
        "interior_exterior_texture_contrast",
        "aspect_ratio_plausibility",
        "multi_scale_saliency_consistency",
        "score_edge_alignment",
        "nms_survivor_density",
        "activation_centroid_consistency",
    ]
    new_top3 = ["boundary_phase_coherence", "interior_exterior_texture_contrast", "score_edge_alignment"]
    raw3 = ["fft_edge_truncation_64", "phase_edge_64", "phase_abs_high_11"]

    combinations = [
        ("new_top3", new_top3),
        ("all_new7", new7),
        ("raw_ifft_individual3", raw3),
        ("reference_raw_ifft_recipe", ["reference_raw_ifft_recipe"]),
        ("new_top3 + raw_ifft_recipe", [*new_top3, "reference_raw_ifft_recipe"]),
        ("new_top3 + raw_ifft_individual3", [*new_top3, *raw3]),
        ("all_new7 + raw_ifft_recipe", [*new7, "reference_raw_ifft_recipe"]),
        ("all_new7 + raw_ifft_individual3", [*new7, *raw3]),
        ("all_signals", names),
    ]

    for label, feat_names in combinations:
        idx = [names.index(n) for n in feat_names]
        results.append(evaluate_task(X_train[:, idx], X_val[:, idx], y_taskA_train, y_taskA_val, label, "rank_diff"))
        results.append(evaluate_task(X_train[:, idx], X_val[:, idx], y_taskB_train, y_taskB_val, label, "regression"))
        for task_label, y_train, y_val in classification_tasks:
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            results.append(evaluate_task(X_train[:, idx], X_val[:, idx], y_train, y_val, label, "classification", task_name=task_label))

    return results


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    data = np.load(args.cache)
    train_signals, val_signals, names, train_arrays, val_arrays = build_signal_matrices(
        data, args.data_root, args.coco_json, args.max_size, device
    )

    X_train = np.stack([train_signals[n] for n in names], axis=1).astype(np.float64)
    X_val = np.stack([val_signals[n] for n in names], axis=1).astype(np.float64)
    X_train = _safe_impute(X_train)
    X_val = _safe_impute(X_val)

    # Task A: rank difference (IoU rank - confidence rank)
    train_iou_rank = rank(train_arrays["best_iou"])
    train_conf_rank = rank(train_arrays["label_probs"])
    val_iou_rank = rank(val_arrays["best_iou"])
    val_conf_rank = rank(val_arrays["label_probs"])
    y_taskA_train = train_iou_rank - train_conf_rank
    y_taskA_val = val_iou_rank - val_conf_rank

    # Task B: residual IoU
    y_taskB_train = train_arrays["best_iou"].astype(np.float64) - train_arrays["label_probs"].astype(np.float64)
    y_taskB_val = val_arrays["best_iou"].astype(np.float64) - val_arrays["label_probs"].astype(np.float64)

    # Task C1: high-confidence false positive classification
    y_taskC1_train = ((train_arrays["label_probs"] >= args.high_conf_threshold) & (train_arrays["best_iou"] < args.iou_threshold)).astype(int)
    y_taskC1_val = ((val_arrays["label_probs"] >= args.high_conf_threshold) & (val_arrays["best_iou"] < args.iou_threshold)).astype(int)

    # Task C2: low-confidence high-IoU (rescue target)
    y_taskC2_train = ((train_arrays["label_probs"] < 0.3) & (train_arrays["best_iou"] >= args.iou_threshold)).astype(int)
    y_taskC2_val = ((val_arrays["label_probs"] < 0.3) & (val_arrays["best_iou"] >= args.iou_threshold)).astype(int)

    # Task C3: rank underestimation (IoU rank > confidence rank)
    y_taskC3_train = (train_iou_rank > train_conf_rank).astype(int)
    y_taskC3_val = (val_iou_rank > val_conf_rank).astype(int)

    print(f"Task A (rank diff): train range [{y_taskA_train.min():.0f}, {y_taskA_train.max():.0f}] "
          f"val range [{y_taskA_val.min():.0f}, {y_taskA_val.max():.0f}]")
    print(f"Task B (IoU residual): train mean={y_taskB_train.mean():.4f} std={y_taskB_train.std():.4f} "
          f"val mean={y_taskB_val.mean():.4f} std={y_taskB_val.std():.4f}")
    print(f"Task C1 (high-conf FP): train pos={y_taskC1_train.sum()}/{len(y_taskC1_train)} "
          f"val pos={y_taskC1_val.sum()}/{len(y_taskC1_val)}")
    print(f"Task C2 (low-conf high-IoU): train pos={y_taskC2_train.sum()}/{len(y_taskC2_train)} "
          f"val pos={y_taskC2_val.sum()}/{len(y_taskC2_val)}")
    print(f"Task C3 (rank underestimation): train pos={y_taskC3_train.sum()}/{len(y_taskC3_train)} "
          f"val pos={y_taskC3_val.sum()}/{len(y_taskC3_val)}")

    classification_tasks = [
        ("C1_high_conf_fp", y_taskC1_train, y_taskC1_val),
        ("C2_low_conf_high_iou", y_taskC2_train, y_taskC2_val),
        ("C3_rank_underestimation", y_taskC3_train, y_taskC3_val),
    ]

    print("Running ablations...")
    ablations = run_ablations(
        X_train, X_val,
        y_taskA_train, y_taskA_val,
        y_taskB_train, y_taskB_val,
        classification_tasks,
        names,
    )

    report = {
        "cache": str(args.cache),
        "data_root": str(args.data_root),
        "coco_json": str(args.coco_json),
        "train_count": int(train_arrays["labels"].shape[0]),
        "val_count": int(val_arrays["labels"].shape[0]),
        "signals": names,
        "tasks": {
            "task_A_rank_diff": {"train_range": [float(y_taskA_train.min()), float(y_taskA_train.max())], "val_range": [float(y_taskA_val.min()), float(y_taskA_val.max())]},
            "task_B_residual": {"train_mean": float(y_taskB_train.mean()), "train_std": float(y_taskB_train.std()), "val_mean": float(y_taskB_val.mean()), "val_std": float(y_taskB_val.std())},
            "task_C1_high_conf_fp": {"train_pos": int(y_taskC1_train.sum()), "val_pos": int(y_taskC1_val.sum())},
            "task_C2_low_conf_high_iou": {"train_pos": int(y_taskC2_train.sum()), "val_pos": int(y_taskC2_val.sum())},
            "task_C3_rank_underestimation": {"train_pos": int(y_taskC3_train.sum()), "val_pos": int(y_taskC3_val.sum())},
        },
        "ablations": ablations,
    }

    report_path = args.out_dir / "report_v2.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    csv_path = args.out_dir / "ablations_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["label", "task", "val_mse", "val_spearman", "val_auc", "val_ap", "val_r_at_p70"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in ablations:
            writer.writerow({k: r.get(k, "") for k in fields})

    print(f"\nReport: {report_path}")
    print(f"CSV: {csv_path}")

    for task_name, task_key in [("A: rank diff", "rank_diff"), ("B: IoU residual", "regression"), ("C1: high-conf FP", "C1_high_conf_fp"), ("C2: low-conf high-IoU", "C2_low_conf_high_iou"), ("C3: rank underestimation", "C3_rank_underestimation")]:
        print(f"\n=== {task_name} ===")
        task_results = [r for r in ablations if r["task"] == task_key]
        if task_key in ("regression", "rank_diff"):
            for r in sorted(task_results, key=lambda x: abs(x["val_spearman"]), reverse=True)[:10]:
                print(f"{r['label']:<45} MSE={r['val_mse']:.5f} Spear={r['val_spearman']:.4f}")
        else:
            for r in sorted(task_results, key=lambda x: x["val_r_at_p70"], reverse=True)[:10]:
                print(f"{r['label']:<45} AUC={r['val_auc']:.4f} AP={r['val_ap']:.4f} R@P0.7={r['val_r_at_p70']:.4f}")


if __name__ == "__main__":
    main()
