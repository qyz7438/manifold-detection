"""Structural complementarity analysis of interpretable signals on real NWPU data.

Loads cached NWPU proposals (from round2150 raw-iFFT cache), recomputes the 7
non-network interpretable signals plus legacy/raw-iFFT signals on the original
images, then measures pairwise correlations and regression/classification gains
when signals are combined.  This uses real detector outputs, not synthetic masks.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.diagnose_interpretable_reward_signals as diag
from spectral_detection_posttrain.signals.fft.raw_ifft_verifier import (
    LEGACY_IFFT_FEATURE_NAMES,
    fit_train_effect_scorer,
)


OUTPUT_DIR = Path("runs/nwpu_signal_complementarity_analysis")
DEFAULT_CACHE = Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz")
DEFAULT_DATA_ROOT = Path("data/NWPU VHR-10 dataset")
DEFAULT_COCO_JSON = Path("data/NWPU_VHR10_coco.json")

RAW_IFFT_REFERENCE_FEATURES = [
    "fft_edge_truncation@64",
    "phase_edge@64",
    "phase_abs_high@11",
]


def _extract_split(data: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    keys = ["labels", "best_iou", "label_probs", "rollout_scores", "class_ids", "image_ids", "proposal_boxes"]
    return {key: np.asarray(data[f"{split}_{key}"]) for key in keys}


def _build_signal_matrices(
    data: np.lib.npyio.NpzFile,
    data_root: Path,
    coco_json: Path,
    max_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_arrays = _extract_split(data, "train")
    val_arrays = _extract_split(data, "val")
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

    # Reference raw-iFFT recipe (3 features, train effect-sum scorer)
    train_raw = diag.load_legacy_feature_matrix(data, "train", RAW_IFFT_REFERENCE_FEATURES, train_indices)
    val_raw = diag.load_legacy_feature_matrix(data, "val", RAW_IFFT_REFERENCE_FEATURES, val_indices)
    raw_scorer = fit_train_effect_scorer(train_raw, train_arrays["labels"].astype(bool), method="train_effect_sum")
    train_signals["reference_raw_ifft_recipe"] = raw_scorer.score(train_raw)
    val_signals["reference_raw_ifft_recipe"] = raw_scorer.score(val_raw)

    # Individual raw-iFFT reference features as old signals
    for spec in RAW_IFFT_REFERENCE_FEATURES:
        name = spec.replace("@", "_")
        train_signals[name] = diag.load_legacy_feature_matrix(data, "train", [spec], train_indices).ravel()
        val_signals[name] = diag.load_legacy_feature_matrix(data, "val", [spec], val_indices).ravel()

    # Final ordering: 7 new signals, then old signals
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


def _safe_impute(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isfinite(X), X, np.nan)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _recall_at_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float = 0.70) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    thresholds = np.append(thresholds, 1.0)
    valid = precision >= target_precision
    if not valid.any():
        return 0.0, 0.0
    best_idx = int(np.argmax(recall[valid]))
    return float(recall[valid][best_idx]), float(thresholds[valid][best_idx])


def _analyze_correlations(X: np.ndarray, names: list[str]) -> dict:
    rho = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                rho[i, j] = 1.0
            else:
                r, _ = spearmanr(X[:, i], X[:, j], nan_policy="omit")
                rho[i, j] = float(r) if np.isfinite(r) else 0.0
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], float(abs(rho[i, j]))))
    pairs.sort(key=lambda x: x[2])
    return {
        "spearman_matrix": {ki: {kj: float(rho[i, j]) for j, kj in enumerate(names)} for i, ki in enumerate(names)},
        "least_correlated_pairs": pairs[:15],
        "most_correlated_pairs": pairs[-15:][::-1],
    }


def _evaluate_feature_set(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_bin_train: np.ndarray,
    y_bin_val: np.ndarray,
    feature_names: list[str],
    label: str,
    seed: int = 42,
) -> dict:
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z = scaler.transform(X_val)

    reg = Ridge(alpha=1.0, random_state=seed)
    reg.fit(X_train_z, y_train)
    val_pred = reg.predict(X_val_z)
    spear_iou = float(spearmanr(y_val, val_pred, nan_policy="omit")[0])
    mse = float(np.mean((y_val - val_pred) ** 2))

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train_z, y_bin_train)
    proba = clf.predict_proba(X_val_z)[:, 1]
    auc = float(roc_auc_score(y_bin_val, proba))
    ap = float(average_precision_score(y_bin_val, proba))
    r_at_p70, threshold_p70 = _recall_at_precision(y_bin_val, proba)

    return {
        "label": label,
        "features": feature_names,
        "n_features": len(feature_names),
        "val_mse": mse,
        "val_spearman_iou": spear_iou,
        "val_auc": auc,
        "val_ap": ap,
        "val_r_at_p70": r_at_p70,
        "val_threshold_p70": threshold_p70,
    }


def _run_ablations(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_bin_train: np.ndarray,
    y_bin_val: np.ndarray,
    names: list[str],
) -> list[dict]:
    results: list[dict] = []
    for i, name in enumerate(names):
        results.append(
            _evaluate_feature_set(
                X_train[:, [i]], X_val[:, [i]], y_train, y_val, y_bin_train, y_bin_val, [name], f"single:{name}"
            )
        )

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
        results.append(
            _evaluate_feature_set(
                X_train[:, idx], X_val[:, idx], y_train, y_val, y_bin_train, y_bin_val, feat_names, label
            )
        )

    return results


def _print_summary(report: dict) -> None:
    print("\n=== Ablations (ranked by Val Spearman vs continuous IoU) ===")
    ablations = sorted(report["ablations"], key=lambda r: r["val_spearman_iou"], reverse=True)
    print(f"{'Label':<45} {'n':>3} {'MSE':>8} {'Spear':>7} {'AUC':>7} {'AP':>7} {'R@P0.7':>7}")
    print("-" * 90)
    for r in ablations:
        print(
            f"{r['label']:<45} {r['n_features']:>3} "
            f"{r['val_mse']:>8.5f} {r['val_spearman_iou']:>7.4f} "
            f"{r['val_auc']:>7.4f} {r['val_ap']:>7.4f} {r['val_r_at_p70']:>7.4f}"
        )

    print("\n=== Most complementary pairs (lowest absolute Spearman |r|) ===")
    for a, b, c in report["correlation_analysis"]["least_correlated_pairs"][:10]:
        print(f"  {a}  <->  {b}  |r|={c:.3f}")

    print("\n=== Most redundant pairs (highest absolute Spearman |r|) ===")
    for a, b, c in report["correlation_analysis"]["most_correlated_pairs"][:10]:
        print(f"  {a}  <->  {b}  |r|={c:.3f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = DEFAULT_CACHE
    data_root = DEFAULT_DATA_ROOT
    coco_json = DEFAULT_COCO_JSON
    max_size = 480
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(cache)
    train_signals, val_signals, names, train_arrays, val_arrays = _build_signal_matrices(
        data, data_root, coco_json, max_size, device
    )

    X_train = np.stack([train_signals[n] for n in names], axis=1).astype(np.float64)
    X_val = np.stack([val_signals[n] for n in names], axis=1).astype(np.float64)
    X_train = _safe_impute(X_train)
    X_val = _safe_impute(X_val)

    y_train = train_arrays["best_iou"].astype(np.float64)
    y_val = val_arrays["best_iou"].astype(np.float64)
    y_bin_train = train_arrays["labels"].astype(int)
    y_bin_val = val_arrays["labels"].astype(int)

    print("Analyzing correlations...")
    corr_analysis = _analyze_correlations(X_train, names)

    print("Running ablations...")
    ablations = _run_ablations(X_train, X_val, y_train, y_val, y_bin_train, y_bin_val, names)

    report = {
        "cache": str(cache),
        "data_root": str(data_root),
        "coco_json": str(coco_json),
        "train_count": int(train_arrays["labels"].shape[0]),
        "val_count": int(val_arrays["labels"].shape[0]),
        "train_positive": int(train_arrays["labels"].sum()),
        "val_positive": int(val_arrays["labels"].sum()),
        "signals": names,
        "correlation_analysis": corr_analysis,
        "ablations": ablations,
    }

    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    _print_summary(report)


if __name__ == "__main__":
    main()
