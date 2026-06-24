from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_CACHE = Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz")

RAW_IFFT_FEATURE_NAMES = [
    "raw_edge",
    "low_edge",
    "high_edge",
    "phase_edge",
    "high_minus_low_edge",
    "phase_abs_diff",
    "high_abs_diff",
    "low_abs_diff",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "high_low_energy_ratio",
]

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

CLS_SUMMARY_NAMES = [
    "bg_prob",
    "top1_object_prob",
    "top2_object_prob",
    "top1_minus_top2",
    "cls_entropy",
]

CATEGORY_NAMES = {
    1: "airplane",
    2: "ship",
    3: "storage_tank",
    4: "baseball_diamond",
    5: "tennis_court",
    6: "basketball_court",
    7: "ground_track_field",
    8: "harbor",
    9: "bridge",
    10: "vehicle",
}

SCENE_GROUPS = {
    "sports": {4, 5, 6, 7},
    "maritime": {2, 8},
    "vehicle": {10},
    "compact_round": {3},
    "airplane": {1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose scalar raw/iFFT verifier signals for LC-HI rescue.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/round2205_scalar_signal_diagnostics"))
    parser.add_argument("--crops", type=int, nargs="+", default=[7, 11, 15, 21, 64])
    parser.add_argument("--target-precision", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.shape[0], dtype=np.float64)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(rankdata_simple(np.asarray(x, dtype=np.float64)), rankdata_simple(np.asarray(y, dtype=np.float64)))


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, *, precision_targets: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    output: dict[str, Any] = {
        "count": int(labels.shape[0]),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "prevalence": float(positive_count / max(1, labels.shape[0])),
        "auc": 0.0,
        "average_precision": 0.0,
    }
    if positive_count > 0 and negative_count > 0 and np.unique(scores).shape[0] > 1:
        output["auc"] = float(roc_auc_score(labels.astype(np.int32), scores))
        output["average_precision"] = float(average_precision_score(labels.astype(np.int32), scores))

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels.astype(np.int64))
    rank = np.arange(1, sorted_labels.shape[0] + 1)
    precision = tp / rank
    recall = tp / max(1, positive_count)
    for target in precision_targets:
        valid = np.flatnonzero(precision >= float(target))
        key = f"recall_at_precision_{target:g}"
        if valid.shape[0] == 0:
            output[key] = 0.0
            output[f"{key}_selected"] = 0
            output[f"{key}_tp"] = 0
            output[f"{key}_fp"] = 0
            output[f"{key}_precision"] = 0.0
            output[f"{key}_threshold"] = None
            continue
        best_recall = recall[valid].max()
        best_candidates = valid[recall[valid] == best_recall]
        best = int(best_candidates[-1])
        output[key] = float(recall[best])
        output[f"{key}_selected"] = int(rank[best])
        output[f"{key}_tp"] = int(tp[best])
        output[f"{key}_fp"] = int(rank[best] - tp[best])
        output[f"{key}_precision"] = float(precision[best])
        output[f"{key}_threshold"] = float(sorted_scores[best])
    return output


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray, *, target_precision: float) -> dict[str, Any]:
    metrics = ranking_metrics(scores, labels, precision_targets=(float(target_precision),))
    key = f"recall_at_precision_{target_precision:g}"
    return {
        "threshold": metrics.get(f"{key}_threshold"),
        "train_recall": metrics.get(key, 0.0),
        "train_precision": metrics.get(f"{key}_precision", 0.0),
        "train_selected": metrics.get(f"{key}_selected", 0),
        "train_tp": metrics.get(f"{key}_tp", 0),
        "train_fp": metrics.get(f"{key}_fp", 0),
    }


def apply_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float | None) -> dict[str, Any]:
    if threshold is None:
        return {"selected": 0, "tp": 0, "fp": 0, "precision": 0.0, "recall": 0.0}
    selected = scores >= float(threshold)
    tp = int((selected & labels).sum())
    fp = int((selected & (~labels)).sum())
    total = int(selected.sum())
    positives = int(labels.sum())
    return {
        "selected": total,
        "tp": tp,
        "fp": fp,
        "precision": float(tp / max(1, total)),
        "recall": float(tp / max(1, positives)),
    }


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train)
    std = np.nanstd(train)
    if not math.isfinite(float(std)) or float(std) < 1e-12:
        std = 1.0
    return (train - mean) / std, (val - mean) / std


def effect_stats(values: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    pos = values[labels]
    neg = values[~labels]
    if pos.size == 0 or neg.size == 0:
        return {"pos_mean": None, "neg_mean": None, "delta": None, "cohen_d": None}
    delta = float(np.nanmean(pos) - np.nanmean(neg))
    pooled = math.sqrt((float(np.nanvar(pos)) + float(np.nanvar(neg))) / 2.0)
    cohen = delta / pooled if pooled > 1e-12 else 0.0
    return {
        "pos_mean": finite_float(np.nanmean(pos)),
        "neg_mean": finite_float(np.nanmean(neg)),
        "delta": finite_float(delta),
        "cohen_d": finite_float(cohen),
    }


def role_for_row(row: dict[str, Any], *, target_precision: float) -> str:
    fixed = row["fixed_threshold_val"]
    val = row["val_metrics"]
    sign_stable = bool(row["direction_stable"])
    prob_corr = abs(float(row.get("spearman_label_prob", 0.0) or 0.0))
    fixed_precision = float(fixed["precision"])
    fixed_recall = float(fixed["recall"])
    val_ap = float(val["average_precision"])
    val_auc = float(val["auc"])
    selected = int(fixed["selected"])
    if prob_corr >= 0.75 and val_ap >= 0.12:
        return "baseline_confidence_proxy"
    if sign_stable and fixed_precision >= target_precision and fixed_recall >= 0.10 and selected >= 3:
        return "gate_candidate"
    if sign_stable and fixed_precision >= 0.50 and selected >= 3:
        return "weak_gate_candidate"
    if sign_stable and (val_ap >= 0.14 or val_auc >= 0.65):
        return "ranker_or_sample_weight"
    if sign_stable and (val_ap >= 0.09 or val_auc >= 0.58):
        return "diagnostic_only"
    if not sign_stable:
        return "unstable_direction"
    return "weak_or_noise"


def collect_features(data: np.lib.npyio.NpzFile, crops: list[int]) -> list[tuple[str, str, str, np.ndarray, np.ndarray]]:
    specs: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    for idx, name in enumerate(RAW_IFFT_FEATURE_NAMES):
        specs.append(("raw_ifft", name, name, data["train_raw_ifft"][:, idx], data["val_raw_ifft"][:, idx]))
    for crop in crops:
        train_key = f"train_legacy_ifft_{int(crop)}"
        val_key = f"val_legacy_ifft_{int(crop)}"
        if train_key not in data.files or val_key not in data.files:
            continue
        for idx, name in enumerate(LEGACY_FEATURE_NAMES):
            specs.append(
                (
                    f"legacy_ifft_{int(crop)}",
                    f"{name}@{int(crop)}",
                    name,
                    data[train_key][:, idx],
                    data[val_key][:, idx],
                )
            )
    for idx, name in enumerate(CLS_SUMMARY_NAMES):
        specs.append(("cls_summary", name, name, data["train_cls_summary"][:, idx], data["val_cls_summary"][:, idx]))

    specs.extend(
        [
            ("baseline_proxy", "matched_label_prob", "matched_label_prob", data["train_label_probs"], data["val_label_probs"]),
            ("baseline_proxy", "rollout_score", "rollout_score", data["train_rollout_scores"], data["val_rollout_scores"]),
        ]
    )
    return specs


def group_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    *,
    min_pos: int = 2,
    min_neg: int = 20,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group_name, group_classes in SCENE_GROUPS.items():
        mask = np.isin(class_ids, list(group_classes))
        pos = int((mask & labels).sum())
        neg = int((mask & (~labels)).sum())
        if pos < min_pos or neg < min_neg:
            output[group_name] = {"positive_count": pos, "negative_count": neg, "auc": None, "average_precision": None}
            continue
        metrics = ranking_metrics(scores[mask], labels[mask], precision_targets=(0.7,))
        output[group_name] = {
            "positive_count": pos,
            "negative_count": neg,
            "auc": finite_float(metrics["auc"]),
            "average_precision": finite_float(metrics["average_precision"]),
            "recall_at_precision_0.7": finite_float(metrics["recall_at_precision_0.7"]),
        }
    return output


def per_class_delta(
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    *,
    min_pos: int = 2,
    min_neg: int = 10,
) -> list[dict[str, Any]]:
    rows = []
    for class_id in sorted(int(c) for c in np.unique(class_ids)):
        mask = class_ids == class_id
        pos = int((mask & labels).sum())
        neg = int((mask & (~labels)).sum())
        if pos < min_pos or neg < min_neg:
            continue
        stats = effect_stats(scores[mask], labels[mask])
        metrics = ranking_metrics(scores[mask], labels[mask], precision_targets=(0.7,))
        rows.append(
            {
                "class_id": class_id,
                "class_name": CATEGORY_NAMES.get(class_id, str(class_id)),
                "positive_count": pos,
                "negative_count": neg,
                "delta": stats["delta"],
                "cohen_d": stats["cohen_d"],
                "auc": finite_float(metrics["auc"]),
                "average_precision": finite_float(metrics["average_precision"]),
                "recall_at_precision_0.7": finite_float(metrics["recall_at_precision_0.7"]),
            }
        )
    return rows


def summarize_feature_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_name"])].append(row)
    summary = []
    for base_name, items in grouped.items():
        best = max(items, key=lambda item: (float(item["val_metrics"]["average_precision"]), float(item["val_metrics"]["auc"])))
        stable_count = sum(1 for item in items if item["direction_stable"])
        roles = Counter(str(item["role"]) for item in items)
        summary.append(
            {
                "base_name": base_name,
                "variant_count": len(items),
                "stable_count": int(stable_count),
                "best_signal": best["signal"],
                "best_space": best["space"],
                "best_val_auc": finite_float(best["val_metrics"]["auc"]),
                "best_val_ap": finite_float(best["val_metrics"]["average_precision"]),
                "best_fixed_precision": finite_float(best["fixed_threshold_val"]["precision"]),
                "best_fixed_recall": finite_float(best["fixed_threshold_val"]["recall"]),
                "roles": dict(roles),
            }
        )
    return sorted(summary, key=lambda row: (row["best_val_ap"] or 0.0, row["best_val_auc"] or 0.0), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]], *, flatten: bool = True) -> None:
    if not rows:
        return
    if flatten:
        flat_rows = []
        for row in rows:
            flat: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            continue
                        flat[f"{key}.{subkey}"] = subvalue
                elif isinstance(value, list):
                    continue
                else:
                    flat[key] = value
            flat_rows.append(flat)
    else:
        flat_rows = rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.cache)
    train_labels = data["train_labels"].astype(bool)
    val_labels = data["val_labels"].astype(bool)
    train_prob = data["train_label_probs"].astype(np.float64)
    val_prob = data["val_label_probs"].astype(np.float64)
    train_iou = data["train_best_iou"].astype(np.float64)
    val_iou = data["val_best_iou"].astype(np.float64)
    val_class = data["val_class_ids"].astype(np.int64)

    rows = []
    for space, signal, base_name, train_values, val_values in collect_features(data, list(args.crops)):
        train_values = np.asarray(train_values, dtype=np.float64)
        val_values = np.asarray(val_values, dtype=np.float64)
        train_z, val_z = standardize(train_values, val_values)
        train_stats = effect_stats(train_z, train_labels)
        val_stats = effect_stats(val_z, val_labels)
        train_delta = float(train_stats["delta"] or 0.0)
        direction = 1.0 if train_delta >= 0 else -1.0
        train_score = direction * train_z
        val_score = direction * val_z
        train_metrics = ranking_metrics(train_score, train_labels)
        val_metrics = ranking_metrics(val_score, val_labels)
        calibration = calibrate_threshold(train_score, train_labels, target_precision=float(args.target_precision))
        fixed_val = apply_threshold(val_score, val_labels, calibration["threshold"])
        val_oracle_metrics = ranking_metrics(-val_score, val_labels)
        row = {
            "space": space,
            "signal": signal,
            "base_name": base_name,
            "direction": int(direction),
            "direction_stable": bool((float(val_stats["delta"] or 0.0) == 0.0) or (train_delta == 0.0) or (np.sign(train_delta) == np.sign(float(val_stats["delta"] or 0.0)))),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "val_reverse_direction_metrics": {
                "auc": finite_float(val_oracle_metrics["auc"]),
                "average_precision": finite_float(val_oracle_metrics["average_precision"]),
            },
            "train_calibration": calibration,
            "fixed_threshold_val": fixed_val,
            "spearman_label_prob": spearman(val_score, val_prob),
            "spearman_best_iou": spearman(val_score, val_iou),
            "group_metrics": group_metrics(val_score, val_labels, val_class),
            "per_class": per_class_delta(val_score, val_labels, val_class),
        }
        row["role"] = role_for_row(row, target_precision=float(args.target_precision))
        rows.append(row)

    top_by_val_ap = sorted(rows, key=lambda row: (float(row["val_metrics"]["average_precision"]), float(row["val_metrics"]["auc"])), reverse=True)[
        : int(args.top_k)
    ]
    top_by_fixed = sorted(
        rows,
        key=lambda row: (
            float(row["fixed_threshold_val"]["precision"]),
            float(row["fixed_threshold_val"]["recall"]),
            int(row["fixed_threshold_val"]["tp"]),
            float(row["val_metrics"]["average_precision"]),
        ),
        reverse=True,
    )[: int(args.top_k)]
    top_non_proxy = [
        row
        for row in top_by_val_ap
        if row["space"] not in {"baseline_proxy", "cls_summary"} and abs(float(row.get("spearman_label_prob") or 0.0)) < 0.7
    ][: int(args.top_k)]

    group_top: dict[str, list[dict[str, Any]]] = {}
    for group_name in SCENE_GROUPS:
        candidates = [
            row
            for row in rows
            if row["group_metrics"].get(group_name, {}).get("average_precision") is not None
        ]
        group_top[group_name] = sorted(
            candidates,
            key=lambda row: (
                float(row["group_metrics"][group_name].get("average_precision") or 0.0),
                float(row["group_metrics"][group_name].get("auc") or 0.0),
            ),
            reverse=True,
        )[:15]

    report = {
        "config": {
            "cache": str(args.cache),
            "crops": [int(c) for c in args.crops],
            "target_precision": float(args.target_precision),
        },
        "counts": {
            "train_candidates": int(train_labels.shape[0]),
            "train_positive": int(train_labels.sum()),
            "train_negative": int((~train_labels).sum()),
            "val_candidates": int(val_labels.shape[0]),
            "val_positive": int(val_labels.sum()),
            "val_negative": int((~val_labels).sum()),
        },
        "role_counts": dict(Counter(str(row["role"]) for row in rows)),
        "feature_family_summary": summarize_feature_family(rows),
        "top_by_val_ap": top_by_val_ap,
        "top_by_fixed_threshold": top_by_fixed,
        "top_non_proxy_by_val_ap": top_non_proxy,
        "group_top": group_top,
        "all_signals": rows,
    }

    report_path = args.out_dir / "scalar_signal_diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_dir / "all_scalar_signals.csv", rows)
    write_csv(args.out_dir / "top_by_val_ap.csv", top_by_val_ap)
    write_csv(args.out_dir / "top_by_fixed_threshold.csv", top_by_fixed)
    write_csv(args.out_dir / "feature_family_summary.csv", report["feature_family_summary"], flatten=False)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "counts": report["counts"],
                "role_counts": report["role_counts"],
                "top_by_val_ap": [
                    {
                        "signal": row["signal"],
                        "space": row["space"],
                        "role": row["role"],
                        "val_auc": row["val_metrics"]["auc"],
                        "val_ap": row["val_metrics"]["average_precision"],
                        "fixed_val": row["fixed_threshold_val"],
                        "spearman_label_prob": row["spearman_label_prob"],
                    }
                    for row in top_by_val_ap[:10]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
