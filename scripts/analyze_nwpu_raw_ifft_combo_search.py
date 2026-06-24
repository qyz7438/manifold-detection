from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


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
    parser = argparse.ArgumentParser(description="Search raw iFFT legacy feature combinations for AP75 LC-HI rescue.")
    parser.add_argument(
        "--cache",
        default="runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz",
    )
    parser.add_argument("--run-name", default="round2151_raw_ifft_combo_search_ap75")
    parser.add_argument("--crops", type=int, nargs="+", default=[7, 11, 15, 21, 64])
    parser.add_argument("--top-n-per-crop", type=int, default=14)
    parser.add_argument("--top-n-all-crop", type=int, default=24)
    parser.add_argument("--max-combo-size", type=int, default=6)
    parser.add_argument("--fusion-top-combos", type=int, default=40)
    return parser.parse_args()


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    output: dict[str, float | int] = {
        "count": int(labels.shape[0]),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "auc": 0.0,
        "average_precision": 0.0,
    }
    if positive_count > 0 and negative_count > 0 and np.unique(scores).shape[0] > 1:
        output["auc"] = float(roc_auc_score(labels.astype(np.int32), scores))
        output["average_precision"] = float(average_precision_score(labels.astype(np.int32), scores))

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels.astype(np.int64))
    rank = np.arange(1, sorted_labels.shape[0] + 1)
    precision = tp / rank
    recall = tp / max(1, positive_count)
    for target in (0.7, 0.8, 0.9):
        valid = np.flatnonzero(precision >= target)
        key = f"recall_at_precision_{target:g}"
        if valid.shape[0] == 0:
            output[key] = 0.0
            output[f"{key}_selected"] = 0
            output[f"{key}_tp"] = 0
            output[f"{key}_fp"] = 0
            output[f"{key}_precision"] = 0.0
            output[f"{key}_threshold"] = 0.0
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


def sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            row["metrics"].get("recall_at_precision_0.7", 0.0),  # type: ignore[index,union-attr]
            row["metrics"].get("recall_at_precision_0.8", 0.0),  # type: ignore[index,union-attr]
            row["metrics"].get("average_precision", 0.0),  # type: ignore[index,union-attr]
            row["metrics"].get("auc", 0.0),  # type: ignore[index,union-attr]
        ),
        reverse=True,
    )


def zscore_pair(train_features: np.ndarray, val_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_features)
    return scaler.transform(train_features), scaler.transform(val_features)


def train_effect_weights(train_z: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    positive = train_z[train_labels].mean(axis=0)
    negative = train_z[~train_labels].mean(axis=0)
    return positive - negative


def center_margin_scores(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    val_z: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    train_subset = train_z[:, columns]
    val_subset = val_z[:, columns]
    positive_center = train_subset[train_labels].mean(axis=0, keepdims=True)
    negative_center = train_subset[~train_labels].mean(axis=0, keepdims=True)
    positive_distance = np.linalg.norm(val_subset - positive_center, axis=1)
    negative_distance = np.linalg.norm(val_subset - negative_center, axis=1)
    return negative_distance - positive_distance


def add_row(
    rows: list[dict[str, object]],
    *,
    space: str,
    method: str,
    feature_names: list[str],
    columns: tuple[int, ...],
    scores: np.ndarray,
    val_labels: np.ndarray,
) -> None:
    rows.append(
        {
            "space": space,
            "method": method,
            "feature_count": len(columns),
            "features": [feature_names[index] for index in columns],
            "metrics": ranking_metrics(scores, val_labels),
        }
    )


def univariate_rows(
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, object]]:
    train_z, val_z = zscore_pair(train_features, val_features)
    weights = train_effect_weights(train_z, train_labels)
    rows = []
    for index, name in enumerate(feature_names):
        scores = val_z[:, index] * weights[index]
        rows.append(
            {
                "index": index,
                "feature": name,
                "train_effect_weight": float(weights[index]),
                "metrics": ranking_metrics(scores, val_labels),
            }
        )
    return sort_rows(rows)


def search_small_combinations(
    *,
    space: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    feature_names: list[str],
    candidate_indices: list[int],
    max_combo_size: int,
) -> list[dict[str, object]]:
    train_z, val_z = zscore_pair(train_features, val_features)
    weights = train_effect_weights(train_z, train_labels)
    rows: list[dict[str, object]] = []
    max_k = min(max_combo_size, len(candidate_indices))
    for size in range(1, max_k + 1):
        for combo in itertools.combinations(candidate_indices, size):
            columns = np.asarray(combo, dtype=np.int64)
            add_row(
                rows,
                space=space,
                method="train_effect_sum",
                feature_names=feature_names,
                columns=combo,
                scores=val_z[:, columns] @ weights[columns],
                val_labels=val_labels,
            )
            add_row(
                rows,
                space=space,
                method="center_margin",
                feature_names=feature_names,
                columns=combo,
                scores=center_margin_scores(train_z, train_labels, val_z, columns),
                val_labels=val_labels,
            )
    return sort_rows(rows)


def logreg_rows(
    *,
    space: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    feature_names: list[str],
    candidate_indices: list[int],
    max_combo_size: int,
) -> list[dict[str, object]]:
    train_z, val_z = zscore_pair(train_features, val_features)
    rows: list[dict[str, object]] = []
    max_k = min(max_combo_size, len(candidate_indices))
    for size in range(1, max_k + 1):
        for combo in itertools.combinations(candidate_indices, size):
            columns = np.asarray(combo, dtype=np.int64)
            for regularization in (0.03, 0.1, 0.3, 1.0):
                model = LogisticRegression(
                    C=regularization,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                )
                model.fit(train_z[:, columns], train_labels.astype(np.int32))
                scores = model.decision_function(val_z[:, columns])
                rows.append(
                    {
                        "space": space,
                        "method": f"logreg_balanced_C{regularization:g}",
                        "feature_count": len(combo),
                        "features": [feature_names[index] for index in combo],
                        "metrics": ranking_metrics(scores, val_labels),
                    }
                )
    return sort_rows(rows)


def fit_roi35(train_roi: np.ndarray, val_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    train_scaled, val_scaled = zscore_pair(train_roi, val_roi)
    components = min(35, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=components, whiten=True, random_state=42).fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(val_scaled), float(pca.explained_variance_ratio_.sum())


def materialize_feature_set(
    data: np.lib.npyio.NpzFile,
    feature_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_columns = []
    val_columns = []
    for label in feature_labels:
        feature_name, crop_part = label.split("@")
        crop = int(crop_part)
        index = LEGACY_FEATURE_NAMES.index(feature_name)
        train_columns.append(data[f"train_legacy_ifft_{crop}"][:, index])
        val_columns.append(data[f"val_legacy_ifft_{crop}"][:, index])
    return np.stack(train_columns, axis=1), np.stack(val_columns, axis=1)


def evaluate_fusions(
    *,
    data: np.lib.npyio.NpzFile,
    combo_rows: list[dict[str, object]],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    max_rows: int,
) -> tuple[list[dict[str, object]], float]:
    cls_train, cls_val = zscore_pair(data["train_cls_summary"], data["val_cls_summary"])
    roi_train, roi_val, roi_variance = fit_roi35(data["train_roi_l2"], data["val_roi_l2"])
    results: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    unique_rows = []
    for row in combo_rows:
        key = tuple(row["features"])  # type: ignore[arg-type]
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
        if len(unique_rows) >= max_rows:
            break

    for row in unique_rows:
        legacy_labels = list(row["features"])  # type: ignore[arg-type]
        legacy_train, legacy_val = materialize_feature_set(data, legacy_labels)
        legacy_train_z, legacy_val_z = zscore_pair(legacy_train, legacy_val)
        spaces = [
            ("legacy_combo", legacy_train_z, legacy_val_z),
            ("cls_legacy_combo", np.concatenate([cls_train, legacy_train_z], axis=1), np.concatenate([cls_val, legacy_val_z], axis=1)),
            ("roi35_legacy_combo", np.concatenate([roi_train, legacy_train_z], axis=1), np.concatenate([roi_val, legacy_val_z], axis=1)),
            (
                "cls_roi35_legacy_combo",
                np.concatenate([cls_train, roi_train, legacy_train_z], axis=1),
                np.concatenate([cls_val, roi_val, legacy_val_z], axis=1),
            ),
        ]
        for space, train_features, val_features in spaces:
            weights = train_effect_weights(train_features, train_labels)
            results.append(
                {
                    "space": space,
                    "method": "train_effect_sum",
                    "feature_count": int(train_features.shape[1]),
                    "legacy_features": legacy_labels,
                    "metrics": ranking_metrics(val_features @ weights, val_labels),
                }
            )
            columns = np.arange(train_features.shape[1])
            results.append(
                {
                    "space": space,
                    "method": "center_margin",
                    "feature_count": int(train_features.shape[1]),
                    "legacy_features": legacy_labels,
                    "metrics": ranking_metrics(center_margin_scores(train_features, train_labels, val_features, columns), val_labels),
                }
            )
            for regularization in (0.03, 0.1, 0.3, 1.0):
                model = LogisticRegression(
                    C=regularization,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                )
                model.fit(train_features, train_labels.astype(np.int32))
                results.append(
                    {
                        "space": space,
                        "method": f"logreg_balanced_C{regularization:g}",
                        "feature_count": int(train_features.shape[1]),
                        "legacy_features": legacy_labels,
                        "metrics": ranking_metrics(model.decision_function(val_features), val_labels),
                    }
                )
    return sort_rows(results), roi_variance


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.cache)
    train_labels = data["train_labels"].astype(bool)
    val_labels = data["val_labels"].astype(bool)
    report: dict[str, object] = {
        "cache": str(args.cache),
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
        "notes": [
            "Combination search uses train-derived scoring rules, then ranks many candidate feature subsets by validation labels.",
            "Use this as an exploratory upper-bound screen; freeze selected subsets before a final verifier run.",
        ],
        "single_feature_top": {},
        "combo_top": {},
    }

    combo_pool: list[dict[str, object]] = []
    for crop in args.crops:
        train_features = data[f"train_legacy_ifft_{crop}"]
        val_features = data[f"val_legacy_ifft_{crop}"]
        feature_names = [f"{name}@{crop}" for name in LEGACY_FEATURE_NAMES]
        single_rows = univariate_rows(train_features, val_features, train_labels, val_labels, feature_names)
        report["single_feature_top"][f"legacy_ifft_{crop}"] = single_rows
        candidate_indices = [int(row["index"]) for row in single_rows[: args.top_n_per_crop]]
        combo_rows = search_small_combinations(
            space=f"legacy_ifft_{crop}",
            train_features=train_features,
            val_features=val_features,
            train_labels=train_labels,
            val_labels=val_labels,
            feature_names=feature_names,
            candidate_indices=candidate_indices,
            max_combo_size=args.max_combo_size,
        )
        log_rows = logreg_rows(
            space=f"legacy_ifft_{crop}",
            train_features=train_features,
            val_features=val_features,
            train_labels=train_labels,
            val_labels=val_labels,
            feature_names=feature_names,
            candidate_indices=candidate_indices[:10],
            max_combo_size=min(4, args.max_combo_size),
        )
        merged = sort_rows(combo_rows[:120] + log_rows[:120])
        report["combo_top"][f"legacy_ifft_{crop}"] = merged[:80]
        combo_pool.extend(merged[:30])
        best = merged[0]["metrics"]  # type: ignore[index]
        print(
            f"crop={crop} best R70={best['recall_at_precision_0.7']:.4f} "
            f"R80={best['recall_at_precision_0.8']:.4f} AP={best['average_precision']:.4f}"
        )

    all_train = np.concatenate([data[f"train_legacy_ifft_{crop}"] for crop in args.crops], axis=1)
    all_val = np.concatenate([data[f"val_legacy_ifft_{crop}"] for crop in args.crops], axis=1)
    all_feature_names = [f"{name}@{crop}" for crop in args.crops for name in LEGACY_FEATURE_NAMES]
    single_all = univariate_rows(all_train, all_val, train_labels, val_labels, all_feature_names)
    report["single_feature_top"]["all_crops"] = single_all[:60]
    top_indices = [int(row["index"]) for row in single_all[: args.top_n_all_crop]]
    reduced_train = all_train[:, top_indices]
    reduced_val = all_val[:, top_indices]
    reduced_names = [all_feature_names[index] for index in top_indices]
    combo_rows = search_small_combinations(
        space=f"all_crop_top{args.top_n_all_crop}",
        train_features=reduced_train,
        val_features=reduced_val,
        train_labels=train_labels,
        val_labels=val_labels,
        feature_names=reduced_names,
        candidate_indices=list(range(reduced_train.shape[1])),
        max_combo_size=args.max_combo_size,
    )
    log_rows = logreg_rows(
        space=f"all_crop_top{args.top_n_all_crop}",
        train_features=reduced_train,
        val_features=reduced_val,
        train_labels=train_labels,
        val_labels=val_labels,
        feature_names=reduced_names,
        candidate_indices=list(range(min(12, reduced_train.shape[1]))),
        max_combo_size=min(4, args.max_combo_size),
    )
    all_merged = sort_rows(combo_rows[:160] + log_rows[:120])
    report["combo_top"][f"all_crop_top{args.top_n_all_crop}"] = all_merged[:100]
    combo_pool.extend(all_merged[:40])
    best = all_merged[0]["metrics"]  # type: ignore[index]
    print(
        f"all_crop best R70={best['recall_at_precision_0.7']:.4f} "
        f"R80={best['recall_at_precision_0.8']:.4f} AP={best['average_precision']:.4f}"
    )

    fusion_rows, roi_variance = evaluate_fusions(
        data=data,
        combo_rows=sort_rows(combo_pool),
        train_labels=train_labels,
        val_labels=val_labels,
        max_rows=args.fusion_top_combos,
    )
    report["roi35_explained_variance"] = roi_variance
    report["fusion_top"] = fusion_rows[:120]

    leaderboard = sort_rows(combo_pool + fusion_rows)
    report["leaderboard_top"] = leaderboard[:80]
    output_path = run_dir / "combo_search_report.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    for index, row in enumerate(leaderboard[:25], start=1):
        metrics = row["metrics"]  # type: ignore[index]
        features = row.get("legacy_features", row.get("features", []))
        print(
            f"#{index:02d} {row['space']} {row['method']} "
            f"k={row['feature_count']} "
            f"R70={metrics['recall_at_precision_0.7']:.4f} "
            f"R80={metrics['recall_at_precision_0.8']:.4f} "
            f"R90={metrics['recall_at_precision_0.9']:.4f} "
            f"AP={metrics['average_precision']:.4f} "
            f"AUC={metrics['auc']:.4f} "
            f"sel70={metrics['recall_at_precision_0.7_selected']} "
            f"tp70={metrics['recall_at_precision_0.7_tp']} "
            f"fp70={metrics['recall_at_precision_0.7_fp']} "
            f"features={features}"
        )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
