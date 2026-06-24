from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.analysis.dimensionality import (  # noqa: E402
    binary_ranking_metrics,
    center_margin_scores,
    pca_dimensionality_summary,
)
from spectral_detection_posttrain.utils.io import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused manifold analysis for pre-detection-head box_features.")
    parser.add_argument(
        "--cache",
        default="runs/round2148_final_head_dim_full_ap75/candidate_features.npz",
        help="Candidate feature cache containing train_features/val_features.",
    )
    parser.add_argument("--run-name", default="round2196_box_feature_manifold_focus")
    parser.add_argument("--pca-components", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--knn-k", type=int, default=5)
    return parser.parse_args()


def ensure_bool(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(bool)


def zscore_train_val(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(val), scaler


def fit_pca(train: np.ndarray, val: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, float]:
    n_components = min(int(components), train.shape[0] - 1, train.shape[1])
    if n_components <= 0:
        return np.empty((train.shape[0], 0)), np.empty((val.shape[0], 0)), 0.0
    pca = PCA(n_components=n_components, whiten=True, random_state=42).fit(train)
    return pca.transform(train), pca.transform(val), float(pca.explained_variance_ratio_.sum())


def knn_density_ratio_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    positive = train_features[train_labels]
    negative = train_features[~train_labels]
    if positive.size == 0 or negative.size == 0:
        return np.zeros((query_features.shape[0],), dtype=np.float64)

    def mean_distance(reference: np.ndarray) -> np.ndarray:
        k_eff = min(max(1, int(k)), reference.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(reference)
        distances, _ = nn.kneighbors(query_features, return_distance=True)
        return distances.mean(axis=1)

    return mean_distance(negative) - mean_distance(positive)


def logistic_metrics(train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray) -> dict[str, object]:
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    model.fit(train_features, train_labels.astype(np.int32))
    val_scores = model.decision_function(val_features)
    metrics = binary_ranking_metrics(val_scores, val_labels)
    cv_auc = 0.0
    if int(train_labels.sum()) >= 3 and int((~train_labels).sum()) >= 3:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_auc = float(cross_val_score(model, train_features, train_labels.astype(np.int32), cv=cv, scoring="roc_auc").mean())
    return {"val": metrics, "train_cv_auc": cv_auc}


def describe_score_distribution(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = ensure_bool(labels)
    positive = scores[labels]
    negative = scores[~labels]
    return {
        "positive_mean": float(positive.mean()) if positive.size else 0.0,
        "positive_std": float(positive.std()) if positive.size else 0.0,
        "negative_mean": float(negative.mean()) if negative.size else 0.0,
        "negative_std": float(negative.std()) if negative.size else 0.0,
        "positive_p90": float(np.quantile(positive, 0.9)) if positive.size else 0.0,
        "negative_p90": float(np.quantile(negative, 0.9)) if negative.size else 0.0,
    }


def evaluate_space(
    name: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    *,
    k: int,
) -> dict[str, object]:
    center_scores = center_margin_scores(train_features, train_labels, val_features)
    knn_scores = knn_density_ratio_scores(train_features, train_labels, val_features, k=k)
    return {
        "name": name,
        "feature_dim": int(train_features.shape[1]),
        "center_margin": {
            "metrics": binary_ranking_metrics(center_scores, val_labels),
            "score_distribution": describe_score_distribution(center_scores, val_labels),
        },
        "knn_density_ratio": {
            "metrics": binary_ranking_metrics(knn_scores, val_labels),
            "score_distribution": describe_score_distribution(knn_scores, val_labels),
        },
        "logistic_balanced": logistic_metrics(train_features, train_labels, val_features, val_labels),
    }


def classwise_center_margin_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_classes: np.ndarray,
    query_features: np.ndarray,
    query_classes: np.ndarray,
    *,
    min_pos: int = 2,
    min_neg: int = 2,
) -> tuple[np.ndarray, dict[str, object]]:
    global_scores = center_margin_scores(train_features, train_labels, query_features)
    scores = global_scores.copy()
    diagnostics: dict[str, object] = {
        "used_classes": [],
        "fallback_classes": [],
    }
    for class_id in np.unique(query_classes.astype(np.int64)):
        query_mask = query_classes == class_id
        train_mask = train_classes == class_id
        pos_count = int((train_mask & train_labels).sum())
        neg_count = int((train_mask & (~train_labels)).sum())
        if pos_count < int(min_pos) or neg_count < int(min_neg):
            diagnostics["fallback_classes"].append(  # type: ignore[union-attr]
                {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
            )
            continue
        class_scores = center_margin_scores(
            train_features[train_mask],
            train_labels[train_mask],
            query_features[query_mask],
        )
        scores[query_mask] = class_scores
        diagnostics["used_classes"].append(  # type: ignore[union-attr]
            {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
        )
    return scores, diagnostics


def classwise_knn_density_ratio_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_classes: np.ndarray,
    query_features: np.ndarray,
    query_classes: np.ndarray,
    *,
    k: int,
    min_pos: int = 2,
    min_neg: int = 2,
) -> tuple[np.ndarray, dict[str, object]]:
    global_scores = knn_density_ratio_scores(train_features, train_labels, query_features, k=k)
    scores = global_scores.copy()
    diagnostics: dict[str, object] = {
        "used_classes": [],
        "fallback_classes": [],
    }
    for class_id in np.unique(query_classes.astype(np.int64)):
        query_mask = query_classes == class_id
        train_mask = train_classes == class_id
        pos_count = int((train_mask & train_labels).sum())
        neg_count = int((train_mask & (~train_labels)).sum())
        if pos_count < int(min_pos) or neg_count < int(min_neg):
            diagnostics["fallback_classes"].append(  # type: ignore[union-attr]
                {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
            )
            continue
        class_scores = knn_density_ratio_scores(
            train_features[train_mask],
            train_labels[train_mask],
            query_features[query_mask],
            k=k,
        )
        scores[query_mask] = class_scores
        diagnostics["used_classes"].append(  # type: ignore[union-attr]
            {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
        )
    return scores, diagnostics


def evaluate_classwise_space(
    name: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    train_classes: np.ndarray,
    val_classes: np.ndarray,
    *,
    k: int,
) -> dict[str, object]:
    center_scores, center_diag = classwise_center_margin_scores(
        train_features,
        train_labels,
        train_classes,
        val_features,
        val_classes,
    )
    knn_scores, knn_diag = classwise_knn_density_ratio_scores(
        train_features,
        train_labels,
        train_classes,
        val_features,
        val_classes,
        k=k,
    )
    return {
        "name": name,
        "feature_dim": int(train_features.shape[1]),
        "classwise_center_margin": {
            "metrics": binary_ranking_metrics(center_scores, val_labels),
            "score_distribution": describe_score_distribution(center_scores, val_labels),
            "diagnostics": center_diag,
        },
        "classwise_knn_density_ratio": {
            "metrics": binary_ranking_metrics(knn_scores, val_labels),
            "score_distribution": describe_score_distribution(knn_scores, val_labels),
            "diagnostics": knn_diag,
        },
    }


def top_rows(report: dict[str, object], metric_key: str = "recall_at_precision_0.7") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, item in report["spaces"].items():  # type: ignore[index,union-attr]
        assert isinstance(item, dict)
        for method in ("center_margin", "knn_density_ratio"):
            metrics = item[method]["metrics"]  # type: ignore[index]
            rows.append(
                {
                    "space": name,
                    "method": method,
                    "feature_dim": item["feature_dim"],
                    "auc": metrics["auc"],
                    "average_precision": metrics["average_precision"],
                    "recall_at_precision_0.7": metrics["recall_at_precision_0.7"],
                    "recall_at_precision_0.8": metrics["recall_at_precision_0.8"],
                    "recall_at_precision_0.9": metrics["recall_at_precision_0.9"],
                }
            )
        metrics = item["logistic_balanced"]["val"]  # type: ignore[index]
        rows.append(
            {
                "space": name,
                "method": "logistic_balanced_C0.1",
                "feature_dim": item["feature_dim"],
                "auc": metrics["auc"],
                "average_precision": metrics["average_precision"],
                "recall_at_precision_0.7": metrics["recall_at_precision_0.7"],
                "recall_at_precision_0.8": metrics["recall_at_precision_0.8"],
                "recall_at_precision_0.9": metrics["recall_at_precision_0.9"],
                "train_cv_auc": item["logistic_balanced"]["train_cv_auc"],  # type: ignore[index]
            }
        )
        if "classwise_center_margin" in item:
            for method in ("classwise_center_margin", "classwise_knn_density_ratio"):
                metrics = item[method]["metrics"]  # type: ignore[index]
                rows.append(
                    {
                        "space": name,
                        "method": method,
                        "feature_dim": item["feature_dim"],
                        "auc": metrics["auc"],
                        "average_precision": metrics["average_precision"],
                        "recall_at_precision_0.7": metrics["recall_at_precision_0.7"],
                        "recall_at_precision_0.8": metrics["recall_at_precision_0.8"],
                        "recall_at_precision_0.9": metrics["recall_at_precision_0.9"],
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            float(row.get(metric_key, 0.0)),
            float(row.get("average_precision", 0.0)),
            float(row.get("auc", 0.0)),
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.cache)

    train_labels = ensure_bool(data["train_labels"])
    val_labels = ensure_bool(data["val_labels"])
    train_classes = np.asarray(data["train_class_ids"], dtype=np.int64)
    val_classes = np.asarray(data["val_class_ids"], dtype=np.int64)
    train_features = np.asarray(data["train_features"], dtype=np.float64)
    val_features = np.asarray(data["val_features"], dtype=np.float64)
    train_features_l2 = np.asarray(data["train_features_l2"], dtype=np.float64)
    val_features_l2 = np.asarray(data["val_features_l2"], dtype=np.float64)

    train_z, val_z, _ = zscore_train_val(train_features, val_features)
    train_l2_z, val_l2_z, _ = zscore_train_val(train_features_l2, val_features_l2)

    report: dict[str, object] = {
        "cache": str(args.cache),
        "train_count": int(train_labels.shape[0]),
        "train_positive_count": int(train_labels.sum()),
        "train_negative_count": int((~train_labels).sum()),
        "val_count": int(val_labels.shape[0]),
        "val_positive_count": int(val_labels.sum()),
        "val_negative_count": int((~val_labels).sum()),
        "feature_dim": int(train_features.shape[1]),
        "dimensionality": {
            "features": pca_dimensionality_summary(train_features, max_components=256),
            "features_l2": pca_dimensionality_summary(train_features_l2, max_components=256),
        },
        "spaces": {},
    }

    spaces: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "features_raw": (train_features, val_features),
        "features_z": (train_z, val_z),
        "features_l2": (train_features_l2, val_features_l2),
        "features_l2_z": (train_l2_z, val_l2_z),
    }
    for components in args.pca_components:
        p_train, p_val, explained = fit_pca(train_z, val_z, components)
        if p_train.shape[1] > 0:
            spaces[f"features_z_pca{p_train.shape[1]}"] = (p_train, p_val)
            report["spaces"][f"features_z_pca{p_train.shape[1]}_explained"] = explained  # type: ignore[index]
        lp_train, lp_val, l_explained = fit_pca(train_l2_z, val_l2_z, components)
        if lp_train.shape[1] > 0:
            spaces[f"features_l2_z_pca{lp_train.shape[1]}"] = (lp_train, lp_val)
            report["spaces"][f"features_l2_z_pca{lp_train.shape[1]}_explained"] = l_explained  # type: ignore[index]

    evaluated = {}
    for name, (train_space, val_space) in spaces.items():
        item = evaluate_space(
            name,
            train_space,
            val_space,
            train_labels,
            val_labels,
            k=int(args.knn_k),
        )
        item.update(
            evaluate_classwise_space(
                name,
                train_space,
                val_space,
                train_labels,
                val_labels,
                train_classes,
                val_classes,
                k=int(args.knn_k),
            )
        )
        evaluated[name] = item
    report["spaces"] = evaluated
    report["leaderboard"] = top_rows(report)

    save_json(vars(args), run_dir / "config.json")
    save_json(report, run_dir / "box_feature_manifold_report.json")
    print(json.dumps(report["leaderboard"][:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
