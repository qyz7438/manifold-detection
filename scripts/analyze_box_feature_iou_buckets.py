from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import round2129_nwpu_posttrain_smoke as round2129  # noqa: E402
from spectral_detection_posttrain.analysis.dimensionality import (  # noqa: E402
    binary_ranking_metrics,
    center_margin_scores,
    pca_dimensionality_summary,
)
from spectral_detection_posttrain.rlvr.confidence_rescue import match_boxes_to_targets  # noqa: E402
from spectral_detection_posttrain.utils.io import save_json  # noqa: E402
from spectral_detection_posttrain.utils.seed import set_seed  # noqa: E402


BUCKETS = [
    ("iou_0_0p3", 0.0, 0.3),
    ("iou_0p3_0p5", 0.3, 0.5),
    ("iou_0p5_0p75", 0.5, 0.75),
    ("iou_ge_0p75", 0.75, 1.000001),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pre-detection-head box_features by IoU buckets.")
    parser.add_argument("--run-name", default="round2198_box_feature_iou_bucket_manifold")
    parser.add_argument("--limit-train", type=int, default=100000)
    parser.add_argument("--limit-val", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=100)
    parser.add_argument("--rollout-score-threshold", type=float, default=0.001)
    parser.add_argument("--rollout-detections-per-img", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--low-conf-max", type=float, default=0.5)
    parser.add_argument("--pca-components", type=int, nargs="+", default=[32, 48, 56, 58, 64, 96, 128])
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--max-train-per-bin", type=int, default=2000)
    parser.add_argument("--max-val-per-bin", type=int, default=2000)
    return parser.parse_args()


def bucket_ids(ious: np.ndarray) -> np.ndarray:
    ids = np.full((ious.shape[0],), -1, dtype=np.int64)
    for index, (_, lo, hi) in enumerate(BUCKETS):
        if index == 0:
            mask = (ious >= lo) & (ious <= hi)
        elif index == len(BUCKETS) - 1:
            mask = ious >= lo
        else:
            mask = (ious > lo) & (ious < hi)
        ids[mask] = index
    return ids


def balanced_subset(bucket_id: np.ndarray, *, max_per_bin: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = []
    for bucket in range(len(BUCKETS)):
        bucket_indices = np.flatnonzero(bucket_id == bucket)
        if bucket_indices.shape[0] > int(max_per_bin):
            bucket_indices = rng.choice(bucket_indices, size=int(max_per_bin), replace=False)
        indices.append(bucket_indices)
    if not indices:
        return np.empty((0,), dtype=np.int64)
    return np.sort(np.concatenate(indices))


@torch.no_grad()
def collect_split(model, loader, device: torch.device, args: argparse.Namespace) -> dict[str, np.ndarray]:
    features = []
    features_l2 = []
    best_ious = []
    matched_probs = []
    class_ids = []
    image_ids = []
    proposal_scores = []

    model.eval()
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for image, target, output in zip(images, targets, outputs):
            proposals, scores = round2129.select_rollout_proposals(output, args)
            if proposals.numel() == 0:
                continue
            class_logits, _, box_features, _, _ = round2129.extract_roi_outputs_and_features_for_boxes(
                model,
                [image.to(device)],
                [proposals],
            )
            gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
            best_iou, best_labels = match_boxes_to_targets(proposals.to(device), gt_boxes, gt_labels)
            label_probs = round2129.matched_label_probabilities(class_logits, best_labels)
            features.append(box_features.detach().cpu().numpy())
            features_l2.append(F.normalize(box_features.detach(), p=2, dim=1).cpu().numpy())
            best_ious.append(best_iou.detach().cpu().numpy())
            matched_probs.append(label_probs.detach().cpu().numpy())
            class_ids.append(best_labels.detach().cpu().numpy())
            image_id = target.get("image_id", torch.tensor([-1]))
            image_value = int(image_id.flatten()[0].item()) if torch.is_tensor(image_id) else int(image_id)
            image_ids.append(np.full((int(proposals.shape[0]),), image_value, dtype=np.int64))
            proposal_scores.append(scores.detach().cpu().numpy())

    if not features:
        return {
            "features": np.empty((0, 1024), dtype=np.float32),
            "features_l2": np.empty((0, 1024), dtype=np.float32),
            "best_iou": np.empty((0,), dtype=np.float32),
            "matched_prob": np.empty((0,), dtype=np.float32),
            "class_id": np.empty((0,), dtype=np.int64),
            "image_id": np.empty((0,), dtype=np.int64),
            "proposal_score": np.empty((0,), dtype=np.float32),
        }
    return {
        "features": np.concatenate(features, axis=0),
        "features_l2": np.concatenate(features_l2, axis=0),
        "best_iou": np.concatenate(best_ious, axis=0),
        "matched_prob": np.concatenate(matched_probs, axis=0),
        "class_id": np.concatenate(class_ids, axis=0),
        "image_id": np.concatenate(image_ids, axis=0),
        "proposal_score": np.concatenate(proposal_scores, axis=0),
    }


def zscore(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(val)


def pca_space(train: np.ndarray, val: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, float]:
    n_components = min(int(components), train.shape[0] - 1, train.shape[1])
    if n_components <= 0:
        return np.empty((train.shape[0], 0)), np.empty((val.shape[0], 0)), 0.0
    pca = PCA(n_components=n_components, whiten=True, random_state=42).fit(train)
    return pca.transform(train), pca.transform(val), float(pca.explained_variance_ratio_.sum())


def knn_density_ratio_scores(train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, *, k: int) -> np.ndarray:
    positive = train_features[train_labels]
    negative = train_features[~train_labels]
    if positive.size == 0 or negative.size == 0:
        return np.zeros((val_features.shape[0],), dtype=np.float64)

    def mean_distance(reference: np.ndarray) -> np.ndarray:
        k_eff = min(max(1, int(k)), reference.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(reference)
        distances, _ = nn.kneighbors(val_features, return_distance=True)
        return distances.mean(axis=1)

    return mean_distance(negative) - mean_distance(positive)


def classwise_center_margin_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_classes: np.ndarray,
    val_features: np.ndarray,
    val_classes: np.ndarray,
    *,
    min_pos: int = 5,
    min_neg: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    global_scores = center_margin_scores(train_features, train_labels, val_features)
    scores = global_scores.copy()
    diagnostics: dict[str, object] = {"used_classes": [], "fallback_classes": []}
    for class_id in np.unique(val_classes.astype(np.int64)):
        train_mask = train_classes == int(class_id)
        val_mask = val_classes == int(class_id)
        pos_count = int((train_mask & train_labels).sum())
        neg_count = int((train_mask & (~train_labels)).sum())
        if pos_count < int(min_pos) or neg_count < int(min_neg):
            diagnostics["fallback_classes"].append(  # type: ignore[union-attr]
                {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
            )
            continue
        scores[val_mask] = center_margin_scores(
            train_features[train_mask],
            train_labels[train_mask],
            val_features[val_mask],
        )
        diagnostics["used_classes"].append(  # type: ignore[union-attr]
            {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
        )
    return scores, diagnostics


def classwise_knn_density_ratio_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_classes: np.ndarray,
    val_features: np.ndarray,
    val_classes: np.ndarray,
    *,
    k: int,
    min_pos: int = 5,
    min_neg: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    global_scores = knn_density_ratio_scores(train_features, train_labels, val_features, k=k)
    scores = global_scores.copy()
    diagnostics: dict[str, object] = {"used_classes": [], "fallback_classes": []}
    for class_id in np.unique(val_classes.astype(np.int64)):
        train_mask = train_classes == int(class_id)
        val_mask = val_classes == int(class_id)
        pos_count = int((train_mask & train_labels).sum())
        neg_count = int((train_mask & (~train_labels)).sum())
        if pos_count < int(min_pos) or neg_count < int(min_neg):
            diagnostics["fallback_classes"].append(  # type: ignore[union-attr]
                {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
            )
            continue
        scores[val_mask] = knn_density_ratio_scores(
            train_features[train_mask],
            train_labels[train_mask],
            val_features[val_mask],
            k=k,
        )
        diagnostics["used_classes"].append(  # type: ignore[union-attr]
            {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
        )
    return scores, diagnostics


def logistic_scores(train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
    model.fit(train_features, train_labels.astype(np.int32))
    return model.decision_function(val_features)


def evaluate_binary(
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_positive: np.ndarray,
    val_positive: np.ndarray,
    train_classes: np.ndarray,
    val_classes: np.ndarray,
    *,
    k: int,
) -> dict[str, dict[str, float]]:
    class_center_scores, class_center_diag = classwise_center_margin_scores(
        train_features,
        train_positive,
        train_classes,
        val_features,
        val_classes,
    )
    class_knn_scores, class_knn_diag = classwise_knn_density_ratio_scores(
        train_features,
        train_positive,
        train_classes,
        val_features,
        val_classes,
        k=k,
    )
    output = {
        "center_margin": binary_ranking_metrics(
            center_margin_scores(train_features, train_positive, val_features),
            val_positive,
        ),
        "knn_density_ratio": binary_ranking_metrics(
            knn_density_ratio_scores(train_features, train_positive, val_features, k=k),
            val_positive,
        ),
        "logistic_balanced": binary_ranking_metrics(
            logistic_scores(train_features, train_positive, val_features),
            val_positive,
        ),
        "classwise_center_margin": binary_ranking_metrics(class_center_scores, val_positive),
        "classwise_knn_density_ratio": binary_ranking_metrics(class_knn_scores, val_positive),
    }
    output["classwise_center_diagnostics"] = class_center_diag  # type: ignore[assignment]
    output["classwise_knn_diagnostics"] = class_knn_diag  # type: ignore[assignment]
    return output


def count_buckets(split: dict[str, np.ndarray]) -> dict[str, int]:
    ids = bucket_ids(split["best_iou"])
    return {name: int((ids == index).sum()) for index, (name, _, _) in enumerate(BUCKETS)}


def mean_by_bucket(values: np.ndarray, ids: np.ndarray) -> dict[str, float]:
    return {
        name: float(values[ids == index].mean()) if (ids == index).any() else 0.0
        for index, (name, _, _) in enumerate(BUCKETS)
    }


def build_spaces(train: dict[str, np.ndarray], val: dict[str, np.ndarray], components: list[int]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    train_z, val_z = zscore(train["features"], val["features"])
    train_l2_z, val_l2_z = zscore(train["features_l2"], val["features_l2"])
    spaces: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "features_raw": (train["features"], val["features"]),
        "features_z": (train_z, val_z),
        "features_l2": (train["features_l2"], val["features_l2"]),
        "features_l2_z": (train_l2_z, val_l2_z),
    }
    explained: dict[str, float] = {}
    for n in components:
        p_train, p_val, exp = pca_space(train_z, val_z, int(n))
        if p_train.shape[1] > 0:
            key = f"features_z_pca{p_train.shape[1]}"
            spaces[key] = (p_train, p_val)
            explained[key] = exp
        lp_train, lp_val, l_exp = pca_space(train_l2_z, val_l2_z, int(n))
        if lp_train.shape[1] > 0:
            key = f"features_l2_z_pca{lp_train.shape[1]}"
            spaces[key] = (lp_train, lp_val)
            explained[key] = l_exp
    return spaces, explained


def evaluate_spaces(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    args: argparse.Namespace,
    *,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    task_name: str,
) -> dict[str, object]:
    train_subset = {key: value[train_indices] for key, value in train.items()}
    val_subset = {key: value[val_indices] for key, value in val.items()}
    train_bucket = bucket_ids(train_subset["best_iou"])
    val_bucket = bucket_ids(val_subset["best_iou"])
    train_positive = train_bucket == 3
    val_positive = val_bucket == 3
    train_valid = (train_bucket == 0) | (train_bucket == 3)
    val_valid = (val_bucket == 0) | (val_bucket == 3)

    task_report: dict[str, object] = {
        "task": task_name,
        "train_count": int(train_subset["best_iou"].shape[0]),
        "val_count": int(val_subset["best_iou"].shape[0]),
        "train_bucket_counts": count_buckets(train_subset),
        "val_bucket_counts": count_buckets(val_subset),
        "train_low_conf_count": int((train_subset["matched_prob"] <= float(args.low_conf_max)).sum()),
        "val_low_conf_count": int((val_subset["matched_prob"] <= float(args.low_conf_max)).sum()),
        "matched_prob_mean_by_bucket": {
            "train": mean_by_bucket(train_subset["matched_prob"], train_bucket),
            "val": mean_by_bucket(val_subset["matched_prob"], val_bucket),
        },
        "spaces": {},
    }
    if int(train_positive.sum()) == 0 or int((~train_positive[train_valid]).sum()) == 0:
        task_report["error"] = "missing train positive or negative bucket"
        return task_report
    if int(val_positive.sum()) == 0 or int((~val_positive[val_valid]).sum()) == 0:
        task_report["error"] = "missing val positive or negative bucket"
        return task_report

    binary_train = {key: value[train_valid] for key, value in train_subset.items()}
    binary_val = {key: value[val_valid] for key, value in val_subset.items()}
    binary_train_positive = bucket_ids(binary_train["best_iou"]) == 3
    binary_val_positive = bucket_ids(binary_val["best_iou"]) == 3
    spaces, explained = build_spaces(binary_train, binary_val, list(args.pca_components))
    rows = []
    evaluated = {}
    for name, (train_features, val_features) in spaces.items():
        item = evaluate_binary(
            train_features,
            val_features,
            binary_train_positive,
            binary_val_positive,
            binary_train["class_id"],
            binary_val["class_id"],
            k=int(args.knn_k),
        )
        if name in explained:
            item["explained_variance"] = explained[name]  # type: ignore[index]
        evaluated[name] = item
        for method, metrics in item.items():
            if method in {"explained_variance", "classwise_center_diagnostics", "classwise_knn_diagnostics"}:
                continue
            rows.append(
                {
                    "space": name,
                    "method": method,
                    "feature_dim": int(train_features.shape[1]),
                    "auc": float(metrics["auc"]),
                    "average_precision": float(metrics["average_precision"]),
                    "recall_at_precision_0.7": float(metrics["recall_at_precision_0.7"]),
                    "recall_at_precision_0.8": float(metrics["recall_at_precision_0.8"]),
                    "recall_at_precision_0.9": float(metrics["recall_at_precision_0.9"]),
                }
            )
    task_report["spaces"] = evaluated
    task_report["leaderboard"] = sorted(
        rows,
        key=lambda row: (
            row["recall_at_precision_0.7"],
            row["average_precision"],
            row["auc"],
        ),
        reverse=True,
    )
    return task_report


def main() -> None:
    args = parse_args()
    args.rescue_mode = True
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), run_dir / "config.json")

    loader_args = SimpleNamespace(
        limit_train=int(args.limit_train),
        limit_val=int(args.limit_val),
        batch_size=int(args.batch_size),
    )
    train_loader, val_loader = round2129.build_loaders(loader_args)
    model = round2129.build_nwpu_model(device)
    round2129.configure_detector_rollout(
        model,
        score_threshold=float(args.rollout_score_threshold),
        detections_per_img=int(args.rollout_detections_per_img),
    )
    model.eval()

    train = collect_split(model, train_loader, device, args)
    val = collect_split(model, val_loader, device, args)
    np.savez_compressed(
        run_dir / "iou_bucket_box_features.npz",
        **{f"train_{key}": value for key, value in train.items()},
        **{f"val_{key}": value for key, value in val.items()},
    )

    train_bucket = bucket_ids(train["best_iou"])
    val_bucket = bucket_ids(val["best_iou"])
    train_balanced = balanced_subset(train_bucket, max_per_bin=int(args.max_train_per_bin), seed=42)
    val_balanced = balanced_subset(val_bucket, max_per_bin=int(args.max_val_per_bin), seed=123)
    train_low_conf = np.flatnonzero(train["matched_prob"] <= float(args.low_conf_max))
    val_low_conf = np.flatnonzero(val["matched_prob"] <= float(args.low_conf_max))

    report: dict[str, object] = {
        "device": str(device),
        "feature_dim": int(train["features"].shape[1]) if train["features"].ndim == 2 else 0,
        "buckets": [{"name": name, "lo": lo, "hi": hi} for name, lo, hi in BUCKETS],
        "all_train_count": int(train["best_iou"].shape[0]),
        "all_val_count": int(val["best_iou"].shape[0]),
        "bucket_counts": {
            "train": count_buckets(train),
            "val": count_buckets(val),
        },
        "dimensionality": {
            "features": pca_dimensionality_summary(train["features"], max_components=256),
            "features_l2": pca_dimensionality_summary(train["features_l2"], max_components=256),
        },
        "tasks": {
            "balanced_iou_0_0p3_vs_ge_0p75": evaluate_spaces(
                train,
                val,
                args,
                train_indices=train_balanced,
                val_indices=val_balanced,
                task_name="balanced_iou_0_0p3_vs_ge_0p75",
            ),
            "low_conf_iou_0_0p3_vs_ge_0p75": evaluate_spaces(
                train,
                val,
                args,
                train_indices=train_low_conf,
                val_indices=val_low_conf,
                task_name="low_conf_iou_0_0p3_vs_ge_0p75",
            ),
        },
    }
    save_json(report, run_dir / "iou_bucket_manifold_report.json")
    print(json.dumps(report["tasks"], ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
