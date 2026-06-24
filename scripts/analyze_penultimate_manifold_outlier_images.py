from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DEFAULT_FULL_CACHE = Path("runs/round2199_box_feature_classwise_iou_bucket_manifold/iou_bucket_box_features.npz")
DEFAULT_LC_CACHE = Path("runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz")
DEFAULT_COCO = Path("data/NWPU_VHR10_coco.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank NWPU images whose penultimate ROI box_features leave the train manifold."
    )
    parser.add_argument("--full-cache", type=Path, default=DEFAULT_FULL_CACHE)
    parser.add_argument("--lc-cache", type=Path, default=DEFAULT_LC_CACHE)
    parser.add_argument("--coco-json", type=Path, default=DEFAULT_COCO)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/round2204_penultimate_manifold_outlier_images"))
    parser.add_argument("--pca-components", type=int, default=96)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--low-conf-max", type=float, default=0.5)
    parser.add_argument("--high-iou-min", type=float, default=0.75)
    parser.add_argument("--low-iou-max", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def percentile(values: np.ndarray, q: float) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return safe_float(np.percentile(values, q))


def zscore_from_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(reference))
    std = float(np.nanstd(reference))
    if std < 1e-12:
        std = 1.0
    return (values - mean) / std


def fit_pca_scores(
    train_x: np.ndarray,
    val_x: np.ndarray,
    *,
    pca_components: int,
    knn_k: int,
) -> dict[str, np.ndarray | float | int]:
    scaler = StandardScaler().fit(train_x)
    train_z = scaler.transform(train_x)
    val_z = scaler.transform(val_x)
    n_components = min(int(pca_components), train_z.shape[0] - 1, train_z.shape[1])
    pca = PCA(n_components=n_components, whiten=False, random_state=42).fit(train_z)

    train_p = pca.transform(train_z)
    val_p = pca.transform(val_z)
    train_recon = pca.inverse_transform(train_p)
    val_recon = pca.inverse_transform(val_p)
    train_recon_resid = np.linalg.norm(train_z - train_recon, axis=1) / math.sqrt(train_z.shape[1])
    val_recon_resid = np.linalg.norm(val_z - val_recon, axis=1) / math.sqrt(train_z.shape[1])

    k_train = min(max(2, int(knn_k) + 1), train_p.shape[0])
    nn = NearestNeighbors(n_neighbors=k_train, metric="euclidean")
    nn.fit(train_p)
    train_dist, _ = nn.kneighbors(train_p, return_distance=True)
    val_dist, _ = nn.kneighbors(val_p, n_neighbors=min(int(knn_k), train_p.shape[0]), return_distance=True)
    train_knn = train_dist[:, 1:].mean(axis=1)
    val_knn = val_dist.mean(axis=1)

    return {
        "train_z": train_z,
        "val_z": val_z,
        "train_pca": train_p,
        "val_pca": val_p,
        "train_recon_resid": train_recon_resid,
        "val_recon_resid": val_recon_resid,
        "train_knn": train_knn,
        "val_knn": val_knn,
        "val_recon_z": zscore_from_reference(val_recon_resid, train_recon_resid),
        "val_knn_z": zscore_from_reference(val_knn, train_knn),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_components": int(n_components),
    }


def class_tp_knn_scores(
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    train_class: np.ndarray,
    val_class: np.ndarray,
    train_iou: np.ndarray,
    *,
    high_iou_min: float,
    knn_k: int,
    min_ref: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.full((val_pca.shape[0],), np.nan, dtype=np.float64)
    diagnostics: dict[str, Any] = {"used_classes": [], "fallback_classes": []}
    for class_id in sorted(int(c) for c in np.unique(val_class) if int(c) > 0):
        ref_mask = (train_class == class_id) & (train_iou >= high_iou_min)
        val_mask = val_class == class_id
        ref = train_pca[ref_mask]
        if ref.shape[0] < min_ref:
            diagnostics["fallback_classes"].append({"class_id": class_id, "ref_count": int(ref.shape[0])})
            continue

        k_self = min(max(2, int(knn_k) + 1), ref.shape[0])
        nn = NearestNeighbors(n_neighbors=k_self, metric="euclidean")
        nn.fit(ref)
        ref_dist, _ = nn.kneighbors(ref, return_distance=True)
        ref_knn = ref_dist[:, 1:].mean(axis=1)

        k_val = min(int(knn_k), ref.shape[0])
        val_dist, _ = nn.kneighbors(val_pca[val_mask], n_neighbors=k_val, return_distance=True)
        class_scores = zscore_from_reference(val_dist.mean(axis=1), ref_knn)
        scores[val_mask] = class_scores
        diagnostics["used_classes"].append(
            {
                "class_id": class_id,
                "ref_count": int(ref.shape[0]),
                "ref_knn_mean": safe_float(ref_knn.mean()),
                "ref_knn_std": safe_float(ref_knn.std()),
            }
        )
    return scores, diagnostics


def load_coco_metadata(coco_json: Path) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}
    images = {
        int(image["id"]): {
            "image_id": int(image["id"]),
            "file_name": image.get("file_name"),
            "width": image.get("width"),
            "height": image.get("height"),
            "gt_count": 0,
            "gt_categories": {},
        }
        for image in coco.get("images", [])
    }
    per_image_categories: dict[int, Counter[int]] = defaultdict(Counter)
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        per_image_categories[image_id][category_id] += 1
    for image_id, counts in per_image_categories.items():
        if image_id not in images:
            continue
        images[image_id]["gt_count"] = int(sum(counts.values()))
        images[image_id]["gt_categories"] = {
            categories.get(category_id, str(category_id)): int(count)
            for category_id, count in sorted(counts.items())
        }
    return images, categories


def category_name(categories: dict[int, str], class_id: int) -> str:
    return categories.get(int(class_id), f"class_{int(class_id)}")


def proposal_record(
    *,
    index: int,
    image_id: int,
    class_id: int,
    iou: float,
    prob: float,
    proposal_score: float,
    recon_z: float,
    knn_z: float,
    class_tp_z: float,
    images: dict[int, dict[str, Any]],
    categories: dict[int, str],
) -> dict[str, Any]:
    meta = images.get(int(image_id), {"file_name": None, "gt_categories": {}})
    return {
        "index": int(index),
        "image_id": int(image_id),
        "file_name": meta.get("file_name"),
        "gt_categories": meta.get("gt_categories", {}),
        "class_id": int(class_id),
        "class_name": category_name(categories, int(class_id)),
        "best_iou": safe_float(iou),
        "matched_prob": safe_float(prob),
        "proposal_score": safe_float(proposal_score),
        "recon_z": safe_float(recon_z),
        "global_knn_z": safe_float(knn_z),
        "class_tp_knn_z": safe_float(class_tp_z),
        "outlier_score": safe_float(np.nanmax(np.array([recon_z, knn_z, class_tp_z], dtype=np.float64))),
    }


def aggregate_images(
    *,
    image_ids: np.ndarray,
    class_ids: np.ndarray,
    ious: np.ndarray,
    probs: np.ndarray,
    proposal_scores: np.ndarray,
    recon_z: np.ndarray,
    knn_z: np.ndarray,
    class_tp_z: np.ndarray,
    images: dict[int, dict[str, Any]],
    categories: dict[int, str],
    low_conf_max: float,
    high_iou_min: float,
    low_iou_max: float,
) -> list[dict[str, Any]]:
    records = []
    for image_id in sorted(int(i) for i in np.unique(image_ids)):
        mask = image_ids == image_id
        lchi = mask & (probs <= low_conf_max) & (ious >= high_iou_min) & (class_ids > 0)
        hcli = mask & (probs > low_conf_max) & (ious <= low_iou_max) & (class_ids > 0)
        high_iou = mask & (ious >= high_iou_min) & (class_ids > 0)
        low_iou = mask & (ious <= low_iou_max)
        class_counter = Counter(int(c) for c in class_ids[mask] if int(c) > 0)
        dominant = [
            {"class_id": cid, "class_name": category_name(categories, cid), "count": int(count)}
            for cid, count in class_counter.most_common(4)
        ]
        outlier = np.nanmax(np.stack([recon_z[mask], knn_z[mask], class_tp_z[mask]], axis=0), axis=0)
        meta = images.get(image_id, {"file_name": None, "gt_categories": {}})
        records.append(
            {
                "image_id": image_id,
                "file_name": meta.get("file_name"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "gt_count": meta.get("gt_count", 0),
                "gt_categories": meta.get("gt_categories", {}),
                "proposal_count": int(mask.sum()),
                "lc_hi_count": int(lchi.sum()),
                "hc_li_count": int(hcli.sum()),
                "high_iou_count": int(high_iou.sum()),
                "low_iou_count": int(low_iou.sum()),
                "dominant_matched_classes": dominant,
                "mean_matched_prob": safe_float(probs[mask].mean()),
                "max_iou": safe_float(ious[mask].max()),
                "recon_z_p95": percentile(recon_z[mask], 95),
                "global_knn_z_p95": percentile(knn_z[mask], 95),
                "class_tp_knn_z_p95": percentile(class_tp_z[mask], 95),
                "outlier_p95": percentile(outlier, 95),
                "lc_hi_outlier_max": percentile(
                    np.nanmax(np.stack([recon_z[lchi], knn_z[lchi], class_tp_z[lchi]], axis=0), axis=0)
                    if int(lchi.sum()) > 0
                    else np.array([], dtype=np.float64),
                    100,
                ),
                "lc_hi_class_tp_knn_z_max": percentile(class_tp_z[lchi], 100),
                "lc_hi_mean_prob": safe_float(probs[lchi].mean()) if int(lchi.sum()) > 0 else None,
                "lc_hi_mean_iou": safe_float(ious[lchi].mean()) if int(lchi.sum()) > 0 else None,
                "proposal_score_p95": percentile(proposal_scores[mask], 95),
            }
        )
    return records


def category_summary(
    *,
    class_ids: np.ndarray,
    ious: np.ndarray,
    probs: np.ndarray,
    recon_z: np.ndarray,
    knn_z: np.ndarray,
    class_tp_z: np.ndarray,
    categories: dict[int, str],
    low_conf_max: float,
    high_iou_min: float,
) -> list[dict[str, Any]]:
    rows = []
    outlier = np.nanmax(np.stack([recon_z, knn_z, class_tp_z], axis=0), axis=0)
    for class_id in sorted(int(c) for c in np.unique(class_ids) if int(c) > 0):
        mask = class_ids == class_id
        lchi = mask & (probs <= low_conf_max) & (ious >= high_iou_min)
        rows.append(
            {
                "class_id": class_id,
                "class_name": category_name(categories, class_id),
                "count": int(mask.sum()),
                "lc_hi_count": int(lchi.sum()),
                "mean_iou": safe_float(ious[mask].mean()),
                "mean_prob": safe_float(probs[mask].mean()),
                "outlier_p95": percentile(outlier[mask], 95),
                "class_tp_knn_z_p95": percentile(class_tp_z[mask], 95),
                "lc_hi_outlier_max": percentile(outlier[lchi], 100),
                "lc_hi_class_tp_knn_z_max": percentile(class_tp_z[lchi], 100),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = np.load(args.full_cache)
    train_x = full["train_features_l2"].astype(np.float64)
    val_x = full["val_features_l2"].astype(np.float64)
    train_iou = full["train_best_iou"].astype(np.float64)
    val_iou = full["val_best_iou"].astype(np.float64)
    train_class = full["train_class_id"].astype(np.int64)
    val_class = full["val_class_id"].astype(np.int64)
    val_prob = full["val_matched_prob"].astype(np.float64)
    val_image_id = full["val_image_id"].astype(np.int64)
    val_proposal_score = full["val_proposal_score"].astype(np.float64)

    images, categories = load_coco_metadata(args.coco_json)
    scores = fit_pca_scores(train_x, val_x, pca_components=args.pca_components, knn_k=args.knn_k)
    class_tp_z, class_diag = class_tp_knn_scores(
        scores["train_pca"],  # type: ignore[arg-type]
        scores["val_pca"],  # type: ignore[arg-type]
        train_class,
        val_class,
        train_iou,
        high_iou_min=args.high_iou_min,
        knn_k=args.knn_k,
    )

    recon_z = scores["val_recon_z"]  # type: ignore[assignment]
    knn_z = scores["val_knn_z"]  # type: ignore[assignment]
    image_rows = aggregate_images(
        image_ids=val_image_id,
        class_ids=val_class,
        ious=val_iou,
        probs=val_prob,
        proposal_scores=val_proposal_score,
        recon_z=recon_z,
        knn_z=knn_z,
        class_tp_z=class_tp_z,
        images=images,
        categories=categories,
        low_conf_max=args.low_conf_max,
        high_iou_min=args.high_iou_min,
        low_iou_max=args.low_iou_max,
    )

    proposal_rows = [
        proposal_record(
            index=i,
            image_id=int(val_image_id[i]),
            class_id=int(val_class[i]),
            iou=float(val_iou[i]),
            prob=float(val_prob[i]),
            proposal_score=float(val_proposal_score[i]),
            recon_z=float(recon_z[i]),
            knn_z=float(knn_z[i]),
            class_tp_z=float(class_tp_z[i]),
            images=images,
            categories=categories,
        )
        for i in range(val_x.shape[0])
    ]

    lchi_mask = (val_prob <= args.low_conf_max) & (val_iou >= args.high_iou_min) & (val_class > 0)
    lchi_rows = [row for row, keep in zip(proposal_rows, lchi_mask) if bool(keep)]
    lc_li_mask = (val_prob <= args.low_conf_max) & (val_iou <= args.low_iou_max)
    hcli_mask = (val_prob > args.low_conf_max) & (val_iou <= args.low_iou_max) & (val_class > 0)

    def by_key(rows: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: -1e18 if row.get(key) is None else float(row.get(key)),
            reverse=reverse,
        )[: args.top_k]

    def by_proposal_outlier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: -1e18 if row.get("outlier_score") is None else float(row["outlier_score"]),
            reverse=True,
        )[: args.top_k]

    report = {
        "config": {
            "full_cache": str(args.full_cache),
            "lc_cache": str(args.lc_cache),
            "coco_json": str(args.coco_json),
            "pca_components": int(scores["pca_components"]),
            "pca_explained_variance": safe_float(scores["explained_variance"]),
            "knn_k": int(args.knn_k),
            "low_conf_max": float(args.low_conf_max),
            "high_iou_min": float(args.high_iou_min),
            "low_iou_max": float(args.low_iou_max),
        },
        "counts": {
            "train_proposals": int(train_x.shape[0]),
            "val_proposals": int(val_x.shape[0]),
            "val_images": int(len(np.unique(val_image_id))),
            "val_lc_hi": int(lchi_mask.sum()),
            "val_lc_li": int(lc_li_mask.sum()),
            "val_hc_li": int(hcli_mask.sum()),
        },
        "score_reference": {
            "val_recon_z_p95": percentile(recon_z, 95),
            "val_global_knn_z_p95": percentile(knn_z, 95),
            "val_class_tp_knn_z_p95": percentile(class_tp_z, 95),
            "lc_hi_class_tp_knn_z_p95": percentile(class_tp_z[lchi_mask], 95),
            "lc_hi_outlier_p95": percentile(
                np.nanmax(np.stack([recon_z[lchi_mask], knn_z[lchi_mask], class_tp_z[lchi_mask]], axis=0), axis=0),
                95,
            )
            if int(lchi_mask.sum()) > 0
            else None,
        },
        "class_tp_knn_diagnostics": class_diag,
        "category_summary": category_summary(
            class_ids=val_class,
            ious=val_iou,
            probs=val_prob,
            recon_z=recon_z,
            knn_z=knn_z,
            class_tp_z=class_tp_z,
            categories=categories,
            low_conf_max=args.low_conf_max,
            high_iou_min=args.high_iou_min,
        ),
        "top_images_by_lc_hi_outlier": by_key(
            [row for row in image_rows if int(row["lc_hi_count"]) > 0],
            "lc_hi_outlier_max",
        ),
        "top_images_by_class_tp_outlier": by_key(image_rows, "class_tp_knn_z_p95"),
        "top_images_by_global_outlier": by_key(image_rows, "outlier_p95"),
        "top_lc_hi_proposals": by_proposal_outlier(lchi_rows),
        "top_hc_li_proposals": by_proposal_outlier([row for row, keep in zip(proposal_rows, hcli_mask) if bool(keep)]),
        "top_lc_li_proposals": by_proposal_outlier([row for row, keep in zip(proposal_rows, lc_li_mask) if bool(keep)]),
    }

    report_path = args.out_dir / "penultimate_manifold_outlier_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_dir / "top_images_by_lc_hi_outlier.csv", report["top_images_by_lc_hi_outlier"])
    write_csv(args.out_dir / "top_images_by_global_outlier.csv", report["top_images_by_global_outlier"])
    write_csv(args.out_dir / "top_lc_hi_proposals.csv", report["top_lc_hi_proposals"])
    write_csv(args.out_dir / "category_summary.csv", report["category_summary"])

    print(json.dumps({"report": str(report_path), "counts": report["counts"], "config": report["config"]}, indent=2))


if __name__ == "__main__":
    main()
