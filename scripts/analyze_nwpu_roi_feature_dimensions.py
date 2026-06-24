from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import round2129_nwpu_posttrain_smoke as round2129
from spectral_detection_posttrain.analysis.dimensionality import (
    binary_ranking_metrics,
    center_margin_scores,
    pca_dimensionality_summary,
)
from spectral_detection_posttrain.rlvr.confidence_rescue import (
    ConfidenceRescueConfig,
    match_boxes_to_targets,
)
from spectral_detection_posttrain.rlvr.roi_policy_loss import resize_boxes_to_image
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze NWPU ROI feature dimensionality for real LC-HI rescue.")
    parser.add_argument("--run-name", default="round2145_nwpu_roi_dimensionality")
    parser.add_argument("--limit-train", type=int, default=100000)
    parser.add_argument("--limit-val", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=100)
    parser.add_argument("--rollout-score-threshold", type=float, default=0.001)
    parser.add_argument("--rollout-detections-per-img", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--rescue-low-conf-max", type=float, default=0.5)
    parser.add_argument("--rescue-high-conf-min", type=float, default=0.7)
    parser.add_argument("--rescue-high-iou-min", type=float, default=0.5)
    parser.add_argument("--rescue-low-iou-max", type=float, default=0.3)
    parser.add_argument("--pca-components", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--include-fft", action="store_true", help="Also analyze FFT spectra from ROI pooled features.")
    return parser.parse_args()


def rescue_config_from_args(args) -> ConfidenceRescueConfig:
    return ConfidenceRescueConfig(
        low_conf_max=float(args.rescue_low_conf_max),
        high_conf_min=float(args.rescue_high_conf_min),
        high_iou_min=float(args.rescue_high_iou_min),
        low_iou_max=float(args.rescue_low_iou_max),
    )


def extract_roi_outputs_features_and_pooled(model, images: list[torch.Tensor], boxes: list[torch.Tensor]):
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    transformed, _ = model.transform(images, None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    scaled_boxes = [
        resize_boxes_to_image(b.to(transformed.tensors.device), original, new)
        for b, original, new in zip(boxes, original_sizes, transformed.image_sizes)
    ]
    roi_features = model.roi_heads.box_roi_pool(features, scaled_boxes, transformed.image_sizes)
    box_features = model.roi_heads.box_head(roi_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    return class_logits, box_regression, box_features, roi_features, scaled_boxes, transformed.image_sizes


def fft_spectrum_features(roi_features: torch.Tensor) -> dict[str, torch.Tensor]:
    fft = torch.fft.rfft2(roi_features, dim=(-2, -1), norm="ortho")
    amp = torch.log1p(torch.abs(fft))
    phase = torch.angle(fft)

    freq_h = torch.fft.fftfreq(roi_features.shape[-2], device=roi_features.device)
    freq_w = torch.fft.rfftfreq(roi_features.shape[-1], device=roi_features.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x.pow(2) + grid_y.pow(2))
    radius = radius / radius.max().clamp_min(1e-6)
    masks = [
        (radius <= 0.3).float(),
        ((radius > 0.3) & (radius <= 0.7)).float(),
        (radius > 0.7).float(),
    ]

    def band_sum(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (tensor * mask).flatten(2).sum(dim=2)

    band_features = torch.cat(
        [band_sum(amp, mask) for mask in masks]
        + [band_sum(torch.sin(phase), mask) for mask in masks]
        + [band_sum(torch.cos(phase), mask) for mask in masks],
        dim=1,
    )
    return {
        "fft_amp": amp.flatten(1),
        "fft_band": band_features,
    }


@torch.no_grad()
def collect_split_features(model, loader, device: torch.device, args) -> dict[str, np.ndarray]:
    cfg = rescue_config_from_args(args)
    features = []
    features_l2 = []
    cls_logits = []
    cls_probs = []
    cls_summary = []
    bbox_regression_all = []
    bbox_pred_deltas = []
    final_head = []
    final_head_l2 = []
    fft_amp = []
    fft_amp_l2 = []
    fft_band = []
    fft_band_l2 = []
    labels = []
    best_ious = []
    rollout_scores = []
    label_probs = []
    class_ids = []
    image_ids = []

    model.eval()
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for image, target, output in zip(images, targets, outputs):
            proposals, proposal_scores = round2129.select_rollout_proposals(output, args)
            if proposals.numel() == 0:
                continue
            if bool(args.include_fft):
                class_logits, box_regression, box_features, roi_features, _, _ = extract_roi_outputs_features_and_pooled(
                    model,
                    [image.to(device)],
                    [proposals],
                )
            else:
                class_logits, box_regression, box_features, _, _ = round2129.extract_roi_outputs_and_features_for_boxes(
                    model,
                    [image.to(device)],
                    [proposals],
                )
                roi_features = None
            gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).to(device)
            best_iou, best_labels = match_boxes_to_targets(proposals.to(device), gt_boxes, gt_labels)
            matched_probs = round2129.matched_label_probabilities(class_logits, best_labels)
            low_conf = matched_probs <= float(cfg.low_conf_max)
            lchi = low_conf & (best_iou >= float(cfg.high_iou_min)) & (best_labels > 0)
            lcli = low_conf & (best_iou <= float(cfg.low_iou_max)) & (best_labels > 0)
            candidate = lchi | lcli
            if not candidate.any():
                continue
            selected_features = box_features[candidate].detach()
            features.append(selected_features.cpu().numpy())
            features_l2.append(F.normalize(selected_features, p=2, dim=1).cpu().numpy())
            selected_logits = class_logits[candidate].detach()
            selected_probs = F.softmax(selected_logits, dim=1)
            object_probs = selected_probs[:, 1:] if selected_probs.shape[1] > 1 else selected_probs
            topk = min(2, object_probs.shape[1])
            top_values = torch.topk(object_probs, k=topk, dim=1).values
            max_object_prob = top_values[:, 0]
            second_object_prob = top_values[:, 1] if topk > 1 else torch.zeros_like(max_object_prob)
            background_prob = selected_probs[:, 0] if selected_probs.shape[1] > 1 else torch.zeros_like(max_object_prob)
            entropy = -(selected_probs * selected_probs.clamp_min(1e-12).log()).sum(dim=1)
            selected_cls_summary = torch.stack(
                [
                    background_prob,
                    max_object_prob,
                    second_object_prob,
                    max_object_prob - second_object_prob,
                    entropy,
                ],
                dim=1,
            )
            selected_bbox_regression = box_regression[candidate].detach()
            selected_pred_labels = object_probs.argmax(dim=1) + 1 if selected_probs.shape[1] > 1 else torch.zeros_like(max_object_prob, dtype=torch.long)
            selected_pred_deltas = round2129.class_box_deltas(
                selected_bbox_regression,
                selected_pred_labels,
                round2129.NUM_CLASSES,
            )
            selected_final_head = torch.cat(
                [selected_logits, selected_probs, selected_cls_summary, selected_bbox_regression, selected_pred_deltas],
                dim=1,
            )
            cls_logits.append(selected_logits.cpu().numpy())
            cls_probs.append(selected_probs.cpu().numpy())
            cls_summary.append(selected_cls_summary.cpu().numpy())
            bbox_regression_all.append(selected_bbox_regression.cpu().numpy())
            bbox_pred_deltas.append(selected_pred_deltas.cpu().numpy())
            final_head.append(selected_final_head.cpu().numpy())
            final_head_l2.append(F.normalize(selected_final_head, p=2, dim=1).cpu().numpy())
            if roi_features is not None:
                spectra = fft_spectrum_features(roi_features[candidate].detach())
                selected_fft_amp = spectra["fft_amp"]
                selected_fft_band = spectra["fft_band"]
                fft_amp.append(selected_fft_amp.cpu().numpy())
                fft_amp_l2.append(F.normalize(selected_fft_amp, p=2, dim=1).cpu().numpy())
                fft_band.append(selected_fft_band.cpu().numpy())
                fft_band_l2.append(F.normalize(selected_fft_band, p=2, dim=1).cpu().numpy())
            labels.append(lchi[candidate].detach().cpu().numpy().astype(bool))
            best_ious.append(best_iou[candidate].detach().cpu().numpy())
            rollout_scores.append(proposal_scores.to(device)[candidate].detach().cpu().numpy())
            label_probs.append(matched_probs[candidate].detach().cpu().numpy())
            class_ids.append(best_labels[candidate].detach().cpu().numpy())
            image_id = target.get("image_id", torch.tensor([-1]))
            image_value = int(image_id.flatten()[0].item()) if torch.is_tensor(image_id) else int(image_id)
            image_ids.append(np.full((int(candidate.sum().item()),), image_value, dtype=np.int64))

    if not features:
        return {
            "features": np.empty((0, 0), dtype=np.float32),
            "features_l2": np.empty((0, 0), dtype=np.float32),
            "cls_logits": np.empty((0, 0), dtype=np.float32),
            "cls_probs": np.empty((0, 0), dtype=np.float32),
            "cls_summary": np.empty((0, 0), dtype=np.float32),
            "bbox_regression": np.empty((0, 0), dtype=np.float32),
            "bbox_pred_deltas": np.empty((0, 0), dtype=np.float32),
            "final_head": np.empty((0, 0), dtype=np.float32),
            "final_head_l2": np.empty((0, 0), dtype=np.float32),
            "fft_amp": np.empty((0, 0), dtype=np.float32),
            "fft_amp_l2": np.empty((0, 0), dtype=np.float32),
            "fft_band": np.empty((0, 0), dtype=np.float32),
            "fft_band_l2": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=bool),
            "best_iou": np.empty((0,), dtype=np.float32),
            "rollout_scores": np.empty((0,), dtype=np.float32),
            "label_probs": np.empty((0,), dtype=np.float32),
            "class_ids": np.empty((0,), dtype=np.int64),
            "image_ids": np.empty((0,), dtype=np.int64),
        }
    return {
        "features": np.concatenate(features, axis=0),
        "features_l2": np.concatenate(features_l2, axis=0),
        "cls_logits": np.concatenate(cls_logits, axis=0),
        "cls_probs": np.concatenate(cls_probs, axis=0),
        "cls_summary": np.concatenate(cls_summary, axis=0),
        "bbox_regression": np.concatenate(bbox_regression_all, axis=0),
        "bbox_pred_deltas": np.concatenate(bbox_pred_deltas, axis=0),
        "final_head": np.concatenate(final_head, axis=0),
        "final_head_l2": np.concatenate(final_head_l2, axis=0),
        "fft_amp": np.concatenate(fft_amp, axis=0) if fft_amp else np.empty((0, 0), dtype=np.float32),
        "fft_amp_l2": np.concatenate(fft_amp_l2, axis=0) if fft_amp_l2 else np.empty((0, 0), dtype=np.float32),
        "fft_band": np.concatenate(fft_band, axis=0) if fft_band else np.empty((0, 0), dtype=np.float32),
        "fft_band_l2": np.concatenate(fft_band_l2, axis=0) if fft_band_l2 else np.empty((0, 0), dtype=np.float32),
        "labels": np.concatenate(labels, axis=0),
        "best_iou": np.concatenate(best_ious, axis=0),
        "rollout_scores": np.concatenate(rollout_scores, axis=0),
        "label_probs": np.concatenate(label_probs, axis=0),
        "class_ids": np.concatenate(class_ids, axis=0),
        "image_ids": np.concatenate(image_ids, axis=0),
    }


def fit_pca_transform(train_features: np.ndarray, query_features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, float]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_components = min(int(n_components), train_features.shape[0] - 1, train_features.shape[1])
    if n_components <= 0:
        return np.empty((train_features.shape[0], 0)), np.empty((query_features.shape[0], 0)), 0.0
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    query_scaled = scaler.transform(query_features)
    pca = PCA(n_components=n_components, whiten=True, random_state=42).fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(query_scaled), float(pca.explained_variance_ratio_.sum())


def knn_density_ratio_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    k: int = 5,
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    positive = train_features[train_labels]
    negative = train_features[~train_labels]
    if positive.size == 0 or negative.size == 0:
        return np.zeros((query_features.shape[0],), dtype=np.float64)

    def mean_knn_distance(reference: np.ndarray) -> np.ndarray:
        k_eff = min(max(1, int(k)), reference.shape[0])
        neighbors = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        neighbors.fit(reference)
        distances, _ = neighbors.kneighbors(query_features, return_distance=True)
        return distances.mean(axis=1)

    return mean_knn_distance(negative) - mean_knn_distance(positive)


def evaluate_feature_space(name: str, train_features: np.ndarray, val_features: np.ndarray, train_labels: np.ndarray, val_labels: np.ndarray) -> dict[str, object]:
    report: dict[str, object] = {
        "name": name,
        "train_count": int(train_features.shape[0]),
        "val_count": int(val_features.shape[0]),
        "train_positive_count": int(train_labels.sum()),
        "val_positive_count": int(val_labels.sum()),
    }
    report["center_margin"] = binary_ranking_metrics(
        center_margin_scores(train_features, train_labels, val_features),
        val_labels,
    )
    report["knn_density_ratio"] = binary_ranking_metrics(
        knn_density_ratio_scores(train_features, train_labels, val_features, k=5),
        val_labels,
    )
    return report


def class_breakdown(labels: np.ndarray, class_ids: np.ndarray) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for class_id in sorted(set(class_ids.astype(int).tolist())):
        mask = class_ids == class_id
        output[str(int(class_id))] = {
            "candidate_count": int(mask.sum()),
            "positive_count": int((mask & labels).sum()),
            "negative_count": int((mask & (~labels)).sum()),
        }
    return output


def main():
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

    train = collect_split_features(model, train_loader, device, args)
    val = collect_split_features(model, val_loader, device, args)
    np.savez_compressed(run_dir / "candidate_features.npz", **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()})

    report: dict[str, object] = {
        "device": str(device),
        "train": {
            "candidate_count": int(train["labels"].shape[0]),
            "positive_count": int(train["labels"].sum()),
            "negative_count": int((~train["labels"]).sum()),
            "class_breakdown": class_breakdown(train["labels"], train["class_ids"]),
            "label_prob_mean": float(train["label_probs"].mean()) if train["label_probs"].size else 0.0,
        },
        "val": {
            "candidate_count": int(val["labels"].shape[0]),
            "positive_count": int(val["labels"].sum()),
            "negative_count": int((~val["labels"]).sum()),
            "class_breakdown": class_breakdown(val["labels"], val["class_ids"]),
            "label_prob_mean": float(val["label_probs"].mean()) if val["label_probs"].size else 0.0,
        },
        "feature_spaces": {},
        "pca_sweeps": {},
    }

    feature_keys = [
        "features",
        "features_l2",
        "cls_logits",
        "cls_probs",
        "cls_summary",
        "bbox_regression",
        "bbox_pred_deltas",
        "final_head",
        "final_head_l2",
    ]
    if bool(args.include_fft):
        feature_keys.extend(["fft_amp", "fft_amp_l2", "fft_band", "fft_band_l2"])

    for feature_key in feature_keys:
        if train[feature_key].size == 0 or val[feature_key].size == 0:
            continue
        report["feature_spaces"][feature_key] = {
            "dimensionality": pca_dimensionality_summary(train[feature_key], max_components=256),
            "separability": evaluate_feature_space(
                feature_key,
                train[feature_key],
                val[feature_key],
                train["labels"],
                val["labels"],
            ),
        }
        sweep = []
        for n_components in args.pca_components:
            train_pca, val_pca, explained = fit_pca_transform(train[feature_key], val[feature_key], int(n_components))
            if train_pca.shape[1] == 0:
                continue
            item = evaluate_feature_space(
                f"{feature_key}_pca{train_pca.shape[1]}",
                train_pca,
                val_pca,
                train["labels"],
                val["labels"],
            )
            item["components"] = int(train_pca.shape[1])
            item["explained_variance"] = explained
            sweep.append(item)
        report["pca_sweeps"][feature_key] = sweep

    save_json(report, run_dir / "dimension_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
