from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import round2129_nwpu_posttrain_smoke as round2129
from scripts.analyze_nwpu_roi_feature_dimensions import (
    evaluate_feature_space,
    fit_pca_transform,
)
from spectral_detection_posttrain.analysis.dimensionality import pca_dimensionality_summary
from spectral_detection_posttrain.analysis.raw_ifft_features import (
    crop_and_resize_boxes,
    penn_fudan_legacy_ifft_metric_bank,
    raw_ifft_feature_summary,
)
from spectral_detection_posttrain.rlvr.confidence_rescue import (
    ConfidenceRescueConfig,
    match_boxes_to_targets,
)
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze raw image crop iFFT features on NWPU AP75 LC-HI candidates.")
    parser.add_argument("--run-name", default="round2150_nwpu_raw_ifft_dim")
    parser.add_argument("--limit-train", type=int, default=100000)
    parser.add_argument("--limit-val", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-proposals", type=int, default=100)
    parser.add_argument("--rollout-score-threshold", type=float, default=0.001)
    parser.add_argument("--rollout-detections-per-img", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--rescue-low-conf-max", type=float, default=0.5)
    parser.add_argument("--rescue-high-conf-min", type=float, default=0.7)
    parser.add_argument("--rescue-high-iou-min", type=float, default=0.75)
    parser.add_argument("--rescue-low-iou-max", type=float, default=0.3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--legacy-crop-sizes", type=int, nargs="+", default=[7, 11, 15, 21, 64])
    return parser.parse_args()


def rescue_config_from_args(args) -> ConfidenceRescueConfig:
    return ConfidenceRescueConfig(
        low_conf_max=float(args.rescue_low_conf_max),
        high_conf_min=float(args.rescue_high_conf_min),
        high_iou_min=float(args.rescue_high_iou_min),
        low_iou_max=float(args.rescue_low_iou_max),
    )


@torch.no_grad()
def collect_split_features(model, loader, device: torch.device, args) -> dict[str, np.ndarray]:
    cfg = rescue_config_from_args(args)
    raw_ifft = []
    legacy_banks: dict[int, list[np.ndarray]] = {int(size): [] for size in args.legacy_crop_sizes}
    cls_summary = []
    roi_l2 = []
    labels = []
    best_ious = []
    label_probs = []
    rollout_scores = []
    class_ids = []
    image_ids = []
    proposal_boxes = []

    model.eval()
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for image, target, output in zip(images, targets, outputs):
            proposals, proposal_scores = round2129.select_rollout_proposals(output, args)
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
            matched_probs = round2129.matched_label_probabilities(class_logits, best_labels)
            low_conf = matched_probs <= float(cfg.low_conf_max)
            lchi = low_conf & (best_iou >= float(cfg.high_iou_min)) & (best_labels > 0)
            lcli = low_conf & (best_iou <= float(cfg.low_iou_max)) & (best_labels > 0)
            candidate = lchi | lcli
            if not candidate.any():
                continue

            selected_logits = class_logits[candidate].detach()
            probs = F.softmax(selected_logits, dim=1)
            object_probs = probs[:, 1:] if probs.shape[1] > 1 else probs
            top_values = torch.topk(object_probs, k=min(2, object_probs.shape[1]), dim=1).values
            top1 = top_values[:, 0]
            top2 = top_values[:, 1] if top_values.shape[1] > 1 else torch.zeros_like(top1)
            bg = probs[:, 0] if probs.shape[1] > 1 else torch.zeros_like(top1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            summary = torch.stack([bg, top1, top2, top1 - top2, entropy], dim=1)

            selected_boxes = proposals[candidate.cpu()].to(device)
            crops = crop_and_resize_boxes(image.to(device), selected_boxes, crop_size=int(args.crop_size))
            raw_ifft.append(raw_ifft_feature_summary(crops).cpu().numpy())
            for crop_size in args.legacy_crop_sizes:
                legacy_crops = crops if int(crop_size) == int(args.crop_size) else crop_and_resize_boxes(
                    image.to(device),
                    selected_boxes,
                    crop_size=int(crop_size),
                )
                legacy_banks[int(crop_size)].append(penn_fudan_legacy_ifft_metric_bank(legacy_crops).cpu().numpy())
            cls_summary.append(summary.cpu().numpy())
            roi_l2.append(F.normalize(box_features[candidate].detach(), p=2, dim=1).cpu().numpy())
            labels.append(lchi[candidate].detach().cpu().numpy().astype(bool))
            best_ious.append(best_iou[candidate].detach().cpu().numpy())
            label_probs.append(matched_probs[candidate].detach().cpu().numpy())
            rollout_scores.append(proposal_scores.to(device)[candidate].detach().cpu().numpy())
            class_ids.append(best_labels[candidate].detach().cpu().numpy())
            image_id = target.get("image_id", torch.tensor([-1]))
            image_value = int(image_id.flatten()[0].item()) if torch.is_tensor(image_id) else int(image_id)
            image_ids.append(np.full((int(candidate.sum().item()),), image_value, dtype=np.int64))
            proposal_boxes.append(selected_boxes.detach().cpu().numpy())

    if not labels:
        return {
            "raw_ifft": np.empty((0, 12), dtype=np.float32),
            **{f"legacy_ifft_{int(size)}": np.empty((0, 23), dtype=np.float32) for size in args.legacy_crop_sizes},
            "cls_summary": np.empty((0, 5), dtype=np.float32),
            "roi_l2": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=bool),
            "best_iou": np.empty((0,), dtype=np.float32),
            "label_probs": np.empty((0,), dtype=np.float32),
            "rollout_scores": np.empty((0,), dtype=np.float32),
            "class_ids": np.empty((0,), dtype=np.int64),
            "image_ids": np.empty((0,), dtype=np.int64),
            "proposal_boxes": np.empty((0, 4), dtype=np.float32),
        }
    return {
        "raw_ifft": np.concatenate(raw_ifft, axis=0),
        **{f"legacy_ifft_{int(size)}": np.concatenate(values, axis=0) for size, values in legacy_banks.items()},
        "cls_summary": np.concatenate(cls_summary, axis=0),
        "roi_l2": np.concatenate(roi_l2, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "best_iou": np.concatenate(best_ious, axis=0),
        "label_probs": np.concatenate(label_probs, axis=0),
        "rollout_scores": np.concatenate(rollout_scores, axis=0),
        "class_ids": np.concatenate(class_ids, axis=0),
        "image_ids": np.concatenate(image_ids, axis=0),
        "proposal_boxes": np.concatenate(proposal_boxes, axis=0),
    }


def zscore_pair(train_features: np.ndarray, val_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_features)
    return scaler.transform(train_features), scaler.transform(val_features)


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
    np.savez_compressed(
        run_dir / "candidate_raw_ifft_features.npz",
        **{f"train_{k}": v for k, v in train.items()},
        **{f"val_{k}": v for k, v in val.items()},
    )

    train_labels = train["labels"]
    val_labels = val["labels"]
    train_roi35, val_roi35, roi_explained = fit_pca_transform(train["roi_l2"], val["roi_l2"], 35)
    train_ifft_z, val_ifft_z = zscore_pair(train["raw_ifft"], val["raw_ifft"])
    train_cls_z, val_cls_z = zscore_pair(train["cls_summary"], val["cls_summary"])

    spaces = {
        "raw_ifft": (train["raw_ifft"], val["raw_ifft"]),
        "raw_ifft_z": (train_ifft_z, val_ifft_z),
        "cls_summary": (train["cls_summary"], val["cls_summary"]),
        "cls_summary_z": (train_cls_z, val_cls_z),
        "roi_l2_pca35": (train_roi35, val_roi35),
        "cls_summary_raw_ifft": (
            np.concatenate([train_cls_z, train_ifft_z], axis=1),
            np.concatenate([val_cls_z, val_ifft_z], axis=1),
        ),
        "cls_summary_roi35": (
            np.concatenate([train_cls_z, train_roi35], axis=1),
            np.concatenate([val_cls_z, val_roi35], axis=1),
        ),
        "cls_summary_roi35_raw_ifft": (
            np.concatenate([train_cls_z, train_roi35, train_ifft_z], axis=1),
            np.concatenate([val_cls_z, val_roi35, val_ifft_z], axis=1),
        ),
        "roi35_raw_ifft": (
            np.concatenate([train_roi35, train_ifft_z], axis=1),
            np.concatenate([val_roi35, val_ifft_z], axis=1),
        ),
    }

    legacy_feature_spaces = {}
    for crop_size in args.legacy_crop_sizes:
        key = f"legacy_ifft_{int(crop_size)}"
        train_legacy_z, val_legacy_z = zscore_pair(train[key], val[key])
        spaces[key] = (train[key], val[key])
        spaces[f"{key}_z"] = (train_legacy_z, val_legacy_z)
        spaces[f"cls_summary_{key}"] = (
            np.concatenate([train_cls_z, train_legacy_z], axis=1),
            np.concatenate([val_cls_z, val_legacy_z], axis=1),
        )
        spaces[f"cls_summary_roi35_{key}"] = (
            np.concatenate([train_cls_z, train_roi35, train_legacy_z], axis=1),
            np.concatenate([val_cls_z, val_roi35, val_legacy_z], axis=1),
        )
        legacy_feature_spaces[key] = {
            "crop_size": int(crop_size),
            "feature_dim": int(train[key].shape[1]),
        }

    report: dict[str, object] = {
        "device": str(device),
        "crop_size": int(args.crop_size),
        "roi_l2_pca35_explained_variance": roi_explained,
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
        "feature_spaces": {},
        "legacy_feature_spaces": legacy_feature_spaces,
        "raw_ifft_feature_names": [
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
        ],
        "legacy_ifft_feature_names": [
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
        ],
    }
    for name, (train_features, val_features) in spaces.items():
        report["feature_spaces"][name] = {
            "dimensionality": pca_dimensionality_summary(train_features, max_components=min(256, train_features.shape[1])),
            "separability": evaluate_feature_space(name, train_features, val_features, train_labels, val_labels),
        }

    save_json(report, run_dir / "raw_ifft_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
