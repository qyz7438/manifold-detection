from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.ops import nms
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch
from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.core.models import build_detector
from spectral_detection_posttrain.signals.fft.fft_features import compute_amplitude_profile, compute_sobel_structure_features
from spectral_detection_posttrain.signals.fft.roi_crop import crop_and_resize_roi
from spectral_detection_posttrain.utils.config import load_config, save_config
from spectral_detection_posttrain.utils.io import ensure_run_dir, load_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache detector candidates with ROI spectral evidence features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--patch-mode", default="none", choices=["none", "background", "object", "edge", "random", "object_inside", "object_edge", "near_object"])
    parser.add_argument("--patch-type", default=None, choices=["random", "checkerboard", "qr", "qr_like", "qr-like"])
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def _empty_candidate_sample(image_id: int, target: dict, amp_bins: int, structure_dim: int, roi_feature_dim: int) -> dict:
    return {
        "image_id": int(image_id),
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.long),
        "scores": torch.empty((0,), dtype=torch.float32),
        "roi_features": torch.empty((0, roi_feature_dim), dtype=torch.float32),
        "amp_profiles": torch.empty((0, amp_bins), dtype=torch.float32),
        "structure_features": torch.empty((0, structure_dim), dtype=torch.float32),
        "raw_r_amp": torch.empty((0,), dtype=torch.float32),
        "ious": torch.empty((0,), dtype=torch.float32),
        "is_tp": torch.empty((0,), dtype=torch.bool),
        "matched_gt_indices": torch.empty((0,), dtype=torch.long),
        "target": _cpu_target(target),
    }


def _cpu_target(target: dict) -> dict:
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}


def _scale_boxes_to_transformed_image(boxes: torch.Tensor, original_size: tuple[int, int], transformed_size: tuple[int, int]) -> torch.Tensor:
    orig_h, orig_w = original_size
    new_h, new_w = transformed_size
    scaled = boxes.clone()
    scaled[:, [0, 2]] *= new_w / float(orig_w)
    scaled[:, [1, 3]] *= new_h / float(orig_h)
    return scaled


@torch.no_grad()
def extract_roi_box_features(model: torch.nn.Module, image: torch.Tensor, boxes: torch.Tensor, device: torch.device) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    original_size = tuple(image.shape[-2:])
    transformed_images, _ = model.transform([image.to(device)], None)
    features = model.backbone(transformed_images.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    transformed_boxes = _scale_boxes_to_transformed_image(
        boxes.to(device),
        original_size=original_size,
        transformed_size=transformed_images.image_sizes[0],
    )
    pooled = model.roi_heads.box_roi_pool(features, [transformed_boxes], transformed_images.image_sizes)
    roi_features = model.roi_heads.box_head(pooled)
    return roi_features.detach().cpu()


def _cosine_r_amp(profile_pred: torch.Tensor, profile_gt: torch.Tensor) -> float:
    cosine = torch.nn.functional.cosine_similarity(profile_pred, profile_gt, dim=0).clamp(-1.0, 1.0)
    return float(torch.exp(-(1.0 - cosine)).item())


def _compute_candidate_features(
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    best_gt_indices: torch.Tensor,
    amp_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amp_profiles: list[torch.Tensor] = []
    structure_features: list[torch.Tensor] = []
    raw_r_amp: list[float] = []
    gt_amp_cache: dict[int, torch.Tensor] = {}

    for pred_box, gt_idx in zip(pred_boxes, best_gt_indices):
        pred_roi = crop_and_resize_roi(image, pred_box)
        amp_profile = compute_amplitude_profile(pred_roi, num_bins=amp_bins)
        structure = compute_sobel_structure_features(pred_roi)
        amp_profiles.append(amp_profile.cpu())
        structure_features.append(structure.cpu())

        gt_index = int(gt_idx.item())
        if gt_index >= 0 and len(gt_boxes) > 0:
            if gt_index not in gt_amp_cache:
                gt_roi = crop_and_resize_roi(image, gt_boxes[gt_index])
                gt_amp_cache[gt_index] = compute_amplitude_profile(gt_roi, num_bins=amp_bins).cpu()
            raw_r_amp.append(_cosine_r_amp(amp_profile.cpu(), gt_amp_cache[gt_index]))
        else:
            raw_r_amp.append(0.0)

    return (
        torch.stack(amp_profiles) if amp_profiles else torch.empty((0, amp_bins), dtype=torch.float32),
        torch.stack(structure_features) if structure_features else torch.empty((0, 8), dtype=torch.float32),
        torch.tensor(raw_r_amp, dtype=torch.float32),
    )


@torch.no_grad()
def build_candidate_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    prediction: dict,
    target: dict,
    device: torch.device,
    config: dict,
    max_candidates: int | None = None,
) -> dict:
    matching_cfg = config["matching"]
    quality_cfg = config.get("quality_head", {})
    amp_bins = int(quality_cfg.get("amp_bins", 32))
    structure_dim = 8
    image_id_tensor = target.get("image_id", torch.tensor([-1]))
    image_id = int(image_id_tensor.flatten()[0].item()) if torch.is_tensor(image_id_tensor) else int(image_id_tensor)

    pred_boxes = prediction.get("boxes", torch.empty((0, 4))).detach().cpu()
    pred_labels = prediction.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
    pred_scores = prediction.get("scores", torch.empty((0,))).detach().cpu()
    keep = pred_scores >= float(matching_cfg.get("score_threshold", 0.05))
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    if max_candidates is not None and len(pred_scores) > max_candidates:
        order = torch.argsort(pred_scores, descending=True)[:max_candidates]
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

    gt_boxes = target.get("boxes", torch.empty((0, 4))).detach().cpu()
    gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
    if len(pred_boxes) == 0:
        return _empty_candidate_sample(image_id, target, amp_bins, structure_dim, int(quality_cfg.get("roi_feature_dim", 1024)))

    filtered_prediction = {"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores}
    matched = match_predictions_to_gt(
        filtered_prediction,
        _cpu_target(target),
        iou_threshold=float(matching_cfg.get("iou_threshold", 0.5)),
        score_threshold=float(matching_cfg.get("score_threshold", 0.05)),
    )
    pred_to_gt = {int(match["pred_index"]): int(match["gt_index"]) for match in matched["matches"]}
    is_tp = torch.zeros((len(pred_boxes),), dtype=torch.bool)
    best_gt_indices = torch.full((len(pred_boxes),), -1, dtype=torch.long)
    best_ious = torch.zeros((len(pred_boxes),), dtype=torch.float32)
    if len(gt_boxes) > 0:
        ious = box_iou(pred_boxes, gt_boxes)
        best_ious, best_gt_indices = ious.max(dim=1)
        same_class = gt_labels[best_gt_indices] == pred_labels
        best_ious = torch.where(same_class, best_ious, torch.zeros_like(best_ious))
        best_gt_indices = torch.where(same_class, best_gt_indices, torch.full_like(best_gt_indices, -1))
    for pred_idx, gt_idx in pred_to_gt.items():
        is_tp[pred_idx] = True
        best_gt_indices[pred_idx] = gt_idx

    roi_features = extract_roi_box_features(model, image, pred_boxes, device)
    if roi_features.numel() == 0:
        roi_features = torch.empty((len(pred_boxes), int(quality_cfg.get("roi_feature_dim", 1024))), dtype=torch.float32)
    amp_profiles, structure_features, raw_r_amp = _compute_candidate_features(image, pred_boxes, gt_boxes, best_gt_indices, amp_bins)

    return {
        "image_id": image_id,
        "boxes": pred_boxes.float(),
        "labels": pred_labels.long(),
        "scores": pred_scores.float(),
        "roi_features": roi_features.float(),
        "amp_profiles": amp_profiles.float(),
        "structure_features": structure_features.float(),
        "raw_r_amp": raw_r_amp.float(),
        "ious": best_ious.float(),
        "is_tp": is_tp,
        "matched_gt_indices": best_gt_indices.long(),
        "target": _cpu_target(target),
    }


def save_candidate_cache(samples: list[dict], output_path: str | Path, config: dict, split: str, patch_mode: str, patch_type: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "split": split,
        "patch_mode": patch_mode,
        "patch_type": patch_type,
        "num_images": len(samples),
        "num_candidates": int(sum(len(sample["scores"]) for sample in samples)),
        "amp_bins": int(config.get("quality_head", {}).get("amp_bins", 32)),
        "structure_dim": 8,
        "roi_feature_dim": int(samples[0]["roi_features"].shape[1]) if samples and samples[0]["roi_features"].ndim == 2 else 0,
    }
    torch.save({"meta": meta, "samples": samples}, output)


def load_candidate_cache(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu")


class RoiSpectralCandidateDataset(Dataset):
    def __init__(self, cache_path: str | Path) -> None:
        payload = load_candidate_cache(cache_path)
        self.meta = payload["meta"]
        self.samples = payload["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def apply_nms_to_prediction(prediction: dict, iou_threshold: float = 0.5, score_threshold: float = 0.05) -> dict:
    boxes = prediction.get("boxes", torch.empty((0, 4))).detach().cpu()
    labels = prediction.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
    scores = prediction.get("scores", torch.empty((0,))).detach().cpu()
    keep_all: list[torch.Tensor] = []
    for label in labels.unique().tolist():
        label_mask = labels == int(label)
        label_indices = torch.nonzero(label_mask, as_tuple=False).flatten()
        label_keep = nms(boxes[label_mask], scores[label_mask], iou_threshold)
        keep_all.append(label_indices[label_keep])
    if keep_all:
        keep = torch.cat(keep_all)
        keep = keep[scores[keep] >= score_threshold]
        keep = keep[torch.argsort(scores[keep], descending=True)]
    else:
        keep = torch.empty((0,), dtype=torch.long)
    return {"boxes": boxes[keep], "labels": labels[keep], "scores": scores[keep]}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    if args.run_name:
        run_dir = ensure_run_dir(args.run_name)
        save_config(config, run_dir / "config.yaml")
        output_path = Path(args.output) if args.output else run_dir / f"{args.split}_candidates.pt"
    else:
        output_path = Path(args.output) if args.output else Path("runs") / f"{args.split}_candidates.pt"

    train_loader, val_loader = build_penn_fudan_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=1,
    )
    loader = train_loader if args.split == "train" else val_loader
    device = resolve_device(config)
    model_cfg = dict(config)
    model_cfg["model"] = dict(config["model"])
    model_cfg["model"]["pretrained"] = False
    model = build_detector(model_cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    patch_cfg = config.get("patch", {})
    patch_type = args.patch_type or str(patch_cfg.get("patch_type", "random"))
    samples = []
    for images, targets in tqdm(loader, desc=f"cache {args.split} candidates"):
        image = images[0]
        target = _cpu_target(targets[0])
        if args.patch_mode != "none":
            image = add_detection_patch(
                image,
                target,
                placement=args.patch_mode,
                patch_type=patch_type,
                patch_size=int(patch_cfg.get("patch_size", 48)),
            )
        output = model([image.to(device)])[0]
        prediction = {k: v.detach().cpu() for k, v in output.items()}
        sample = build_candidate_sample(
            model,
            image.cpu(),
            prediction,
            target,
            device,
            config,
            max_candidates=args.max_candidates or config.get("quality_head", {}).get("max_candidates_per_image"),
        )
        samples.append(sample)

    save_candidate_cache(samples, output_path, config, args.split, args.patch_mode, patch_type if args.patch_mode != "none" else "none")
    print({"output": str(output_path), "num_images": len(samples), "num_candidates": sum(len(sample["scores"]) for sample in samples)})


if __name__ == "__main__":
    main()
