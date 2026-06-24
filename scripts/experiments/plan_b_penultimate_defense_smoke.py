r"""Plan B smoke test: adversarial patch attack + penultimate-layer manifold defense.

This script validates the signal-first defense design:

1.  Fit a penultimate-layer manifold on clean training examples.
2.  Insert a phase-only + high-frequency AFM at the ROI-feature level.
3.  Insert a manifold gate at the box-head output.
4.  Measure clean / adversarial / defended / clean+defended AP50.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nn_f
import torchvision.transforms.functional as tv_f
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectral_detection_posttrain.datasets.penn_fudan import (
    PennFudanDetectionDataset,
)
from spectral_detection_posttrain.eval.detection_metrics import (
    evaluate_detection_predictions,
)
from spectral_detection_posttrain.methods.defense.patch_attack import (
    AdversarialPatchAttack,
)
from spectral_detection_posttrain.methods.defense.detector_patch_attack import (
    ObjectDetectorPatchAttack,
)
from spectral_detection_posttrain.methods.defense.penultimate_manifold_defense import (
    PenultimateManifoldDefense,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import set_seed, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan B penultimate manifold defense smoke test")
    parser.add_argument("--config", default="spectral_detection_posttrain/configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="runs/canonical_baseline_10ep_gpu_20260616_bg/checkpoint_last.pth")
    parser.add_argument("--output-dir", default="runs/experiments/plan_b_penultimate_defense/smoke")
    parser.add_argument("--n-images", type=int, default=10, help="Validation images for evaluation")
    parser.add_argument("--n-train-images", type=int, default=50, help="Clean training images to fit manifold")
    parser.add_argument("--patch-size", type=int, default=80)
    parser.add_argument("--attack-steps", type=int, default=300)
    parser.add_argument("--attack-lr", type=float, default=0.5)
    parser.add_argument("--attack-momentum", type=float, default=0.9)
    parser.add_argument("--gate-strength", type=float, default=0.6)
    parser.add_argument("--high-freq-ratio", type=float, default=0.3)
    parser.add_argument("--manifold-components", type=int, default=50)
    parser.add_argument("--manifold-neighbors", type=int, default=5)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--only-tp-manifold", action="store_true", default=True)
    parser.add_argument("--no-only-tp-manifold", dest="only_tp_manifold", action="store_false")
    parser.add_argument("--afm-only", action="store_true", help="Disable manifold gate, keep only AFM")
    parser.add_argument("--afm-type", default="mplseg_phase_only", help="AFM block type (mplseg_phase_only, phase_high_freq, none, ...)")
    parser.add_argument("--use-detector-patch-attack", action="store_true")
    parser.add_argument("--use-suppression-attack", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resize_image_and_target(
    image: torch.Tensor,
    target: dict[str, Any],
    size: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    c, orig_h, orig_w = image.shape
    new_h, new_w = size
    resized = tv_f.resize(image, size, antialias=True)
    new_target = dict(target)
    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] *= new_w / float(orig_w)
        boxes[:, [1, 3]] *= new_h / float(orig_h)
        new_target["boxes"] = boxes
    if "area" in new_target and len(new_target["area"]) > 0:
        new_target["area"] = (
            (new_target["boxes"][:, 3] - new_target["boxes"][:, 1]).clamp_min(0)
            * (new_target["boxes"][:, 2] - new_target["boxes"][:, 0]).clamp_min(0)
        )
    return resized, new_target


def build_detector(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 2))
    model_kwargs = {
        "min_size": int(model_cfg.get("min_size", 320)),
        "max_size": int(model_cfg.get("max_size", 320)),
    }
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, **model_kwargs
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.eval()
    return model


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_detector(config, device)
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return model


def run_model_on_single(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        outputs = model([image.to(device)])
    return {k: v.detach().cpu() for k, v in outputs[0].items()}


def compute_ap50(predictions: list[dict[str, torch.Tensor]], targets: list[dict[str, Any]]) -> float:
    metrics = evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=0.5,
        score_threshold=0.05,
        high_conf_threshold=0.7,
    )
    return float(metrics.get("ap50", 0.0))


class DetectionConfidenceLoss:
    def __init__(self, model: torch.nn.Module, device: torch.device, target_label: int = 1):
        self.model = model
        self.device = device
        self.target_label = target_label
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model([image.to(self.device)])[0]
        scores = output.get("scores", torch.empty(0, device=self.device))
        labels = output.get("labels", torch.empty(0, device=self.device, dtype=torch.long))
        person_scores = scores[labels == self.target_label]
        if person_scores.numel() > 0:
            max_score_loss = person_scores.max()
            threshold_penalty = (person_scores - 0.05).clamp(min=0.0).sum()
            return max_score_loss + 2.0 * threshold_penalty
        return image.sum() * 1e-3


class DetectionSuppressionLoss:
    """Loss that pushes person-class confidence down (disappearance attack)."""

    def __init__(self, model: torch.nn.Module, device: torch.device, target_label: int = 1):
        self.model = model
        self.device = device
        self.target_label = target_label
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model([image.to(self.device)])[0]
        scores = output.get("scores", torch.empty(0, device=self.device))
        labels = output.get("labels", torch.empty(0, device=self.device, dtype=torch.long))
        person_scores = scores[labels == self.target_label]
        if person_scores.numel() > 0:
            # Minimize the maximum person score and the number of above-threshold detections.
            return -person_scores.max() - 2.0 * (person_scores - 0.05).clamp(min=0.0).sum()
        # Already suppressed; still push patch away from natural to avoid trivial solutions.
        return -image.sum() * 1e-3


def collect_penultimate_features(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    device: torch.device,
    only_best_iou: bool = True,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Collect penultimate (box_head output) features from clean images.

    If ``only_best_iou`` is True, only keep features whose matched proposal has
    IoU >= ``iou_threshold`` with some ground-truth box.  This focuses the
    manifold on true-positive-like activations.
    """
    features: list[np.ndarray] = []

    captured_z: dict[str, torch.Tensor | None] = {"z": None}
    captured_proposals: list[torch.Tensor] = []

    def z_hook(module, inp, out):
        captured_z["z"] = out.detach().clone()

    def proposal_hook(module, args):
        captured_proposals.clear()
        captured_proposals.extend([a.clone() for a in args[1]])

    h_z = model.roi_heads.box_head.register_forward_hook(z_hook)
    h_p = model.roi_heads.box_roi_pool.register_forward_pre_hook(proposal_hook)

    for img, tgt in tqdm(zip(images, targets), total=len(images), desc="collect penultimate"):
        captured_z["z"] = None
        with torch.no_grad():
            model([img.to(device)])
        z = captured_z["z"]
        if z is None or z.shape[0] == 0:
            continue

        if only_best_iou and captured_proposals:
            proposals = torch.cat(captured_proposals, dim=0).to(device)
            gt_boxes = tgt["boxes"].to(device)
            if len(gt_boxes) > 0 and proposals.shape[0] == z.shape[0]:
                ious = box_iou(proposals, gt_boxes)
                best_iou, _ = ious.max(dim=1)
                keep = best_iou >= iou_threshold
                z = z[keep]

        features.append(z.cpu().numpy())

    h_z.remove()
    h_p.remove()
    if not features:
        raise RuntimeError("No penultimate features collected")
    return np.concatenate(features, axis=0)


def collect_anomaly_stats(
    defense: PenultimateManifoldDefense,
    images: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Collect penultimate anomaly scores for a set of images."""
    captured: dict[str, torch.Tensor | None] = {"z": None}

    def hook(module, inp, out):
        captured["z"] = out.detach().clone()

    handle = defense.detector.roi_heads.box_head.register_forward_hook(hook)
    scores_list: list[np.ndarray] = []
    for img in tqdm(images, desc="anomaly stats", leave=False):
        captured["z"] = None
        with torch.no_grad():
            defense([img.to(device)])
        z = captured["z"]
        if z is not None and z.shape[0] > 0:
            scores_list.append(defense.manifold.anomaly_score(z.cpu().numpy()))
    handle.remove()

    if not scores_list:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "frac_above": 0.0}
    all_scores = np.concatenate(scores_list)
    thr = defense.manifold.threshold or 1e9
    return {
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        "max": float(all_scores.max()),
        "frac_above": float((all_scores > thr).mean()),
    }


def build_attack(args: argparse.Namespace, model: torch.nn.Module, device: torch.device):
    if args.use_detector_patch_attack:
        return ObjectDetectorPatchAttack(
            model=model,
            device=device,
            target_label=1,
            patch_size=(args.patch_size, args.patch_size),
            max_iter=args.attack_steps,
            step_size=args.attack_lr,
            momentum=args.attack_momentum,
            tv_weight=0.01,
            eot_transforms=1,
        )
    if args.use_suppression_attack:
        loss_fn = DetectionSuppressionLoss(model, device, target_label=1)
    else:
        loss_fn = DetectionConfidenceLoss(model, device, target_label=1)
    return AdversarialPatchAttack(
        model_or_loss=loss_fn,
        patch_size=(args.patch_size, args.patch_size),
        location=(int(320 * 0.6), (320 - args.patch_size) // 2),
        max_iter=args.attack_steps,
        step_size=args.attack_lr,
        clamp_range=(0.0, 1.0),
        targeted=False,
        smooth_sigma=0.5,
        momentum=args.attack_momentum,
        random_init=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else resolve_device(config)
    print(f"Using device: {device}")

    print("Loading detector...")
    model = load_model(config, args.checkpoint, device)
    model.eval()

    data_cfg = config.get("data", {})
    dataset = PennFudanDetectionDataset(
        root=data_cfg.get("root", "./data"),
        download=bool(data_cfg.get("download", True)),
        max_size=data_cfg.get("max_size", 320),
    )

    indices = list(range(len(dataset)))
    rng = np.random.default_rng(int(config.get("seed", 42)))
    rng.shuffle(indices)
    split = int(len(indices) * float(data_cfg.get("train_fraction", 0.8)))
    train_indices = indices[:split]
    val_indices = indices[split: split + args.n_images]

    print(f"Fitting manifold on {args.n_train_images} clean training images...")
    train_images, train_targets = [], []
    for idx in train_indices[:args.n_train_images]:
        img, tgt = dataset[idx]
        img_sq, tgt_sq = resize_image_and_target(img, tgt, (320, 320))
        train_images.append(img_sq)
        train_targets.append(tgt_sq)

    penultimate_features = collect_penultimate_features(
        model, train_images, train_targets, device, only_best_iou=args.only_tp_manifold
    )
    label = "TP" if args.only_tp_manifold else "all-proposal"
    print(f"Collected {penultimate_features.shape[0]} {label} penultimate vectors, dim={penultimate_features.shape[1]}")

    print("Building defense...")
    defense = PenultimateManifoldDefense(
        detector=model,
        afm_channels=256,
        afm_type=args.afm_type,
        gate_strength=args.gate_strength,
        high_freq_ratio=args.high_freq_ratio,
        manifold_components=args.manifold_components,
        manifold_neighbors=args.manifold_neighbors,
        enable_manifold_gate=not args.afm_only,
    )
    defense = defense.to(device)
    defense.fit_manifold(penultimate_features, threshold_percentile=args.threshold_percentile)
    defense.eval()
    print(f"Manifold threshold (distance): {defense.manifold.threshold:.4f}")

    val_images, val_targets = [], []
    for idx in val_indices:
        img, tgt = dataset[idx]
        img_sq, tgt_sq = resize_image_and_target(img, tgt, (320, 320))
        val_images.append(img_sq)
        val_targets.append(tgt_sq)

    print("Evaluating clean...")
    preds_clean = [run_model_on_single(defense, img, device) for img in tqdm(val_images, desc="clean")]
    ap50_clean = compute_ap50(preds_clean, val_targets)
    print(f"AP50_clean = {ap50_clean:.4f}")

    print("Evaluating clean+defended...")
    preds_clean_defended = [run_model_on_single(defense, img, device) for img in tqdm(val_images, desc="clean defended")]
    ap50_clean_defended = compute_ap50(preds_clean_defended, val_targets)
    print(f"AP50_clean_defended = {ap50_clean_defended:.4f}")

    print("Running attack on original detector...")
    attack = build_attack(args, model, device)
    images_adv = []
    for img, tgt in tqdm(zip(val_images, val_targets), total=len(val_images), desc="attack"):
        adv_img, _ = attack.attack(img, target_boxes=tgt.get("boxes"))
        images_adv.append(adv_img.clamp(0.0, 1.0))

    preds_adv = [run_model_on_single(defense, img, device) for img in tqdm(images_adv, desc="adv")]
    ap50_adv = compute_ap50(preds_adv, val_targets)
    print(f"AP50_adv = {ap50_adv:.4f}")

    preds_defended = [run_model_on_single(defense, img, device) for img in tqdm(images_adv, desc="defended")]
    ap50_defended = compute_ap50(preds_defended, val_targets)
    print(f"AP50_defended = {ap50_defended:.4f}")

    print("Computing anomaly-score diagnostics...")
    clean_stats = collect_anomaly_stats(defense, val_images, device)
    adv_stats = collect_anomaly_stats(defense, images_adv, device)
    print(f"Clean scores: mean={clean_stats['mean']:.3f} std={clean_stats['std']:.3f} max={clean_stats['max']:.3f} above_thr={clean_stats['frac_above']:.3f}")
    print(f"Adv   scores: mean={adv_stats['mean']:.3f} std={adv_stats['std']:.3f} max={adv_stats['max']:.3f} above_thr={adv_stats['frac_above']:.3f}")

    denom = ap50_clean - ap50_adv
    recovery_rate = (
        1.0 if abs(denom) < 1e-8 else float(np.clip((ap50_defended - ap50_adv) / denom, 0.0, 1.0))
    )
    clean_drop = ap50_clean - ap50_clean_defended

    metrics = {
        "AP50_clean": round(float(ap50_clean), 6),
        "AP50_adv": round(float(ap50_adv), 6),
        "AP50_defended": round(float(ap50_defended), 6),
        "AP50_clean_defended": round(float(ap50_clean_defended), 6),
        "recovery_rate": round(recovery_rate, 6),
        "clean_drop": round(float(clean_drop), 6),
        "n_images": len(val_images),
        "n_train": len(train_images),
        "manifold_threshold": float(defense.manifold.threshold or 0.0),
        "anomaly_clean": clean_stats,
        "anomaly_adv": adv_stats,
        "config": {
            "checkpoint": str(args.checkpoint),
            "gate_strength": args.gate_strength,
            "high_freq_ratio": args.high_freq_ratio,
            "manifold_components": args.manifold_components,
            "manifold_neighbors": args.manifold_neighbors,
            "patch_size": args.patch_size,
            "attack_steps": args.attack_steps,
            "attack_lr": args.attack_lr,
        },
    }

    save_json(metrics, output_dir / "metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
