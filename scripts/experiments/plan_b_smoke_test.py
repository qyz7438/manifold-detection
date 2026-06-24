r"""Plan B smoke test: adversarial patch attack + SpectralChordDefense on Penn-Fudan.

This script runs a minimal end-to-end experiment on 10 validation images:
1. Clean detector evaluation (AP50).
2. Adversarial patch attack on each image.
3. SpectralChordDefense purification of adversarial images.
4. SpectralChordDefense purification of clean images (clean-sample loss).
5. Metrics and visualisation outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make project root importable regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as nn_f
import torchvision.transforms.functional as tv_f
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from spectral_detection_posttrain.datasets.penn_fudan import (
    PennFudanDetectionDataset,
    detection_collate,
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
from spectral_detection_posttrain.methods.defense.spectral_chord_defense import (
    SpectralChordDefense,
)
from spectral_detection_posttrain.methods.defense.manifold_natural import (
    NaturalSpectrumModel,
)
from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.chord_transport import ChordTransport
from spectral_detection_posttrain.methods.manifold.riemannian_metric import (
    AdaptiveRiemannianMetric,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import set_seed, resolve_device


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan B smoke test")
    parser.add_argument(
        "--config",
        default="spectral_detection_posttrain/configs/baseline.yaml",
        help="Path to detector config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/canonical_baseline_10ep_gpu_20260616_bg/checkpoint_last.pth",
        help="Path to detector checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/experiments/plan_b_defense/smoke_test",
        help="Directory for metrics and figures.",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=10,
        help="Number of validation images to use.",
    )
    parser.add_argument(
        "--defense-size",
        type=int,
        default=64,
        help="Spatial size used for SpectralChordDefense (smaller = faster).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Manifold latent dimension.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=3.0,
        help="Anomaly-detection z-score threshold.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=48,
        help="Adversarial patch height/width.",
    )
    parser.add_argument(
        "--attack-steps",
        type=int,
        default=75,
        help="PGD steps for patch optimisation.",
    )
    parser.add_argument(
        "--attack-lr",
        type=float,
        default=0.1,
        help="Patch PGD step size.",
    )
    parser.add_argument(
        "--attack-momentum",
        type=float,
        default=0.9,
        help="Patch PGD momentum.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.01,
        help="Total-variation smoothness weight.",
    )
    parser.add_argument(
        "--eot-transforms",
        type=int,
        default=1,
        help="EOT samples per step (1 disables EOT).",
    )
    parser.add_argument(
        "--use-detector-patch-attack",
        action="store_true",
        help="Use DPatch/RP2-style ObjectDetectorPatchAttack instead of generic patch attack.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cuda/cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Image / target utilities
# --------------------------------------------------------------------------- #


def resize_image_and_target(
    image: torch.Tensor,
    target: dict[str, Any],
    size: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resize image and scale boxes to the new spatial size."""
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


def build_detector_from_config(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build the detector described by the config and load the checkpoint."""
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
    model = build_detector_from_config(config, device)
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return model


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #


def batched_inference(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Run model inference on a list of images and move outputs to CPU."""
    with torch.no_grad():
        outputs = model([img.to(device) for img in images])
    return [{k: v.detach().cpu() for k, v in out.items()} for out in outputs]


def run_model_on_single(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run model on a single image and return the single output dict."""
    outputs = batched_inference(model, [image], device)
    return outputs[0]


def compute_ap50(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
) -> float:
    """Compute AP50 from predictions and ground-truth targets."""
    metrics = evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=0.5,
        score_threshold=0.05,
        high_conf_threshold=0.7,
    )
    return float(metrics.get("ap50", 0.0))


def filter_predictions(
    prediction: dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    target_label: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (boxes, scores) for a given label above a confidence threshold."""
    scores = prediction.get("scores", torch.empty(0))
    labels = prediction.get("labels", torch.empty(0, dtype=torch.long))
    boxes = prediction.get("boxes", torch.empty(0, 4))
    mask = (scores >= score_threshold) & (labels == target_label)
    return boxes[mask], scores[mask]


# --------------------------------------------------------------------------- #
# Attack loss wrapper
# --------------------------------------------------------------------------- #


class DetectionConfidenceLoss:
    """Callable that returns a scalar loss lower when detection is worse."""

    def __init__(self, model: torch.nn.Module, device: torch.device, target_label: int = 1):
        self.model = model
        self.device = device
        self.target_label = target_label
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Image is (C,H,W). Model expects a list of images.
        output = self.model([image.to(self.device)])[0]
        scores = output.get("scores", torch.empty(0, device=self.device))
        labels = output.get("labels", torch.empty(0, device=self.device, dtype=torch.long))
        person_scores = scores[labels == self.target_label]
        if person_scores.numel() > 0:
            # Strongly suppress the most confident detection and push all person
            # scores below the evaluation score threshold (0.05).
            max_score_loss = person_scores.max()
            threshold_penalty = (person_scores - 0.05).clamp(min=0.0).sum()
            return max_score_loss + 2.0 * threshold_penalty
        # Fallback: push patch toward black so the attack still degrades inputs
        # even when the detector already suppresses everything.
        return image.sum() * 1e-3


# --------------------------------------------------------------------------- #
# Defense construction
# --------------------------------------------------------------------------- #


def build_spectral_chord_defense(
    defense_size: int,
    latent_dim: int,
    anomaly_threshold: float,
    device: torch.device,
) -> SpectralChordDefense:
    """Build an untrained SpectralChordDefense with identity-like manifold."""
    # rfft2 of a square image of size S gives shape (S, S//2 + 1).
    h = defense_size
    wf = defense_size // 2 + 1
    in_dim = h * wf

    manifold = ComplexSpectralManifold(
        in_dim=in_dim,
        latent_dim=min(latent_dim, in_dim),
        hidden_dim=in_dim,  # required for exact square identity init
    ).to(device)

    metric = AdaptiveRiemannianMetric(
        latent_dim=manifold.latent_dim,
        eps=1e-4,
    ).to(device)

    transport = ChordTransport(
        manifold=manifold,
        metric=metric,
        delta=0.15,
        lambda_step=1.0,
    ).to(device)

    natural_model = NaturalSpectrumModel(reg=1e-6).to(device)

    defense = SpectralChordDefense(
        manifold=manifold,
        transport=transport,
        natural_model=natural_model,
        anomaly_gate_threshold=anomaly_threshold,
        window_size=5,
        preserve_dc=True,
    ).to(device)
    defense.eval()
    return defense


def fit_natural_model(
    defense: SpectralChordDefense,
    clean_images: list[torch.Tensor],
    device: torch.device,
) -> None:
    """Fit the natural spectrum model on the provided clean images."""
    spectra = []
    for img in clean_images:
        img_d = img.to(device).unsqueeze(0)
        spectra.append(torch.fft.rfft2(img_d))
    spectra_tensor = torch.cat(spectra, dim=0)
    defense.natural_model.fit(spectra_tensor)


def apply_defense(
    defense: SpectralChordDefense,
    image: torch.Tensor,
    defense_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize image to defense_size, purify, and resize back."""
    c, h, w = image.shape
    small = tv_f.resize(image, (defense_size, defense_size), antialias=True)
    with torch.no_grad():
        purified_small = defense(small.to(device).unsqueeze(0)).squeeze(0).cpu()
    purified = tv_f.resize(purified_small, (h, w), antialias=True)
    purified = purified.clamp(0.0, 1.0)
    return purified


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert a (C,H,W) tensor in [0,1] to a numpy (H,W,C) image."""
    return image.permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)


def draw_image_with_boxes(
    ax: plt.Axes,
    image: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    target: dict[str, Any] | None = None,
    score_threshold: float = 0.5,
    title: str = "",
) -> None:
    """Draw image with predicted (red) and ground-truth (green) boxes."""
    ax.imshow(tensor_to_numpy(image))
    pred_boxes, _ = filter_predictions(prediction, score_threshold=score_threshold)
    for box in pred_boxes:
        rect = patches.Rectangle(
            (box[0].item(), box[1].item()),
            (box[2] - box[0]).item(),
            (box[3] - box[1]).item(),
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    if target is not None and "boxes" in target:
        gt_boxes = target["boxes"].detach().cpu()
        for box in gt_boxes:
            rect = patches.Rectangle(
                (box[0].item(), box[1].item()),
                (box[2] - box[0]).item(),
                (box[3] - box[1]).item(),
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax.add_patch(rect)
    ax.set_title(title)
    ax.axis("off")


def save_comparison_figure(
    output_path: Path,
    image_clean: torch.Tensor,
    pred_clean: dict[str, torch.Tensor],
    image_adv: torch.Tensor,
    pred_adv: dict[str, torch.Tensor],
    image_defended: torch.Tensor,
    pred_defended: dict[str, torch.Tensor],
    target: dict[str, Any] | None,
) -> None:
    """Save a side-by-side clean/adv/defended figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    draw_image_with_boxes(
        axes[0], image_clean, pred_clean, target, title="Clean"
    )
    draw_image_with_boxes(
        axes[1], image_adv, pred_adv, target, title="Adversarial"
    )
    draw_image_with_boxes(
        axes[2],
        image_defended,
        pred_defended,
        target,
        title="Defended",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = (
        torch.device(args.device)
        if args.device
        else resolve_device(config)
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print("Loading detector...")
    model = load_model(config, args.checkpoint, device)
    model.eval()

    # ------------------------------------------------------------------ #
    # Load validation data (first N images, resized to square 320)
    # ------------------------------------------------------------------ #
    data_cfg = config.get("data", {})
    dataset = PennFudanDetectionDataset(
        root=data_cfg.get("root", "./data"),
        download=bool(data_cfg.get("download", True)),
        max_size=data_cfg.get("max_size", 320),
    )

    # Reproduce canonical train/val split deterministically.
    indices = list(range(len(dataset)))
    rng = np.random.default_rng(int(config.get("seed", 42)))
    rng.shuffle(indices)
    split = int(len(indices) * float(data_cfg.get("train_fraction", 0.8)))
    val_indices = indices[split: split + args.n_images]

    images_320: list[torch.Tensor] = []
    targets_320: list[dict[str, Any]] = []
    for idx in val_indices:
        img, tgt = dataset[idx]
        img_sq, tgt_sq = resize_image_and_target(img, tgt, (320, 320))
        images_320.append(img_sq)
        targets_320.append(tgt_sq)

    print(f"Running smoke test on {len(images_320)} images.")

    # ------------------------------------------------------------------ #
    # Clean evaluation
    # ------------------------------------------------------------------ #
    print("Evaluating on clean images...")
    preds_clean = [
        run_model_on_single(model, img, device) for img in tqdm(images_320, desc="clean")
    ]
    ap50_clean = compute_ap50(preds_clean, targets_320)
    print(f"AP50_clean = {ap50_clean:.4f}")

    # ------------------------------------------------------------------ #
    # Build and fit defense
    # ------------------------------------------------------------------ #
    print("Building SpectralChordDefense...")
    defense = build_spectral_chord_defense(
        defense_size=args.defense_size,
        latent_dim=args.latent_dim,
        anomaly_threshold=args.anomaly_threshold,
        device=device,
    )

    # Fit natural model on clean images at defense resolution.
    print("Fitting natural spectrum model...")
    clean_small = [
        tv_f.resize(img, (args.defense_size, args.defense_size), antialias=True)
        for img in images_320
    ]
    fit_natural_model(defense, clean_small, device)

    # ------------------------------------------------------------------ #
    # Clean + defense evaluation
    # ------------------------------------------------------------------ #
    print("Evaluating on clean + defended images...")
    images_clean_defended: list[torch.Tensor] = []
    for img in tqdm(images_320, desc="clean defend"):
        images_clean_defended.append(
            apply_defense(defense, img, args.defense_size, device)
        )
    preds_clean_defended = [
        run_model_on_single(model, img, device)
        for img in tqdm(images_clean_defended, desc="clean defended infer")
    ]
    ap50_clean_defended = compute_ap50(preds_clean_defended, targets_320)
    print(f"AP50_clean_defended = {ap50_clean_defended:.4f}")

    # ------------------------------------------------------------------ #
    # Adversarial patch attack
    # ------------------------------------------------------------------ #
    print("Running adversarial patch attack...")

    if args.use_detector_patch_attack:
        attack = ObjectDetectorPatchAttack(
            model=model,
            device=device,
            target_label=1,
            patch_size=(args.patch_size, args.patch_size),
            max_iter=args.attack_steps,
            step_size=args.attack_lr,
            momentum=args.attack_momentum,
            tv_weight=args.tv_weight,
            eot_transforms=args.eot_transforms,
        )
    else:
        loss_fn = DetectionConfidenceLoss(model, device, target_label=1)
        location_top = int(320 * 0.6)
        location_left = (320 - args.patch_size) // 2
        attack = AdversarialPatchAttack(
            model_or_loss=loss_fn,
            patch_size=(args.patch_size, args.patch_size),
            location=(location_top, location_left),
            max_iter=args.attack_steps,
            step_size=args.attack_lr,
            clamp_range=(0.0, 1.0),
            targeted=False,
            smooth_sigma=0.5,
            momentum=args.attack_momentum,
            random_init=True,
        )

    images_adv: list[torch.Tensor] = []
    for img, tgt in tqdm(zip(images_320, targets_320), desc="attack", total=len(images_320)):
        boxes = tgt.get("boxes", None)
        adv_img, _ = attack.attack(img, target_boxes=boxes)
        images_adv.append(adv_img.clamp(0.0, 1.0))

    # If using the generic AdversarialPatchAttack, it now also uses GT boxes.

    preds_adv = [
        run_model_on_single(model, img, device) for img in tqdm(images_adv, desc="adv infer")
    ]
    ap50_adv = compute_ap50(preds_adv, targets_320)
    print(f"AP50_adv = {ap50_adv:.4f}")

    # ------------------------------------------------------------------ #
    # Defense on adversarial images
    # ------------------------------------------------------------------ #
    print("Purifying adversarial images...")
    images_defended: list[torch.Tensor] = []
    for img in tqdm(images_adv, desc="defend"):
        images_defended.append(apply_defense(defense, img, args.defense_size, device))

    preds_defended = [
        run_model_on_single(model, img, device)
        for img in tqdm(images_defended, desc="defended infer")
    ]
    ap50_defended = compute_ap50(preds_defended, targets_320)
    print(f"AP50_defended = {ap50_defended:.4f}")

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    denom = ap50_clean - ap50_adv
    if abs(denom) < 1e-8:
        recovery_rate = 1.0 if ap50_defended >= ap50_clean else 0.0
    else:
        recovery_rate = (ap50_defended - ap50_adv) / denom
        recovery_rate = float(np.clip(recovery_rate, 0.0, 1.0))

    clean_drop = ap50_clean - ap50_clean_defended

    metrics = {
        "AP50_clean": round(float(ap50_clean), 6),
        "AP50_adv": round(float(ap50_adv), 6),
        "AP50_defended": round(float(ap50_defended), 6),
        "AP50_clean_defended": round(float(ap50_clean_defended), 6),
        "recovery_rate": round(recovery_rate, 6),
        "clean_drop": round(float(clean_drop), 6),
        "n_images": len(images_320),
        "config": {
            "checkpoint": str(args.checkpoint),
            "defense_size": args.defense_size,
            "latent_dim": args.latent_dim,
            "anomaly_threshold": args.anomaly_threshold,
            "patch_size": args.patch_size,
            "attack_steps": args.attack_steps,
            "attack_lr": args.attack_lr,
            "seed": args.seed,
        },
    }

    metrics_path = output_dir / "metrics.json"
    save_json(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    print("Saving comparison figures...")
    for i in range(min(3, len(images_320))):
        save_comparison_figure(
            figures_dir / f"sample_{i:02d}.png",
            images_320[i],
            preds_clean[i],
            images_adv[i],
            preds_adv[i],
            images_defended[i],
            preds_defended[i],
            targets_320[i],
        )
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
