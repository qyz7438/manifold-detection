r"""Plan B full experiment: adversarial patch attack + SpectralChordDefense.

Runs the attack/defense pipeline on the full Penn-Fudan validation set (or a
configurable subset) and performs a small ablation grid over patch size,
defense size, and anomaly threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tv_f
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan B full experiment")
    parser.add_argument("--config", default="spectral_detection_posttrain/configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="runs/canonical_baseline_10ep_gpu_20260616_bg/checkpoint_last.pth")
    parser.add_argument("--output-dir", default="runs/experiments/plan_b_defense/full")
    parser.add_argument("--n-images", type=int, default=-1,
                        help="Number of validation images; -1 means full validation set.")
    parser.add_argument("--defense-size", type=int, default=160)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--anomaly-threshold", type=float, default=2.5)
    parser.add_argument("--patch-size", type=int, default=80)
    parser.add_argument("--attack-steps", type=int, default=300)
    parser.add_argument("--attack-lr", type=float, default=0.5)
    parser.add_argument("--attack-momentum", type=float, default=0.9)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--eot-transforms", type=int, default=1)
    parser.add_argument("--use-detector-patch-attack", action="store_true", default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-ablations", action="store_true",
                        help="Run small ablation grid (takes longer).")
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


def build_detector_from_config(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
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


def run_model_on_single(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model([image.to(device)])[0]
    return {k: v.detach().cpu() for k, v in output.items()}


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


def build_spectral_chord_defense(
    defense_size: int,
    latent_dim: int,
    anomaly_threshold: float,
    device: torch.device,
) -> SpectralChordDefense:
    h = defense_size
    wf = defense_size // 2 + 1
    in_dim = h * wf
    manifold = ComplexSpectralManifold(
        in_dim=in_dim,
        latent_dim=min(latent_dim, in_dim),
        hidden_dim=in_dim,
    ).to(device)
    metric = AdaptiveRiemannianMetric(latent_dim=manifold.latent_dim, eps=1e-4).to(device)
    transport = ChordTransport(
        manifold=manifold,
        metric=metric,
        delta=0.15,
        lambda_step=0.3,
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


def fit_natural_model(defense: SpectralChordDefense, clean_images: list[torch.Tensor], device: torch.device) -> None:
    spectra = []
    for img in clean_images:
        spectra.append(torch.fft.rfft2(img.to(device).unsqueeze(0)))
    spectra_tensor = torch.cat(spectra, dim=0)
    defense.natural_model.fit(spectra_tensor)


def apply_defense(defense: SpectralChordDefense, image: torch.Tensor, defense_size: int, device: torch.device) -> torch.Tensor:
    c, h, w = image.shape
    small = tv_f.resize(image, (defense_size, defense_size), antialias=True)
    with torch.no_grad():
        purified_small = defense(small.to(device).unsqueeze(0)).squeeze(0).cpu()
    purified = tv_f.resize(purified_small, (h, w), antialias=True)
    return purified.clamp(0.0, 1.0)


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)


def draw_image_with_boxes(
    ax: plt.Axes,
    image: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    target: dict[str, Any] | None = None,
    score_threshold: float = 0.5,
    title: str = "",
) -> None:
    ax.imshow(tensor_to_numpy(image))
    scores = prediction.get("scores", torch.empty(0))
    labels = prediction.get("labels", torch.empty(0, dtype=torch.long))
    boxes = prediction.get("boxes", torch.empty(0, 4))
    mask = (scores >= score_threshold) & (labels == 1)
    for box in boxes[mask]:
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
        for box in target["boxes"].detach().cpu():
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    draw_image_with_boxes(axes[0], image_clean, pred_clean, target, title="Clean")
    draw_image_with_boxes(axes[1], image_adv, pred_adv, target, title="Adversarial")
    draw_image_with_boxes(axes[2], image_defended, pred_defended, target, title="Defended")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single_config(
    args: argparse.Namespace,
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
    device: torch.device,
    patch_size: int,
    defense_size: int,
    anomaly_threshold: float,
    attack_lr: float,
    attack_steps: int,
    output_subdir: Path,
) -> dict[str, Any]:
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Clean
    preds_clean = [run_model_on_single(model, img, device) for img in tqdm(images, desc="clean", leave=False)]
    ap50_clean = compute_ap50(preds_clean, targets)

    # Defense
    defense = build_spectral_chord_defense(defense_size, args.latent_dim, anomaly_threshold, device)
    clean_small = [tv_f.resize(img, (defense_size, defense_size), antialias=True) for img in images]
    fit_natural_model(defense, clean_small, device)

    images_clean_defended = [apply_defense(defense, img, defense_size, device) for img in tqdm(images, desc="clean defend", leave=False)]
    preds_clean_defended = [run_model_on_single(model, img, device) for img in tqdm(images_clean_defended, desc="clean defended infer", leave=False)]
    ap50_clean_defended = compute_ap50(preds_clean_defended, targets)

    # Attack
    if args.use_detector_patch_attack:
        attack = ObjectDetectorPatchAttack(
            model=model,
            device=device,
            target_label=1,
            patch_size=(patch_size, patch_size),
            max_iter=attack_steps,
            step_size=attack_lr,
            momentum=args.attack_momentum,
            tv_weight=args.tv_weight,
            eot_transforms=args.eot_transforms,
        )
    else:
        loss_fn = DetectionConfidenceLoss(model, device, target_label=1)
        location = (int(320 * 0.6), (320 - patch_size) // 2)
        attack = AdversarialPatchAttack(
            model_or_loss=loss_fn,
            patch_size=(patch_size, patch_size),
            location=location,
            max_iter=attack_steps,
            step_size=attack_lr,
            clamp_range=(0.0, 1.0),
            targeted=False,
            smooth_sigma=0.5,
            momentum=args.attack_momentum,
            random_init=True,
        )
    images_adv = [
        attack.attack(img, target_boxes=tgt.get("boxes", None))[0].clamp(0.0, 1.0)
        for img, tgt in tqdm(zip(images, targets), desc="attack", leave=False, total=len(images))
    ]
    preds_adv = [run_model_on_single(model, img, device) for img in tqdm(images_adv, desc="adv infer", leave=False)]
    ap50_adv = compute_ap50(preds_adv, targets)

    # Defended adv
    images_defended = [apply_defense(defense, img, defense_size, device) for img in tqdm(images_adv, desc="defend", leave=False)]
    preds_defended = [run_model_on_single(model, img, device) for img in tqdm(images_defended, desc="defended infer", leave=False)]
    ap50_defended = compute_ap50(preds_defended, targets)

    denom = ap50_clean - ap50_adv
    recovery_rate = float(np.clip((ap50_defended - ap50_adv) / denom, 0.0, 1.0)) if abs(denom) > 1e-8 else (1.0 if ap50_defended >= ap50_clean else 0.0)
    clean_drop = ap50_clean - ap50_clean_defended

    metrics = {
        "AP50_clean": round(float(ap50_clean), 6),
        "AP50_adv": round(float(ap50_adv), 6),
        "AP50_defended": round(float(ap50_defended), 6),
        "AP50_clean_defended": round(float(ap50_clean_defended), 6),
        "recovery_rate": round(float(recovery_rate), 6),
        "clean_drop": round(float(clean_drop), 6),
        "n_images": len(images),
        "config": {
            "patch_size": patch_size,
            "defense_size": defense_size,
            "anomaly_threshold": anomaly_threshold,
            "attack_lr": attack_lr,
            "attack_steps": attack_steps,
        },
    }
    save_json(metrics, output_subdir / "metrics.json")

    for i in range(min(3, len(images))):
        save_comparison_figure(
            output_subdir / "figures" / f"sample_{i:02d}.png",
            images[i], preds_clean[i],
            images_adv[i], preds_adv[i],
            images_defended[i], preds_defended[i],
            targets[i],
        )
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else resolve_device(config)
    print(f"Using device: {device}")

    model = load_model(config, args.checkpoint, device)

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
    val_indices = indices[split:]
    if args.n_images > 0:
        val_indices = val_indices[:args.n_images]

    images_320, targets_320 = [], []
    for idx in val_indices:
        img, tgt = dataset[idx]
        img_sq, tgt_sq = resize_image_and_target(img, tgt, (320, 320))
        images_320.append(img_sq)
        targets_320.append(tgt_sq)

    print(f"Running full experiment on {len(images_320)} validation images.")

    if args.run_ablations:
        configs = [
            (48, 128, 3.0),
            (80, 160, 2.5),
            (112, 224, 2.0),
        ]
    else:
        configs = [(args.patch_size, args.defense_size, args.anomaly_threshold)]

    all_results = []
    for patch_size, defense_size, anomaly_threshold in configs:
        print(f"\n=== config: patch={patch_size}, defense={defense_size}, thr={anomaly_threshold} ===")
        subdir = output_dir / f"p{patch_size}_d{defense_size}_t{anomaly_threshold}"
        metrics = run_single_config(
            args, model, images_320, targets_320, device,
            patch_size, defense_size, anomaly_threshold,
            args.attack_lr, args.attack_steps, subdir,
        )
        all_results.append(metrics)
        print(json.dumps(metrics, indent=2))

    summary = {
        "default_config": all_results[0] if all_results else {},
        "ablations": all_results,
        "n_images": len(images_320),
    }
    save_json(summary, output_dir / "metrics.json")
    print(f"\nAll results saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
