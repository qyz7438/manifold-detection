r"""Post-train detector box features with MGL-OPT manifold guidance.

This trainer loads a pretrained Faster R-CNN detector, freezes the backbone
and RPN, and optionally fine-tunes the box head / box predictor together with
the manifold modules.  The key fix for avoiding the AP50 collapse observed in
early experiments is to apply the manifold loss on the **same RPN proposals**
that the detector uses during training, rather than on ground-truth boxes
only.  This keeps the feature distribution consistent with inference and lets
``L_det`` act as the behavior anchor.

Usage (conservative, recommended first try):
    python -m spectral_detection_posttrain.trainers.detection.train_manifold_posttrain \
        --config configs/mvp.yaml \
        --baseline runs/mvp_pf_baseline/checkpoint_last.pth \
        --run-name mglopt_proposals \
        --lr 1e-5 --lr-manifold 1e-4 \
        --lambda-tr 0.01 --lambda-en 0.001 \
        --epochs 5
"""

from __future__ import annotations

import argparse
import copy
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectral_detection_posttrain.core.models.build_detector import (
    freeze_backbone,
    freeze_box_head,
    freeze_box_predictor,
    freeze_rpn,
)
from spectral_detection_posttrain.core.models.box_heads import (
    get_box_head_type,
    replace_box_head,
)
from spectral_detection_posttrain.core.models.etf_predictor import (
    ETFClassifier,
    replace_cls_score_with_etf,
)
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.experiments.canonical_runner import (
    build_experiment_model,
    checkpoint_metadata,
    prepare_experiment_from_config,
)
from spectral_detection_posttrain.methods.manifold import (
    IntrinsicDimEstimator,
    ManifoldCorrectionPredictor,
    PrototypeBank,
    RemoteSensingPrototypeBank,
    SinkhornAssigner,
    TransportHead,
    compute_class_frequency_weights,
)
from spectral_detection_posttrain.methods.manifold.geometry_metrics import (
    compute_manifold_geometry,
    scalar_geometry_report,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import append_jsonl, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train detector with MGL-OPT manifold guidance.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True, help="Path to pretrained detector checkpoint.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--num-prototypes", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--sinkhorn-eps", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iter", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lambda-tr", type=float, default=0.01)
    parser.add_argument("--lambda-en", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for detector box head / predictor (if trainable).")
    parser.add_argument("--lr-manifold", type=float, default=1e-4,
                        help="Learning rate for manifold modules (transport head).")
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-batches", type=int, default=5,
                        help="Number of batches used to warm-start prototypes from class centers.")
    parser.add_argument("--freeze-prototypes-after-warmup", action="store_true",
                        help="Keep prototype endpoints fixed after the optional warmup pass.")
    parser.add_argument("--normalize-features", action="store_true", default=True,
                        help="L2-normalize box features before manifold loss.")
    parser.add_argument("--no-normalize-features", action="store_true",
                        help="Disable L2 normalization of box features.")
    parser.add_argument("--geometry-every", type=int, default=50,
                        help="Compute geometry metrics every N steps (0 to disable).")
    parser.add_argument("--geometry-buffer-size", type=int, default=4096,
                        help="Max foreground features to accumulate for epoch-end geometry metrics.")
    parser.add_argument("--id-method", default="twonn", choices=["pca", "twonn"],
                        help="Intrinsic-dimension estimator for geometry logging (default: twonn).")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Run validation AP50 every N epochs.")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop if val AP50 does not improve for N evals.")
    parser.add_argument("--use-gt-boxes", action="store_true",
                        help="Use GT boxes for manifold loss (legacy mode, not recommended).")
    parser.add_argument("--freeze-box-head", action="store_true",
                        help="Freeze the Faster R-CNN box_head (most conservative).")
    parser.add_argument("--freeze-box-predictor", action="store_true",
                        help="Freeze the Faster R-CNN box_predictor.")
    parser.add_argument("--manifold-warmup-epochs", type=int, default=0,
                        help="Delay manifold loss for N epochs; only detection loss is used before that.")
    parser.add_argument("--active-manifold-correction", action="store_true",
                        help="Insert prototype-guided feature correction before the ROI box predictor.")
    parser.add_argument("--active-correction-gamma", type=float, default=0.05,
                        help="Residual strength for active feature correction.")
    parser.add_argument("--active-correction-gamma-schedule", default="constant",
                        choices=("constant", "linear_decay"),
                        help="Schedule for active correction gamma across post-training epochs.")
    parser.add_argument("--active-correction-gamma-final", type=float, default=None,
                        help="Final gamma for non-constant active correction schedules.")
    parser.add_argument("--active-correction-mode", default="residual",
                        choices=("residual", "endpoint", "gated_endpoint", "gated-endpoint"),
                        help="Active correction field: learned residual, direct endpoint, or gated endpoint.")
    parser.add_argument("--active-endpoint-gate-init", type=float, default=0.25,
                        help="Initial sigmoid gate value for gated endpoint correction.")
    parser.add_argument("--active-correction-normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Use L2-normalized features for active correction prototype distances.")
    parser.add_argument("--box-head-type", default="original",
                        choices=("original", "conv_lowdim", "conv-lowdim", "bottleneck", "bottleneck_twomlp",
                                 "bottleneck-twomlp", "attention_pool", "attention-pool"),
                        help="Replace the Faster R-CNN box_head with a low-dim-preserving variant.")
    parser.add_argument("--box-head-conv-channels", type=int, default=128,
                        help="Channels for conv_lowdim / bottleneck_twomlp heads.")
    parser.add_argument("--box-head-bottleneck-dim", type=int, default=512,
                        help="Hidden dim for conv_lowdim head.")
    parser.add_argument("--box-head-rank", type=int, default=128,
                        help="Rank for bottleneck (low-rank skip) head.")
    parser.add_argument("--box-head-attention-channels", type=int, default=64,
                        help="Intermediate channels for attention_pool head.")
    parser.add_argument("--rs-orient-bins", type=int, default=1,
                        help="Number of orientation bins for RemoteSensingPrototypeBank (1=disabled).")
    parser.add_argument("--rs-scale-bins", type=int, default=1,
                        help="Number of scale bins for RemoteSensingPrototypeBank (1=disabled).")
    parser.add_argument("--class-reweight", default="none",
                        choices=("none", "inv_sqrt", "effective_num"),
                        help="Class reweighting scheme for prototype updates and manifold losses.")
    parser.add_argument("--class-reweight-beta", type=float, default=0.999,
                        help="Beta for effective_num class reweighting.")
    parser.add_argument("--use-etf-classifier", action="store_true",
                        help="Replace the box_predictor cls_score with a fixed ETF classifier.")
    parser.add_argument("--lambda-fc1-rank", type=float, default=0.0,
                        help="Weight for the fc1 spectral tail loss. Default keeps the loss disabled.")
    parser.add_argument("--lambda-fc1-compact", type=float, default=0.0,
                        help="Weight for the fc1 supervised compactness loss. Default keeps the loss disabled.")
    parser.add_argument("--fc1-rank-target", type=int, default=16,
                        help="Mini-batch rank kept before penalizing the fc1 spectral tail.")
    parser.add_argument("--lambda-logit-preserve", type=float, default=0.0,
                        help="KL anchor to the baseline ROI classifier logits.")
    parser.add_argument("--lambda-bbox-preserve", type=float, default=0.0,
                        help="SmoothL1 anchor to the baseline foreground class bbox deltas.")
    parser.add_argument("--preserve-temperature", type=float, default=2.0,
                        help="Temperature for the baseline logit preservation KL.")
    parser.add_argument("--lambda-proj-intra", type=float, default=0.0,
                        help="Weight for same-class projection endpoints with similar prototype assignment.")
    parser.add_argument("--lambda-proto-div", type=float, default=0.0,
                        help="Weight for monitoring/penalizing collapsed same-class prototypes.")
    parser.add_argument("--lambda-proj-inter", type=float, default=0.0,
                        help="Weight for projection endpoint margin against other class manifolds.")
    parser.add_argument("--projection-inter-margin", type=float, default=0.5,
                        help="Margin for separating a projected endpoint from wrong class prototypes.")
    parser.add_argument("--proto-div-temperature", type=float, default=0.1,
                        help="Temperature for the same-class prototype diversity penalty.")
    parser.add_argument("--lambda-corrected-intra", type=float, default=0.0,
                        help="Weight for compacting the actual active-corrected foreground features.")
    parser.add_argument("--lambda-corrected-inter", type=float, default=0.0,
                        help="Weight for separating active-corrected features from wrong class prototypes.")
    parser.add_argument("--lambda-corrected-inter-preserve", type=float, default=0.0,
                        help="Weight for preserving pre-correction class-centroid separation.")
    parser.add_argument("--lambda-corrected-center-preserve", type=float, default=0.0,
                        help="Weight for preserving each pre-correction class centroid position.")
    parser.add_argument("--lambda-corrected-memory-center-preserve", type=float, default=0.0,
                        help="Weight for preserving corrected class centroids against an EMA class memory.")
    parser.add_argument("--lambda-corrected-memory-inter-preserve", type=float, default=0.0,
                        help="Weight for preserving EMA class-memory inter-centroid distances after correction.")
    parser.add_argument("--lambda-correction-field-preserve", type=float, default=0.0,
                        help="Weight for preserving the warmup active correction field during post-training.")
    parser.add_argument("--corrected-memory-momentum", type=float, default=0.9,
                        help="EMA momentum for the corrected class-centroid memory anchor.")
    parser.add_argument("--corrected-inter-margin", type=float, default=0.5,
                        help="Wrong-class prototype margin for active-corrected features.")
    return parser.parse_args()


def _to_device_targets(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()} for target in targets]


def _set_model_train_for_detection_loss(model: nn.Module) -> None:
    """Put model in training mode for the detection-loss forward pass while
    keeping all BatchNorm layers in eval mode to avoid running-stat drift in
    frozen backbones."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def resize_boxes_to_image(
    boxes: torch.Tensor, original_size: tuple[int, int], new_size: tuple[int, int]
) -> torch.Tensor:
    ratio_h = float(new_size[0]) / float(original_size[0])
    ratio_w = float(new_size[1]) / float(original_size[1])
    ratios = boxes.new_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
    return boxes * ratios


def _resolve_box_head_layers(box_head: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    """Return torchvision-style fc1/fc2 modules, including simple wrappers."""
    fc1 = getattr(box_head, "fc6", None)
    fc2 = getattr(box_head, "fc7", None)
    if fc1 is None or fc2 is None:
        inner = getattr(box_head, "head", None)
        if inner is not None:
            fc1 = getattr(inner, "fc6", fc1)
            fc2 = getattr(inner, "fc7", fc2)
    return fc1, fc2


def _box_head_layer_features(model: nn.Module, roi_pooled: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract intermediate ROI-head features without changing predictor behavior."""
    layers = {"roi_pooled": roi_pooled}
    fc1, _ = _resolve_box_head_layers(model.roi_heads.box_head)
    if fc1 is not None:
        layers["fc1"] = F.relu(fc1(roi_pooled.flatten(start_dim=1)))
    return layers


def extract_gt_box_features(
    model: nn.Module,
    images: list[torch.Tensor],
    targets: list[dict],
    return_layers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | tuple[
    torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]
]:
    """Extract box features for GT boxes and return (features, labels, scaled_boxes)."""
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    transformed, _ = model.transform(images, None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    gt_boxes = [t["boxes"] for t in targets]
    labels = torch.cat([t["labels"] for t in targets], dim=0)

    scaled_boxes = [
        resize_boxes_to_image(b.to(transformed.tensors.device), original, new)
        for b, original, new in zip(gt_boxes, original_sizes, transformed.image_sizes)
    ]

    box_features = model.roi_heads.box_roi_pool(features, scaled_boxes, transformed.image_sizes)
    layer_features = _box_head_layer_features(model, box_features) if return_layers else {}
    box_features = model.roi_heads.box_head(box_features)
    if return_layers:
        return box_features, labels, scaled_boxes, layer_features
    return box_features, labels, scaled_boxes


def extract_proposal_box_features(
    model: nn.Module,
    images: list[torch.Tensor],
    targets: list[dict],
    return_layers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | tuple[
    torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]
]:
    """Extract box features for RPN training proposals.

    Uses the same sampling/labeling path as ``RoIHeads.select_training_samples``,
    so the manifold loss sees the same proposal distribution as the detector's
    own classification head.  Labels follow Faster R-CNN convention:
    0 = background, 1..C = foreground, -1 = ignored.
    """
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    transformed, _ = model.transform(images, None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    proposals, _ = model.rpn(transformed, features, targets)

    # Use the detector's own matcher and sampler to stay distribution-aligned.
    sampled_props, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(
        proposals, targets
    )

    scaled_boxes = [
        resize_boxes_to_image(b.to(transformed.tensors.device), original, new)
        for b, original, new in zip(sampled_props, original_sizes, transformed.image_sizes)
    ]

    box_features = model.roi_heads.box_roi_pool(features, scaled_boxes, transformed.image_sizes)
    layer_features = _box_head_layer_features(model, box_features) if return_layers else {}
    box_features = model.roi_heads.box_head(box_features)
    labels = torch.cat(labels, dim=0)
    if return_layers:
        return box_features, labels, scaled_boxes, layer_features
    return box_features, labels, scaled_boxes


def warmup_class_centers(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: int,
    normalize: bool,
    use_gt_boxes: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect box features over a few batches and compute per-class means/counts."""
    sums = [torch.zeros(1, device=device) for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]
    dims = None

    extractor = extract_gt_box_features if use_gt_boxes else extract_proposal_box_features

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = [img.to(device) for img in images]
            targets = _to_device_targets(targets, device)
            box_features, labels, _ = extractor(model, images, targets)
            if box_features.shape[0] == 0:
                continue
            if normalize:
                box_features = F.normalize(box_features, dim=-1)
            if dims is None:
                dims = box_features.shape[-1]
                sums = [torch.zeros(dims, device=device) for _ in range(num_classes)]

            # Use only foreground proposals for class centers (labels >= 1).
            fg_mask = labels >= 1
            box_features = box_features[fg_mask]
            labels = labels[fg_mask]
            if labels.numel() == 0:
                continue

            for c in range(1, num_classes):
                mask = labels == c
                if mask.any():
                    sums[c] = sums[c] + box_features[mask].sum(dim=0)
                    counts[c] += int(mask.sum().item())

    centers = torch.stack([
        (sums[c] / max(1, counts[c])) if counts[c] > 0 else torch.zeros(dims or 1, device=device)
        for c in range(num_classes)
    ])
    counts_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    return centers, counts_tensor


def initialize_prototypes_from_centers_with_seed(
    prototype_bank: PrototypeBank,
    centers: torch.Tensor,
    *,
    noise_scale: float,
    seed: int,
) -> None:
    set_seed(int(seed))
    prototype_bank.initialize_from_centers(centers, noise_scale=noise_scale)


class ClassCentroidMemory(nn.Module):
    """EMA foreground class centers used as a batch-stable geometry anchor."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        *,
        momentum: float = 0.9,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.register_buffer("centers", torch.zeros(num_classes, feature_dim, device=device, dtype=dtype))
        self.register_buffer("initialized", torch.zeros(num_classes, device=device, dtype=torch.bool))
        self.register_buffer("counts", torch.zeros(num_classes, device=device, dtype=torch.long))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor, *, normalize: bool) -> None:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape (N, {self.feature_dim})")
        if labels.shape != features.shape[:1]:
            raise ValueError("labels must match the feature batch dimension")
        if features.shape[0] == 0:
            return

        reference = F.normalize(features.detach(), dim=-1) if normalize else features.detach()
        reference = reference.to(device=self.centers.device, dtype=self.centers.dtype)
        labels = labels.detach().to(device=self.centers.device).long()
        for cls in labels.unique():
            class_id = int(cls.item())
            if class_id < 1 or class_id >= self.num_classes:
                continue
            mask = labels == class_id
            if not mask.any():
                continue
            class_center = reference[mask].mean(dim=0)
            if bool(self.initialized[class_id].item()):
                self.centers[class_id].mul_(self.momentum).add_(class_center, alpha=1.0 - self.momentum)
            else:
                self.centers[class_id].copy_(class_center)
                self.initialized[class_id] = True
            self.counts[class_id] += int(mask.sum().item())


def infer_box_feature_dim(model: nn.Module, device: torch.device, config: dict) -> int:
    was_training = model.training
    model.eval()
    try:
        dummy_img = [
            torch.zeros(
                3,
                int(config["model"].get("min_size", 320)),
                int(config["model"].get("max_size", 320)),
                device=device,
            )
        ]
        with torch.no_grad():
            transformed, _ = model.transform(dummy_img, None)
            feats = model.backbone(transformed.tensors)
            if isinstance(feats, torch.Tensor):
                feats = OrderedDict([("0", feats)])
            dummy_boxes = [torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)]
            pooled = model.roi_heads.box_roi_pool(feats, dummy_boxes, transformed.image_sizes)
            pooled_features = model.roi_heads.box_head(pooled)
        return int(pooled_features.shape[-1])
    finally:
        model.train(was_training)


def prototype_projection_targets(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    sinkhorn: SinkhornAssigner,
    normalize: bool,
    orient_idx: torch.Tensor | None = None,
    scale_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Project each feature onto its class-conditioned prototype manifold."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if labels.shape != features.shape[:1]:
        raise ValueError("labels must match the feature batch dimension")
    if features.shape[1] != prototype_bank.feature_dim:
        raise ValueError(
            f"features must have dim {prototype_bank.feature_dim}, got {features.shape[1]}"
        )

    labels = labels.long()
    assignment_features = F.normalize(features, dim=-1) if normalize else features
    if features.shape[0] == 0:
        empty_distances = features.new_empty((0, prototype_bank.num_prototypes_per_class))
        return {
            "assignment_features": assignment_features,
            "distances": empty_distances,
            "assignments": empty_distances,
            "target_features": features.new_empty((0, features.shape[1])),
            "class_prototypes": features.new_empty(
                (0, prototype_bank.num_prototypes_per_class, features.shape[1])
            ),
        }

    distances = prototype_bank.compute_distances(
        assignment_features, labels, orient_idx=orient_idx, scale_idx=scale_idx
    )
    assignments = sinkhorn(distances)
    if isinstance(prototype_bank, RemoteSensingPrototypeBank):
        class_prototypes = prototype_bank.prototypes[
            labels, orient_idx, scale_idx
        ].to(device=features.device, dtype=features.dtype)
    else:
        class_prototypes = prototype_bank.prototypes[labels].to(
            device=features.device,
            dtype=features.dtype,
        )
    target_features = torch.einsum("bk,bkd->bd", assignments, class_prototypes)
    return {
        "assignment_features": assignment_features,
        "distances": distances,
        "assignments": assignments,
        "target_features": target_features,
        "class_prototypes": class_prototypes,
    }


def _assignment_weighted_same_class_loss(
    endpoints: torch.Tensor,
    assignments: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if endpoints.shape[0] <= 1:
        return endpoints.sum() * 0.0

    same_class = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    assignment_similarity = assignments @ assignments.T
    weights = assignment_similarity * same_class.to(endpoints.dtype) * off_diagonal.to(endpoints.dtype)
    denom = weights.sum()
    if float(denom.detach().item()) <= 1e-12:
        return endpoints.sum() * 0.0

    pairwise_sqdist = torch.cdist(endpoints, endpoints).square()
    return (weights * pairwise_sqdist).sum() / denom.clamp_min(1e-12)


def _same_class_prototype_diversity_loss(
    prototype_bank: PrototypeBank,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    prototypes = prototype_bank.prototypes
    if prototype_bank.num_prototypes_per_class <= 1 or prototype_bank.num_classes <= 1:
        return prototypes.sum() * 0.0

    losses = []
    temperature = max(float(temperature), 1e-12)
    present_classes = [int(cls.item()) for cls in labels.unique() if int(cls.item()) >= 1]
    for class_id in present_classes:
        class_prototypes = prototypes[class_id]
        pairwise_sqdist = torch.cdist(class_prototypes, class_prototypes).square()
        off_diagonal = ~torch.eye(
            prototype_bank.num_prototypes_per_class,
            dtype=torch.bool,
            device=prototypes.device,
        )
        losses.append(torch.exp(-pairwise_sqdist[off_diagonal] / temperature).mean())

    if not losses:
        return prototypes.sum() * 0.0
    return torch.stack(losses).mean()


def _inter_class_projection_margin_loss(
    endpoints: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    *,
    margin: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if endpoints.shape[0] == 0 or prototype_bank.num_classes <= 2:
        return endpoints.sum() * 0.0

    prototypes = prototype_bank.prototypes.to(device=endpoints.device, dtype=endpoints.dtype)
    all_distances = (endpoints[:, None, None, :] - prototypes[None, :, :, :]).square().sum(dim=-1)
    true_distances = all_distances[torch.arange(labels.numel(), device=endpoints.device), labels.long()]
    true_nearest = true_distances.min(dim=-1).values

    wrong_mask = torch.ones(
        (labels.numel(), prototype_bank.num_classes),
        dtype=torch.bool,
        device=endpoints.device,
    )
    wrong_mask[torch.arange(labels.numel(), device=endpoints.device), labels.long()] = False
    wrong_mask[:, 0] = False
    wrong_distances = all_distances.masked_fill(~wrong_mask[:, :, None], float("inf"))
    wrong_nearest = wrong_distances.flatten(start_dim=1).min(dim=-1).values

    valid = torch.isfinite(wrong_nearest)
    if not valid.any():
        return endpoints.sum() * 0.0
    margins = F.relu(float(margin) + true_nearest[valid] - wrong_nearest[valid])
    if class_weights is not None:
        w = class_weights[valid]
        return (margins * w).sum() / w.sum().clamp_min(1e-12)
    return margins.mean()


def projection_geometry_losses(
    endpoints: torch.Tensor,
    assignments: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    *,
    lambda_intra: float,
    lambda_proto_div: float,
    lambda_inter: float,
    inter_margin: float = 0.5,
    proto_div_temperature: float = 0.1,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Regularize endpoints on the class-conditioned prototype manifold."""
    labels = labels.long()
    loss_intra = _assignment_weighted_same_class_loss(endpoints, assignments, labels)
    loss_proto_div = _same_class_prototype_diversity_loss(
        prototype_bank,
        labels,
        temperature=proto_div_temperature,
    )
    loss_inter = _inter_class_projection_margin_loss(
        endpoints,
        labels,
        prototype_bank,
        margin=inter_margin,
        class_weights=class_weights,
    )

    total = (
        lambda_intra * loss_intra
        + lambda_proto_div * loss_proto_div.to(device=endpoints.device, dtype=endpoints.dtype)
        + lambda_inter * loss_inter
    )
    return {
        "loss_projection_geometry_total": total,
        "loss_projection_intra": loss_intra.detach(),
        "loss_projection_proto_div": loss_proto_div.detach(),
        "loss_projection_inter": loss_inter.detach(),
    }


def _wrong_class_prototype_margin_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    *,
    margin: float,
    normalize: bool,
) -> torch.Tensor:
    if features.shape[0] == 0 or prototype_bank.num_classes <= 2:
        return features.sum() * 0.0

    prototypes = prototype_bank.prototypes.to(device=features.device, dtype=features.dtype)
    distance_features = F.normalize(features, dim=-1) if normalize else features
    distance_prototypes = F.normalize(prototypes, dim=-1) if normalize else prototypes

    all_distances = torch.cdist(
        distance_features[:, None, :],
        distance_prototypes.flatten(start_dim=0, end_dim=1)[None, :, :],
    ).squeeze(1)
    class_ids = torch.arange(prototype_bank.num_classes, device=features.device)
    class_ids = class_ids.repeat_interleave(prototype_bank.num_prototypes_per_class)
    wrong_mask = class_ids[None, :] != labels.long()[:, None]
    wrong_mask = wrong_mask & (class_ids[None, :] != 0)
    wrong_distances = all_distances.masked_fill(~wrong_mask, float("inf"))
    wrong_nearest = wrong_distances.min(dim=-1).values

    valid = torch.isfinite(wrong_nearest)
    if not valid.any():
        return features.sum() * 0.0
    return F.relu(float(margin) - wrong_nearest[valid]).square().mean()


def _relative_class_prototype_margin_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    *,
    margin: float,
    normalize: bool,
) -> torch.Tensor:
    if features.shape[0] == 0 or prototype_bank.num_classes <= 2:
        return features.sum() * 0.0

    prototypes = prototype_bank.prototypes.to(device=features.device, dtype=features.dtype)
    distance_features = F.normalize(features, dim=-1) if normalize else features
    distance_prototypes = F.normalize(prototypes, dim=-1) if normalize else prototypes
    distances = (
        distance_features[:, None, None, :] - distance_prototypes[None, :, :, :]
    ).square().sum(dim=-1)

    rows = torch.arange(labels.numel(), device=features.device)
    true_nearest = distances[rows, labels.long()].min(dim=-1).values

    wrong_mask = torch.ones(
        (labels.numel(), prototype_bank.num_classes),
        dtype=torch.bool,
        device=features.device,
    )
    wrong_mask[rows, labels.long()] = False
    wrong_mask[:, 0] = False
    wrong_distances = distances.masked_fill(~wrong_mask[:, :, None], float("inf"))
    wrong_nearest = wrong_distances.flatten(start_dim=1).min(dim=-1).values

    valid = torch.isfinite(wrong_nearest)
    if not valid.any():
        return features.sum() * 0.0
    return F.relu(float(margin) + true_nearest[valid] - wrong_nearest[valid]).mean()


def _class_centroid_margin_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    normalize: bool,
) -> torch.Tensor:
    if features.shape[0] <= 1:
        return features.sum() * 0.0

    distance_features = F.normalize(features, dim=-1) if normalize else features
    centroids = []
    for cls in labels.long().unique():
        if int(cls.item()) < 1:
            continue
        mask = labels == cls
        if int(mask.sum().item()) == 0:
            continue
        centroids.append(distance_features[mask].mean(dim=0))

    if len(centroids) < 2:
        return features.sum() * 0.0

    centroid_tensor = torch.stack(centroids, dim=0)
    distances = torch.cdist(centroid_tensor, centroid_tensor, p=2)
    off_diagonal = ~torch.eye(
        centroid_tensor.shape[0],
        dtype=torch.bool,
        device=centroid_tensor.device,
    )
    return F.relu(float(margin) - distances[off_diagonal]).square().mean()


def _class_centroid_inter_preservation_loss(
    corrected_features: torch.Tensor,
    reference_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    if corrected_features.shape != reference_features.shape:
        raise ValueError("corrected_features and reference_features must have matching shapes")
    if corrected_features.shape[0] <= 1:
        return corrected_features.sum() * 0.0

    corrected = F.normalize(corrected_features, dim=-1) if normalize else corrected_features
    reference = F.normalize(reference_features, dim=-1) if normalize else reference_features
    corrected_centroids = []
    reference_centroids = []
    for cls in labels.long().unique():
        if int(cls.item()) < 1:
            continue
        mask = labels == cls
        if int(mask.sum().item()) == 0:
            continue
        corrected_centroids.append(corrected[mask].mean(dim=0))
        reference_centroids.append(reference[mask].mean(dim=0))

    if len(corrected_centroids) < 2:
        return corrected_features.sum() * 0.0

    corrected_centroids_t = torch.stack(corrected_centroids, dim=0)
    reference_centroids_t = torch.stack(reference_centroids, dim=0)
    corrected_distances = torch.cdist(corrected_centroids_t, corrected_centroids_t, p=2)
    reference_distances = torch.cdist(reference_centroids_t, reference_centroids_t, p=2).detach()
    off_diagonal = ~torch.eye(
        corrected_centroids_t.shape[0],
        dtype=torch.bool,
        device=corrected_centroids_t.device,
    )
    shrinkage = reference_distances[off_diagonal] - corrected_distances[off_diagonal]
    return F.relu(shrinkage).square().mean()


def _class_centroid_position_preservation_loss(
    corrected_features: torch.Tensor,
    reference_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    if corrected_features.shape != reference_features.shape:
        raise ValueError("corrected_features and reference_features must have matching shapes")
    if corrected_features.shape[0] == 0:
        return corrected_features.sum() * 0.0

    corrected = F.normalize(corrected_features, dim=-1) if normalize else corrected_features
    reference = F.normalize(reference_features, dim=-1) if normalize else reference_features
    losses = []
    for cls in labels.long().unique():
        if int(cls.item()) < 1:
            continue
        mask = labels == cls
        if int(mask.sum().item()) == 0:
            continue
        corrected_center = corrected[mask].mean(dim=0)
        reference_center = reference[mask].mean(dim=0).detach()
        losses.append((corrected_center - reference_center).square().sum())

    if len(losses) == 0:
        return corrected_features.sum() * 0.0
    return torch.stack(losses).mean()


def _class_centroid_memory_preservation_loss(
    corrected_features: torch.Tensor,
    labels: torch.Tensor,
    centroid_memory: ClassCentroidMemory,
    *,
    normalize: bool,
) -> torch.Tensor:
    if corrected_features.shape[0] == 0:
        return corrected_features.sum() * 0.0

    corrected = F.normalize(corrected_features, dim=-1) if normalize else corrected_features
    labels = labels.long()
    losses = []
    for cls in labels.unique():
        class_id = int(cls.item())
        if class_id < 1 or class_id >= centroid_memory.num_classes:
            continue
        if not bool(centroid_memory.initialized[class_id].item()):
            continue
        mask = labels == class_id
        if int(mask.sum().item()) == 0:
            continue
        corrected_center = corrected[mask].mean(dim=0)
        memory_center = centroid_memory.centers[class_id].detach().to(
            device=corrected_features.device,
            dtype=corrected_features.dtype,
        )
        losses.append((corrected_center - memory_center).square().sum())

    if len(losses) == 0:
        return corrected_features.sum() * 0.0
    return torch.stack(losses).mean()


def _class_centroid_memory_inter_preservation_loss(
    corrected_features: torch.Tensor,
    labels: torch.Tensor,
    centroid_memory: ClassCentroidMemory,
    *,
    normalize: bool,
) -> torch.Tensor:
    if corrected_features.shape[0] <= 1:
        return corrected_features.sum() * 0.0

    corrected = F.normalize(corrected_features, dim=-1) if normalize else corrected_features
    labels = labels.long()
    corrected_centroids = []
    memory_centroids = []
    for cls in labels.unique():
        class_id = int(cls.item())
        if class_id < 1 or class_id >= centroid_memory.num_classes:
            continue
        if not bool(centroid_memory.initialized[class_id].item()):
            continue
        mask = labels == class_id
        if int(mask.sum().item()) == 0:
            continue
        corrected_centroids.append(corrected[mask].mean(dim=0))
        memory_centroids.append(
            centroid_memory.centers[class_id]
            .detach()
            .to(device=corrected_features.device, dtype=corrected_features.dtype)
        )

    if len(corrected_centroids) < 2:
        return corrected_features.sum() * 0.0

    corrected_centroids_t = torch.stack(corrected_centroids, dim=0)
    memory_centroids_t = torch.stack(memory_centroids, dim=0)
    corrected_distances = torch.cdist(corrected_centroids_t, corrected_centroids_t, p=2)
    memory_distances = torch.cdist(memory_centroids_t, memory_centroids_t, p=2).detach()
    off_diagonal = ~torch.eye(
        corrected_centroids_t.shape[0],
        dtype=torch.bool,
        device=corrected_centroids_t.device,
    )
    shrinkage = memory_distances[off_diagonal] - corrected_distances[off_diagonal]
    return F.relu(shrinkage).square().mean()


def corrected_feature_geometry_losses(
    corrected_features: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    *,
    lambda_intra: float,
    lambda_inter: float,
    inter_margin: float,
    normalize: bool,
    reference_features: torch.Tensor | None = None,
    lambda_inter_preserve: float = 0.0,
    lambda_center_preserve: float = 0.0,
    centroid_memory: ClassCentroidMemory | None = None,
    lambda_memory_center_preserve: float = 0.0,
    lambda_memory_inter_preserve: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Regularize the actual corrected ROI features used by active inference."""
    labels = labels.long()
    loss_intra = supervised_compactness_loss(corrected_features, labels, normalize=normalize)
    loss_inter = _wrong_class_prototype_margin_loss(
        corrected_features,
        labels,
        prototype_bank,
        margin=inter_margin,
        normalize=normalize,
    ) + _relative_class_prototype_margin_loss(
        corrected_features,
        labels,
        prototype_bank,
        margin=inter_margin,
        normalize=normalize,
    ) + _class_centroid_margin_loss(
        corrected_features,
        labels,
        margin=inter_margin,
        normalize=normalize,
    )
    if reference_features is None or lambda_inter_preserve == 0.0:
        loss_inter_preserve = corrected_features.sum() * 0.0
    else:
        loss_inter_preserve = _class_centroid_inter_preservation_loss(
            corrected_features,
            reference_features,
            labels,
            normalize=normalize,
        )
    if reference_features is None or lambda_center_preserve == 0.0:
        loss_center_preserve = corrected_features.sum() * 0.0
    else:
        loss_center_preserve = _class_centroid_position_preservation_loss(
            corrected_features,
            reference_features,
            labels,
            normalize=normalize,
        )
    if centroid_memory is None or lambda_memory_center_preserve == 0.0:
        loss_memory_center_preserve = corrected_features.sum() * 0.0
    else:
        loss_memory_center_preserve = _class_centroid_memory_preservation_loss(
            corrected_features,
            labels,
            centroid_memory,
            normalize=normalize,
        )
    if centroid_memory is None or lambda_memory_inter_preserve == 0.0:
        loss_memory_inter_preserve = corrected_features.sum() * 0.0
    else:
        loss_memory_inter_preserve = _class_centroid_memory_inter_preservation_loss(
            corrected_features,
            labels,
            centroid_memory,
            normalize=normalize,
        )
    total = (
        lambda_intra * loss_intra
        + lambda_inter * loss_inter
        + lambda_inter_preserve * loss_inter_preserve
        + lambda_center_preserve * loss_center_preserve
        + lambda_memory_center_preserve * loss_memory_center_preserve
        + lambda_memory_inter_preserve * loss_memory_inter_preserve
    )
    return {
        "loss_corrected_geometry_total": total,
        "loss_corrected_intra": loss_intra.detach(),
        "loss_corrected_inter": loss_inter.detach(),
        "loss_corrected_inter_preserve": loss_inter_preserve.detach(),
        "loss_corrected_center_preserve": loss_center_preserve.detach(),
        "loss_corrected_memory_center_preserve": loss_memory_center_preserve.detach(),
        "loss_corrected_memory_inter_preserve": loss_memory_inter_preserve.detach(),
    }


def correction_field_preservation_loss(
    current_corrected_features: torch.Tensor,
    reference_corrected_features: torch.Tensor,
    *,
    lambda_preserve: float,
) -> torch.Tensor:
    """Keep the active correction field close to its warmup reference."""
    if current_corrected_features.shape != reference_corrected_features.shape:
        raise ValueError(
            "current_corrected_features and reference_corrected_features must have "
            f"the same shape, got {current_corrected_features.shape} and "
            f"{reference_corrected_features.shape}"
        )
    if current_corrected_features.shape[0] == 0 or lambda_preserve == 0.0:
        return current_corrected_features.sum() * 0.0
    return float(lambda_preserve) * F.smooth_l1_loss(
        current_corrected_features,
        reference_corrected_features.detach(),
        reduction="mean",
    )


def manifold_losses(
    box_features: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    sinkhorn: SinkhornAssigner,
    transport_head: TransportHead,
    lambda_tr: float,
    lambda_en: float,
    normalize: bool,
    orient_idx: torch.Tensor | None = None,
    scale_idx: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    projection = prototype_projection_targets(
        box_features,
        labels,
        prototype_bank,
        sinkhorn,
        normalize,
        orient_idx=orient_idx,
        scale_idx=scale_idx,
    )
    box_features = projection["assignment_features"]
    distances = projection["distances"]
    transport = transport_head(box_features, distances)

    transported = box_features + transport
    per_sample_loss = (transported - projection["target_features"]).square().sum(dim=-1)
    if class_weights is not None:
        loss_transport = (per_sample_loss * class_weights).sum() / class_weights.sum().clamp_min(1e-12)
    else:
        loss_transport = per_sample_loss.mean()

    energy = (transport ** 2).sum(dim=-1)
    if class_weights is not None:
        loss_energy = (energy * class_weights).sum() / class_weights.sum().clamp_min(1e-12)
    else:
        loss_energy = energy.mean()

    total = lambda_tr * loss_transport + lambda_en * loss_energy
    return {
        "loss_manifold_total": total,
        "loss_transport": loss_transport.detach(),
        "loss_energy": loss_energy.detach(),
    }


def active_manifold_losses(
    box_features: torch.Tensor,
    labels: torch.Tensor,
    prototype_bank: PrototypeBank,
    sinkhorn: SinkhornAssigner,
    correction_predictor: ManifoldCorrectionPredictor,
    lambda_tr: float,
    lambda_en: float,
    normalize: bool,
    orient_idx: torch.Tensor | None = None,
    scale_idx: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(correction_predictor, ManifoldCorrectionPredictor):
        raise TypeError("correction_predictor must be a ManifoldCorrectionPredictor")
    if box_features.shape[0] == 0:
        zero = box_features.new_tensor(0.0)
        return {
            "loss_manifold_total": zero,
            "loss_transport": zero,
            "loss_energy": zero,
        }

    labels = labels.long()
    projection = prototype_projection_targets(
        box_features,
        labels,
        prototype_bank,
        sinkhorn,
        normalize,
        orient_idx=orient_idx,
        scale_idx=scale_idx,
    )

    class_weights_onehot = F.one_hot(labels, num_classes=prototype_bank.num_classes).to(
        device=box_features.device,
        dtype=box_features.dtype,
    )
    transport = correction_predictor.correction_field_from_class_weights(
        box_features,
        class_weights_onehot,
    )

    corrected_features = box_features + correction_predictor.gamma * transport
    aligned_features = F.normalize(corrected_features, dim=-1) if normalize else corrected_features
    per_sample_transport = (aligned_features - projection["target_features"]).square().sum(dim=-1)
    if class_weights is not None:
        loss_transport = (per_sample_transport * class_weights).sum() / class_weights.sum().clamp_min(1e-12)
    else:
        loss_transport = per_sample_transport.mean()

    # Penalize the actual displacement, not the unit residual, so that larger
    # gamma automatically pays a higher energy price.
    actual_displacement = correction_predictor.gamma * transport
    per_sample_energy = (actual_displacement ** 2).sum(dim=-1)
    if class_weights is not None:
        loss_energy = (per_sample_energy * class_weights).sum() / class_weights.sum().clamp_min(1e-12)
    else:
        loss_energy = per_sample_energy.mean()
    total = lambda_tr * loss_transport + lambda_en * loss_energy
    return {
        "loss_manifold_total": total,
        "loss_transport": loss_transport.detach(),
        "loss_energy": loss_energy.detach(),
    }


def maybe_update_prototypes(
    prototype_bank: PrototypeBank,
    features: torch.Tensor,
    labels: torch.Tensor,
    sinkhorn: SinkhornAssigner,
    normalize: bool,
    freeze: bool,
    orient_idx: torch.Tensor | None = None,
    scale_idx: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
) -> bool:
    """EMA-update prototype endpoints unless the run keeps warmup endpoints fixed."""
    if freeze or features.shape[0] == 0:
        return False
    with torch.no_grad():
        feat_for_update = F.normalize(features, dim=-1) if normalize else features
        distances = prototype_bank.compute_distances(
            feat_for_update, labels, orient_idx=orient_idx, scale_idx=scale_idx
        )
        assignments = sinkhorn(distances)
        prototype_bank.update(
            feat_for_update,
            labels,
            assignments,
            orient_idx=orient_idx,
            scale_idx=scale_idx,
            class_weights=class_weights,
        )
    return True


def compute_orient_scale_indices(
    boxes: list[torch.Tensor],
    prototype_bank: PrototypeBank,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (orient_idx, scale_idx) tensors if using a remote-sensing bank.

    Args:
        boxes: list of ``(B_i, 4)`` tensors in the same order as labels.
        prototype_bank: the prototype bank (plain or remote-sensing).

    Returns:
        Tuple of ``(N,)`` long tensors or ``(None, None)`` for plain banks.
    """
    if not isinstance(prototype_bank, RemoteSensingPrototypeBank):
        return None, None
    if not boxes:
        return None, None
    cat_boxes = torch.cat(boxes, dim=0)
    orient_idx = RemoteSensingPrototypeBank.orient_idx_from_boxes(
        cat_boxes, prototype_bank.n_orient_bins
    )
    scale_idx = RemoteSensingPrototypeBank.scale_idx_from_boxes(
        cat_boxes, prototype_bank.n_scale_bins
    )
    return orient_idx, scale_idx


def per_sample_class_weights(
    labels: torch.Tensor,
    class_weight_per_class: torch.Tensor | None,
) -> torch.Tensor | None:
    """Map per-class weights to a per-sample weight vector."""
    if class_weight_per_class is None:
        return None
    return class_weight_per_class[labels.long()]


def update_best_metric(
    *,
    metric_name: str,
    candidate_metrics: dict,
    epoch: int,
    best_value: float,
    best_epoch: int,
    best_metrics: dict,
    min_delta: float = 1e-6,
) -> tuple[float, int, dict, bool]:
    """Update a best-metric record for one validation metric."""
    candidate_value = float(candidate_metrics.get(metric_name, 0.0))
    if candidate_value > best_value + min_delta:
        return candidate_value, epoch, candidate_metrics, True
    return best_value, best_epoch, best_metrics, False


def spectral_tail_loss(features: torch.Tensor, rank: int, normalize: bool) -> torch.Tensor:
    """Penalize variance outside the leading mini-batch principal directions."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if features.shape[0] <= 1 or features.shape[1] == 0:
        return features.sum() * 0.0

    if normalize:
        features = F.normalize(features, dim=-1)
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total_energy = energy.sum()
    if float(total_energy.detach().item()) <= 0.0:
        return centered.sum() * 0.0

    keep = max(0, min(int(rank), energy.numel()))
    if keep >= energy.numel():
        return centered.sum() * 0.0
    return energy[keep:].sum() / total_energy.clamp_min(1e-12)


def supervised_compactness_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Mean squared distance to each present foreground class centroid."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if features.shape[0] <= 1:
        return features.sum() * 0.0

    if normalize:
        features = F.normalize(features, dim=-1)
    labels = labels.long()
    class_losses = []
    for cls in labels.unique():
        if int(cls.item()) < 1:
            continue
        mask = labels == cls
        if int(mask.sum().item()) < 2:
            continue
        cls_features = features[mask]
        centroid = cls_features.mean(dim=0, keepdim=True)
        class_losses.append((cls_features - centroid).square().sum(dim=-1).mean())

    if not class_losses:
        return features.sum() * 0.0
    return torch.stack(class_losses).mean()


def fc1_geometry_losses(
    fc1_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    rank: int,
    lambda_rank: float,
    lambda_compact: float,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    """Layer-wise geometry regularizer for the box_head fc1 representation."""
    rank_loss = spectral_tail_loss(fc1_features, rank=rank, normalize=normalize)
    compact_loss = supervised_compactness_loss(fc1_features, labels, normalize=normalize)
    total = lambda_rank * rank_loss + lambda_compact * compact_loss
    return {
        "loss_fc1_geometry_total": total,
        "loss_fc1_rank": rank_loss.detach(),
        "loss_fc1_compact": compact_loss.detach(),
    }


def _select_labelled_box_deltas(box_regression: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Select the class-specific 4D bbox delta for each foreground proposal."""
    if box_regression.ndim != 2:
        raise ValueError(f"box_regression must be 2D, got shape {box_regression.shape}")
    if box_regression.shape[-1] == 4:
        return box_regression
    if box_regression.shape[-1] % 4 != 0:
        raise ValueError("box_regression width must be 4 or a multiple of 4")

    num_box_classes = box_regression.shape[-1] // 4
    if labels.numel() == 0:
        return box_regression.new_empty((0, 4))
    if int(labels.max().item()) >= num_box_classes:
        raise ValueError(
            f"labels contain class {int(labels.max().item())}, "
            f"but box_regression only has {num_box_classes} class slots"
        )

    reshaped = box_regression.view(box_regression.shape[0], num_box_classes, 4)
    rows = torch.arange(labels.numel(), device=box_regression.device)
    return reshaped[rows, labels.long()]


def prediction_preservation_losses(
    student_logits: torch.Tensor,
    student_bbox: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_bbox: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_logits: float,
    lambda_bbox: float,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Keep ROI classifier/regressor outputs close to the baseline teacher."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have matching shapes")
    if student_bbox.shape != teacher_bbox.shape:
        raise ValueError("student_bbox and teacher_bbox must have matching shapes")
    if student_logits.shape[0] != labels.numel() or student_bbox.shape[0] != labels.numel():
        raise ValueError("labels must match the number of predictions")

    if labels.numel() == 0:
        zero = student_logits.sum() * 0.0 + student_bbox.sum() * 0.0
        return {
            "loss_preserve_total": zero,
            "loss_preserve_logits": zero.detach(),
            "loss_preserve_bbox": zero.detach(),
        }

    teacher_logits = teacher_logits.detach()
    teacher_bbox = teacher_bbox.detach()
    temperature = max(float(temperature), 1e-6)

    log_student = F.log_softmax(student_logits / temperature, dim=-1)
    prob_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    loss_logits = F.kl_div(log_student, prob_teacher, reduction="batchmean") * (temperature ** 2)

    fg_mask = labels >= 1
    if fg_mask.any():
        fg_labels = labels[fg_mask].to(device=student_bbox.device, dtype=torch.long)
        student_selected = _select_labelled_box_deltas(student_bbox[fg_mask], fg_labels)
        teacher_selected = _select_labelled_box_deltas(teacher_bbox[fg_mask], fg_labels)
        loss_bbox = F.smooth_l1_loss(student_selected, teacher_selected, reduction="mean")
    else:
        loss_bbox = student_bbox.sum() * 0.0

    total = lambda_logits * loss_logits + lambda_bbox * loss_bbox
    return {
        "loss_preserve_total": total,
        "loss_preserve_logits": loss_logits.detach(),
        "loss_preserve_bbox": loss_bbox.detach(),
    }


def install_active_manifold_correction(
    model: nn.Module,
    *,
    prototype_bank: PrototypeBank,
    gamma: float,
    tau: float,
    normalize_features: bool,
    correction_mode: str = "residual",
    endpoint_gate_init: float = 0.25,
) -> TransportHead:
    """Install active manifold correction before the detector box predictor."""
    correction_mode = correction_mode.replace("-", "_")
    active_transport_head = TransportHead(
        feature_dim=prototype_bank.feature_dim,
        num_prototypes=prototype_bank.num_classes * prototype_bank.num_prototypes_per_class,
        tau=tau,
        residual_scale=0.0,
    ).to(prototype_bank.prototypes.device)
    model.roi_heads.box_predictor = ManifoldCorrectionPredictor(
        model.roi_heads.box_predictor,
        prototype_bank=prototype_bank,
        transport_head=active_transport_head,
        gamma=gamma,
        tau=tau,
        normalize_features=normalize_features,
        correction_mode=correction_mode,
        endpoint_gate_init=endpoint_gate_init,
    ).to(prototype_bank.prototypes.device)
    return active_transport_head


def compute_corrected_features_for_active_head(
    box_predictor: ManifoldCorrectionPredictor,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the corrected features that the active head would see at inference.

    Uses ground-truth class weights so the geometry metric reflects the
    class-conditional correction field for each sample.
    """
    num_classes = box_predictor.prototype_bank.num_classes
    class_weights = F.one_hot(labels, num_classes=num_classes).to(
        device=features.device, dtype=features.dtype
    )
    transport = box_predictor.correction_field_from_class_weights(features, class_weights)
    return features + box_predictor.gamma * transport


def active_correction_gamma_for_epoch(
    *,
    initial_gamma: float,
    final_gamma: float | None,
    epoch: int,
    total_epochs: int,
    schedule: str,
) -> float:
    if initial_gamma < 0.0:
        raise ValueError("initial_gamma must be non-negative")
    if final_gamma is not None and final_gamma < 0.0:
        raise ValueError("final_gamma must be non-negative")

    schedule = schedule.replace("-", "_")
    if schedule == "constant":
        return float(initial_gamma)
    if schedule != "linear_decay":
        raise ValueError(f"unsupported active correction gamma schedule: {schedule}")

    target_gamma = float(initial_gamma if final_gamma is None else final_gamma)
    if total_epochs <= 1:
        return float(initial_gamma)
    progress = (max(1, int(epoch)) - 1) / max(1, int(total_epochs) - 1)
    progress = min(max(progress, 0.0), 1.0)
    return float(initial_gamma + (target_gamma - initial_gamma) * progress)


def set_active_correction_gamma(model: nn.Module, gamma: float) -> bool:
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    predictor = getattr(getattr(model, "roi_heads", None), "box_predictor", None)
    if not isinstance(predictor, ManifoldCorrectionPredictor):
        return False
    predictor.gamma = float(gamma)
    return True


def _box_predictor_base_parameters(model: nn.Module) -> list[nn.Parameter]:
    predictor = model.roi_heads.box_predictor
    if isinstance(predictor, ManifoldCorrectionPredictor):
        return list(predictor.base_predictor.parameters())
    return list(predictor.parameters())


@torch.no_grad()
def eval_metrics(model: nn.Module, val_loader: DataLoader, device: torch.device, config: dict, num_classes: int) -> dict:
    model.eval()
    predictions = []
    targets = []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in tgt.items()}
                        for tgt in batch_targets])
    metrics = evaluate_detection_predictions(
        predictions, targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
        per_class=True,
        num_classes=num_classes,
    )
    return metrics


@torch.no_grad()
def eval_ap50(model: nn.Module, val_loader: DataLoader, device: torch.device, config: dict) -> float:
    return float(eval_metrics(model, val_loader, device, config)["ap50"])


def initial_sanity_payload(
    raw_baseline_metrics: dict,
    active_initial_metrics: dict,
) -> dict:
    raw_ap50 = float(raw_baseline_metrics["ap50"])
    raw_ap75 = float(raw_baseline_metrics.get("ap75", 0.0))
    active_ap50 = float(active_initial_metrics["ap50"])
    active_ap75 = float(active_initial_metrics.get("ap75", 0.0))
    return {
        "raw_baseline_val_ap50": raw_ap50,
        "raw_baseline_val_ap75": raw_ap75,
        "raw_baseline_metrics": raw_baseline_metrics,
        "active_initial_val_ap50": active_ap50,
        "active_initial_val_ap75": active_ap75,
        "active_initial_metrics": active_initial_metrics,
        "initial_val_ap50": active_ap50,
        "initial_val_ap75": active_ap75,
        "initial_metrics": active_initial_metrics,
    }


def initial_checkpoint_extra_metadata(
    raw_baseline_metrics: dict,
    active_initial_metrics: dict,
) -> dict:
    active_ap50 = float(active_initial_metrics["ap50"])
    active_ap75 = float(active_initial_metrics.get("ap75", 0.0))
    return {
        "epoch": 0,
        "val_ap50": active_ap50,
        "val_ap75": active_ap75,
        **initial_sanity_payload(raw_baseline_metrics, active_initial_metrics),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    context = prepare_experiment_from_config(
        config, args.config, args.run_name, phase="manifold_posttrain", checkpoint_path=args.baseline
    )
    config = context.config
    run_dir = context.run_dir

    train_loader, val_loader = build_detection_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=int(config["posttrain"].get("batch_size", 1)),
    )
    device = resolve_device(config)

    model = build_experiment_model(context, checkpoint_path=args.baseline, device=device, pretrained=False)
    freeze_backbone(model)
    freeze_rpn(model)

    # By default we keep the box head / predictor trainable so the manifold
    # regularizer can actually reshape the feature space.  Freezing them is the
    # most conservative fallback if the user observes any instability.
    if args.freeze_box_head:
        freeze_box_head(model)
    if args.freeze_box_predictor:
        freeze_box_predictor(model)

    if args.lambda_correction_field_preserve > 0.0 and not args.active_manifold_correction:
        raise ValueError("--lambda-correction-field-preserve requires --active-manifold-correction")

    use_preservation = args.lambda_logit_preserve > 0.0 or args.lambda_bbox_preserve > 0.0
    teacher_model = None
    if use_preservation:
        # Capture the original-head teacher *before* swapping the box head so the
        # preservation loss anchors the new head to the pretrained behavior.
        teacher_model = copy.deepcopy(model)
        teacher_model.to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

    model.eval()
    raw_baseline_metrics = eval_metrics(model, val_loader, device, config, num_classes=int(config["model"]["num_classes"]))
    raw_baseline_ap50 = float(raw_baseline_metrics["ap50"])
    raw_baseline_ap75 = float(raw_baseline_metrics.get("ap75", 0.0))
    print(f"Raw baseline AP50/AP75: {raw_baseline_ap50:.4f}/{raw_baseline_ap75:.4f}")

    # Optionally replace the ROI box head with a low-dim-preserving variant.
    # This happens after the raw baseline eval but before feature-dim inference,
    # so all downstream modules see the new representation size.
    if args.box_head_type not in ("", "original"):
        replace_box_head(
            model,
            args.box_head_type,
            rank=args.box_head_rank,
            conv_channels=args.box_head_conv_channels,
            bottleneck_dim=args.box_head_bottleneck_dim,
            attention_channels=args.box_head_attention_channels,
            copy_compatible_weights=True,
        )
        print(f"Replaced box head with {get_box_head_type(model)}")
        if args.freeze_box_head:
            freeze_box_head(model)

    # Determine feature dimension from the actual model without touching
    # BatchNorm running stats in the frozen detector.
    feature_dim = infer_box_feature_dim(model, device, config)

    num_classes = int(config["model"]["num_classes"])

    # Optionally replace the learnable cls_score with a fixed ETF classifier.
    if args.use_etf_classifier:
        replace_cls_score_with_etf(model.roi_heads.box_predictor, num_classes=num_classes)
        print("Replaced box_predictor.cls_score with ETF classifier")
    use_rs_prototypes = args.rs_orient_bins > 1 or args.rs_scale_bins > 1
    if use_rs_prototypes:
        prototype_bank = RemoteSensingPrototypeBank(
            num_classes=num_classes,
            num_prototypes_per_class=args.num_prototypes,
            feature_dim=feature_dim,
            n_orient_bins=args.rs_orient_bins,
            n_scale_bins=args.rs_scale_bins,
            ema_decay=args.ema_decay,
        ).to(device)
        print(
            f"Using RemoteSensingPrototypeBank: orient_bins={args.rs_orient_bins}, "
            f"scale_bins={args.rs_scale_bins}"
        )
    else:
        prototype_bank = PrototypeBank(
            num_classes=num_classes,
            num_prototypes_per_class=args.num_prototypes,
            feature_dim=feature_dim,
            ema_decay=args.ema_decay,
        ).to(device)
    sinkhorn = SinkhornAssigner(eps=args.sinkhorn_eps, max_iter=args.sinkhorn_iter).to(device)
    transport_head = None
    if not args.active_manifold_correction:
        transport_head = TransportHead(
            feature_dim=feature_dim, num_prototypes=args.num_prototypes, tau=args.tau
        ).to(device)
    id_estimator = IntrinsicDimEstimator(method=args.id_method).to(device)

    normalize_features = args.normalize_features and not args.no_normalize_features
    active_transport_head = None
    active_correction_mode = args.active_correction_mode.replace("-", "_")
    if args.active_manifold_correction:
        active_transport_head = install_active_manifold_correction(
            model,
            prototype_bank=prototype_bank,
            gamma=args.active_correction_gamma,
            tau=args.tau,
            normalize_features=args.active_correction_normalize,
            correction_mode=active_correction_mode,
            endpoint_gate_init=args.active_endpoint_gate_init,
        )

    extractor = extract_gt_box_features if args.use_gt_boxes else extract_proposal_box_features

    # Warm-start prototypes from class centers to avoid random-init instability.
    class_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    if args.warmup_batches > 0:
        print(f"Warming up prototypes from {args.warmup_batches} batches...")
        centers, class_counts = warmup_class_centers(
            model, train_loader, device, num_classes, args.warmup_batches, normalize_features, args.use_gt_boxes
        )
        initialize_prototypes_from_centers_with_seed(
            prototype_bank,
            centers,
            noise_scale=0.05,
            seed=int(config.get("seed", 42)),
        )

    class_weight_per_class = None
    if args.class_reweight != "none":
        class_weight_per_class = compute_class_frequency_weights(
            class_counts, mode=args.class_reweight, beta=args.class_reweight_beta
        )
        print(f"Class reweighting: {args.class_reweight}; weights={class_weight_per_class.cpu().tolist()}")

    # Build parameter groups: detector parts get the low LR, manifold modules
    # can afford a higher LR because they are small and randomly initialized.
    param_groups = []
    detector_params = []
    if not args.freeze_box_head:
        detector_params.extend(model.roi_heads.box_head.parameters())
    if not args.freeze_box_predictor:
        detector_params.extend(_box_predictor_base_parameters(model))
    if detector_params:
        param_groups.append({"params": detector_params, "lr": args.lr})
    manifold_params = []
    if transport_head is not None:
        manifold_params.extend(transport_head.parameters())
    if active_transport_head is not None:
        manifold_params.extend(active_transport_head.parameters())
    if isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor):
        manifold_params.extend(model.roi_heads.box_predictor.endpoint_gate_parameters())
    if manifold_params:
        param_groups.append({"params": manifold_params, "lr": args.lr_manifold})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    epochs = args.epochs if args.epochs is not None else int(config["posttrain"].get("epochs", 5))
    geometry_every = args.geometry_every
    eval_every = args.eval_every
    global_step = 0
    use_fc1_geometry = args.lambda_fc1_rank > 0.0 or args.lambda_fc1_compact > 0.0
    use_projection_geometry = (
        args.lambda_proj_intra > 0.0
        or args.lambda_proto_div > 0.0
        or args.lambda_proj_inter > 0.0
    )
    use_corrected_geometry = (
        args.lambda_corrected_intra > 0.0
        or args.lambda_corrected_inter > 0.0
        or args.lambda_corrected_inter_preserve > 0.0
        or args.lambda_corrected_center_preserve > 0.0
        or args.lambda_corrected_memory_center_preserve > 0.0
        or args.lambda_corrected_memory_inter_preserve > 0.0
    )
    use_correction_field_preserve = args.lambda_correction_field_preserve > 0.0
    corrected_centroid_memory = (
        ClassCentroidMemory(
            num_classes=num_classes,
            feature_dim=feature_dim,
            momentum=args.corrected_memory_momentum,
            device=device,
        )
        if (
            args.lambda_corrected_memory_center_preserve > 0.0
            or args.lambda_corrected_memory_inter_preserve > 0.0
        )
        else None
    )
    need_layer_features = use_fc1_geometry or use_preservation

    # Initial validation sanity check before any updates.
    model.eval()
    initial_metrics = eval_metrics(model, val_loader, device, config, num_classes)
    initial_ap50 = float(initial_metrics["ap50"])
    initial_ap75 = float(initial_metrics.get("ap75", 0.0))
    print(f"Active initial AP50/AP75: {initial_ap50:.4f}/{initial_ap75:.4f}")
    save_json(
        initial_sanity_payload(raw_baseline_metrics, initial_metrics),
        run_dir / "initial_sanity.json",
    )
    initial_checkpoint_meta = checkpoint_metadata(
        context,
        initial_checkpoint_extra_metadata(raw_baseline_metrics, initial_metrics),
    )
    save_checkpoint(model, run_dir / "checkpoint_initial.pth", initial_checkpoint_meta)
    save_checkpoint(model, run_dir / "checkpoint_best.pth", initial_checkpoint_meta)
    save_checkpoint(model, run_dir / "checkpoint_best_ap75.pth", initial_checkpoint_meta)

    best_ap50 = initial_ap50
    best_metrics = initial_metrics
    best_epoch = 0
    best_ap75 = initial_ap75
    best_ap75_metrics = initial_metrics
    best_ap75_epoch = 0
    stale_epochs = 0
    completed_epochs = 0
    correction_field_reference = None
    if use_correction_field_preserve:
        if not isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor):
            raise RuntimeError("active correction predictor is required for correction field preservation")
        correction_field_reference = copy.deepcopy(model.roi_heads.box_predictor).to(device)
        correction_field_reference.eval()
        for parameter in correction_field_reference.parameters():
            parameter.requires_grad_(False)

    for epoch in range(1, epochs + 1):
        completed_epochs = epoch
        active_correction_gamma_epoch = active_correction_gamma_for_epoch(
            initial_gamma=args.active_correction_gamma,
            final_gamma=args.active_correction_gamma_final,
            epoch=epoch,
            total_epochs=epochs,
            schedule=args.active_correction_gamma_schedule,
        )
        if args.active_manifold_correction:
            set_active_correction_gamma(model, active_correction_gamma_epoch)

        _set_model_train_for_detection_loss(model)
        prototype_bank.train()
        if transport_head is not None:
            transport_head.train()
        if active_transport_head is not None:
            active_transport_head.train()

        total_loss = 0.0
        total_loss_det = 0.0
        total_loss_tr = 0.0
        total_loss_en = 0.0
        total_loss_fc1_geometry = 0.0
        total_loss_fc1_rank = 0.0
        total_loss_fc1_compact = 0.0
        total_loss_preserve = 0.0
        total_loss_preserve_logits = 0.0
        total_loss_preserve_bbox = 0.0
        total_loss_projection_geometry = 0.0
        total_loss_projection_intra = 0.0
        total_loss_projection_proto_div = 0.0
        total_loss_projection_inter = 0.0
        total_loss_corrected_geometry = 0.0
        total_loss_corrected_intra = 0.0
        total_loss_corrected_inter = 0.0
        total_loss_corrected_inter_preserve = 0.0
        total_loss_corrected_center_preserve = 0.0
        total_loss_corrected_memory_center_preserve = 0.0
        total_loss_corrected_memory_inter_preserve = 0.0
        total_loss_correction_field_preserve = 0.0
        total_seen = 0
        total_boxes = 0
        total_fg_boxes = 0

        manifold_active = epoch > args.manifold_warmup_epochs

        # Feature geometry buffer: accumulate a fixed-size sample of foreground
        # box features across the epoch to measure ID / compactness / separation.
        geometry_feat_buffer: list[torch.Tensor] = []
        geometry_label_buffer: list[torch.Tensor] = []
        geometry_corr_buffer: list[torch.Tensor] = []
        geometry_buffer_size = max(1, args.geometry_buffer_size)

        progress = tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{epochs}")
        for images, targets in progress:
            images = [img.to(device) for img in images]
            targets = _to_device_targets(targets, device)

            # Standard detection loss on the whole image (the behavior anchor).
            loss_dict = model(images, targets)
            loss_det = sum(loss_dict.values())

            # Manifold losses on proposals (or GT boxes if legacy mode requested).
            layer_features: dict[str, torch.Tensor] = {}
            scaled_boxes: list[torch.Tensor] = []
            if need_layer_features:
                box_features, labels, scaled_boxes, layer_features = extractor(
                    model, images, targets, return_layers=True
                )
            else:
                box_features, labels, scaled_boxes = extractor(model, images, targets)
            fc1_loss_dict = {
                "loss_fc1_geometry_total": torch.tensor(0.0, device=device),
                "loss_fc1_rank": torch.tensor(0.0, device=device),
                "loss_fc1_compact": torch.tensor(0.0, device=device),
            }
            preserve_loss_dict = {
                "loss_preserve_total": torch.tensor(0.0, device=device),
                "loss_preserve_logits": torch.tensor(0.0, device=device),
                "loss_preserve_bbox": torch.tensor(0.0, device=device),
            }
            projection_loss_dict = {
                "loss_projection_geometry_total": torch.tensor(0.0, device=device),
                "loss_projection_intra": torch.tensor(0.0, device=device),
                "loss_projection_proto_div": torch.tensor(0.0, device=device),
                "loss_projection_inter": torch.tensor(0.0, device=device),
            }
            corrected_loss_dict = {
                "loss_corrected_geometry_total": torch.tensor(0.0, device=device),
                "loss_corrected_intra": torch.tensor(0.0, device=device),
                "loss_corrected_inter": torch.tensor(0.0, device=device),
                "loss_corrected_inter_preserve": torch.tensor(0.0, device=device),
                "loss_corrected_center_preserve": torch.tensor(0.0, device=device),
                "loss_corrected_memory_center_preserve": torch.tensor(0.0, device=device),
                "loss_corrected_memory_inter_preserve": torch.tensor(0.0, device=device),
            }
            correction_field_preserve_loss = torch.tensor(0.0, device=device)
            if box_features.shape[0] == 0:
                manifold_loss_dict = {
                    "loss_manifold_total": torch.tensor(0.0, device=device),
                    "loss_transport": torch.tensor(0.0, device=device),
                    "loss_energy": torch.tensor(0.0, device=device),
                }
                fg_mask = torch.zeros(0, dtype=torch.bool, device=device)
            else:
                if use_preservation:
                    if teacher_model is None:
                        raise RuntimeError("teacher_model is required when preservation is enabled")
                    if "roi_pooled" not in layer_features:
                        raise RuntimeError("roi_pooled layer features are required for preservation")
                    student_logits, student_bbox = model.roi_heads.box_predictor(box_features)
                    with torch.no_grad():
                        teacher_box_features = teacher_model.roi_heads.box_head(
                            layer_features["roi_pooled"].detach()
                        )
                        teacher_logits, teacher_bbox = teacher_model.roi_heads.box_predictor(teacher_box_features)
                    preserve_loss_dict = prediction_preservation_losses(
                        student_logits,
                        student_bbox,
                        teacher_logits,
                        teacher_bbox,
                        labels,
                        lambda_logits=args.lambda_logit_preserve if manifold_active else 0.0,
                        lambda_bbox=args.lambda_bbox_preserve if manifold_active else 0.0,
                        temperature=args.preserve_temperature,
                    )

                # Foreground proposals carry class labels 1..C; background is 0.
                fg_mask = labels >= 1
                orient_idx_all, scale_idx_all = compute_orient_scale_indices(
                    scaled_boxes, prototype_bank
                )
                orient_idx_fg = orient_idx_all[fg_mask] if orient_idx_all is not None else None
                scale_idx_fg = scale_idx_all[fg_mask] if scale_idx_all is not None else None
                if fg_mask.any():
                    feat_fg = box_features[fg_mask]
                    labels_fg = labels[fg_mask]
                    class_weights_fg = per_sample_class_weights(
                        labels_fg, class_weight_per_class
                    )
                    corrected_features_for_losses = None
                    if args.active_manifold_correction:
                        manifold_loss_dict = active_manifold_losses(
                            feat_fg,
                            labels_fg,
                            prototype_bank,
                            sinkhorn,
                            model.roi_heads.box_predictor,
                            args.lambda_tr if manifold_active else 0.0,
                            args.lambda_en if manifold_active else 0.0,
                            normalize_features,
                            orient_idx=orient_idx_fg,
                            scale_idx=scale_idx_fg,
                            class_weights=class_weights_fg,
                        )
                    else:
                        if transport_head is None:
                            raise RuntimeError("transport_head is required when active correction is disabled")
                        manifold_loss_dict = manifold_losses(
                            feat_fg,
                            labels_fg,
                            prototype_bank,
                            sinkhorn,
                            transport_head,
                            args.lambda_tr if manifold_active else 0.0,
                            args.lambda_en if manifold_active else 0.0,
                            normalize_features,
                            orient_idx=orient_idx_fg,
                            scale_idx=scale_idx_fg,
                            class_weights=class_weights_fg,
                        )
                    if use_fc1_geometry and "fc1" in layer_features:
                        fc1_loss_dict = fc1_geometry_losses(
                            layer_features["fc1"][fg_mask],
                            labels_fg,
                            rank=args.fc1_rank_target,
                            lambda_rank=args.lambda_fc1_rank if manifold_active else 0.0,
                            lambda_compact=args.lambda_fc1_compact if manifold_active else 0.0,
                            normalize=normalize_features,
                        )
                    if use_projection_geometry:
                        projection = prototype_projection_targets(
                            feat_fg,
                            labels_fg,
                            prototype_bank,
                            sinkhorn,
                            normalize_features,
                            orient_idx=orient_idx_fg,
                            scale_idx=scale_idx_fg,
                        )
                        projection_loss_dict = projection_geometry_losses(
                            projection["target_features"],
                            projection["assignments"],
                            labels_fg,
                            prototype_bank,
                            lambda_intra=args.lambda_proj_intra if manifold_active else 0.0,
                            lambda_proto_div=args.lambda_proto_div if manifold_active else 0.0,
                            lambda_inter=args.lambda_proj_inter if manifold_active else 0.0,
                            inter_margin=args.projection_inter_margin,
                            proto_div_temperature=args.proto_div_temperature,
                            class_weights=class_weights_fg,
                        )
                    if (
                        use_corrected_geometry
                        and args.active_manifold_correction
                        and isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)
                    ):
                        if corrected_centroid_memory is not None:
                            corrected_centroid_memory.update(
                                feat_fg.detach(),
                                labels_fg.detach(),
                                normalize=normalize_features,
                            )
                        corrected_features_for_losses = compute_corrected_features_for_active_head(
                            model.roi_heads.box_predictor,
                            feat_fg,
                            labels_fg,
                        )
                        corrected_loss_dict = corrected_feature_geometry_losses(
                            corrected_features_for_losses,
                            labels_fg,
                            prototype_bank,
                            lambda_intra=args.lambda_corrected_intra if manifold_active else 0.0,
                            lambda_inter=args.lambda_corrected_inter if manifold_active else 0.0,
                            inter_margin=args.corrected_inter_margin,
                            normalize=normalize_features,
                            reference_features=feat_fg,
                            lambda_inter_preserve=(
                                args.lambda_corrected_inter_preserve if manifold_active else 0.0
                            ),
                            lambda_center_preserve=(
                                args.lambda_corrected_center_preserve if manifold_active else 0.0
                            ),
                            centroid_memory=corrected_centroid_memory,
                            lambda_memory_center_preserve=(
                                args.lambda_corrected_memory_center_preserve if manifold_active else 0.0
                            ),
                            lambda_memory_inter_preserve=(
                                args.lambda_corrected_memory_inter_preserve if manifold_active else 0.0
                            ),
                        )
                    if (
                        use_correction_field_preserve
                        and args.active_manifold_correction
                        and isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)
                    ):
                        if correction_field_reference is None:
                            raise RuntimeError("correction field reference is required")
                        if corrected_features_for_losses is None:
                            corrected_features_for_losses = compute_corrected_features_for_active_head(
                                model.roi_heads.box_predictor,
                                feat_fg,
                                labels_fg,
                            )
                        with torch.no_grad():
                            reference_corrected_features = compute_corrected_features_for_active_head(
                                correction_field_reference,
                                feat_fg.detach(),
                                labels_fg.detach(),
                            )
                        correction_field_preserve_loss = correction_field_preservation_loss(
                            corrected_features_for_losses,
                            reference_corrected_features,
                            lambda_preserve=(
                                args.lambda_correction_field_preserve if manifold_active else 0.0
                            ),
                        )
                    # Update prototypes after computing losses unless the run
                    # treats warmup prototypes as fixed correction endpoints.
                    maybe_update_prototypes(
                        prototype_bank,
                        feat_fg,
                        labels_fg,
                        sinkhorn,
                        normalize_features,
                        args.freeze_prototypes_after_warmup,
                        orient_idx=orient_idx_fg,
                        scale_idx=scale_idx_fg,
                        class_weights=class_weights_fg,
                    )

                    # Accumulate features for epoch-end geometry diagnostics.
                    if len(geometry_feat_buffer) == 0 or geometry_feat_buffer[0].shape[0] < geometry_buffer_size:
                        with torch.no_grad():
                            feats_to_store = feat_fg.detach()
                            labels_to_store = labels_fg.detach()
                            if args.active_manifold_correction and isinstance(
                                model.roi_heads.box_predictor, ManifoldCorrectionPredictor
                            ):
                                corr_feats = compute_corrected_features_for_active_head(
                                    model.roi_heads.box_predictor, feats_to_store, labels_to_store
                                )
                            else:
                                corr_feats = feats_to_store.clone()
                            geometry_feat_buffer.append(feats_to_store)
                            geometry_label_buffer.append(labels_to_store)
                            geometry_corr_buffer.append(corr_feats)
                else:
                    manifold_loss_dict = {
                        "loss_manifold_total": torch.tensor(0.0, device=device),
                        "loss_transport": torch.tensor(0.0, device=device),
                        "loss_energy": torch.tensor(0.0, device=device),
                    }

            total_loss_batch = (
                loss_det
                + manifold_loss_dict["loss_manifold_total"]
                + fc1_loss_dict["loss_fc1_geometry_total"]
                + preserve_loss_dict["loss_preserve_total"]
                + projection_loss_dict["loss_projection_geometry_total"]
                + corrected_loss_dict["loss_corrected_geometry_total"]
                + correction_field_preserve_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss_batch.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], args.max_grad_norm
                )
            optimizer.step()

            batch_size = len(images)
            total_seen += batch_size
            total_boxes += box_features.shape[0]
            total_fg_boxes += int(fg_mask.sum().item())
            total_loss += float(total_loss_batch.item()) * batch_size
            total_loss_det += float(loss_det.item()) * batch_size
            total_loss_tr += float(manifold_loss_dict["loss_transport"].item()) * batch_size
            total_loss_en += float(manifold_loss_dict["loss_energy"].item()) * batch_size
            total_loss_fc1_geometry += float(fc1_loss_dict["loss_fc1_geometry_total"].item()) * batch_size
            total_loss_fc1_rank += float(fc1_loss_dict["loss_fc1_rank"].item()) * batch_size
            total_loss_fc1_compact += float(fc1_loss_dict["loss_fc1_compact"].item()) * batch_size
            total_loss_preserve += float(preserve_loss_dict["loss_preserve_total"].item()) * batch_size
            total_loss_preserve_logits += float(preserve_loss_dict["loss_preserve_logits"].item()) * batch_size
            total_loss_preserve_bbox += float(preserve_loss_dict["loss_preserve_bbox"].item()) * batch_size
            total_loss_projection_geometry += (
                float(projection_loss_dict["loss_projection_geometry_total"].item()) * batch_size
            )
            total_loss_projection_intra += float(projection_loss_dict["loss_projection_intra"].item()) * batch_size
            total_loss_projection_proto_div += (
                float(projection_loss_dict["loss_projection_proto_div"].item()) * batch_size
            )
            total_loss_projection_inter += float(projection_loss_dict["loss_projection_inter"].item()) * batch_size
            total_loss_corrected_geometry += (
                float(corrected_loss_dict["loss_corrected_geometry_total"].item()) * batch_size
            )
            total_loss_corrected_intra += float(corrected_loss_dict["loss_corrected_intra"].item()) * batch_size
            total_loss_corrected_inter += float(corrected_loss_dict["loss_corrected_inter"].item()) * batch_size
            total_loss_corrected_inter_preserve += (
                float(corrected_loss_dict["loss_corrected_inter_preserve"].item()) * batch_size
            )
            total_loss_corrected_center_preserve += (
                float(corrected_loss_dict["loss_corrected_center_preserve"].item()) * batch_size
            )
            total_loss_corrected_memory_center_preserve += (
                float(corrected_loss_dict["loss_corrected_memory_center_preserve"].item()) * batch_size
            )
            total_loss_corrected_memory_inter_preserve += (
                float(corrected_loss_dict["loss_corrected_memory_inter_preserve"].item()) * batch_size
            )
            total_loss_correction_field_preserve += float(correction_field_preserve_loss.item()) * batch_size
            global_step += 1

            progress.set_postfix(
                loss=total_loss / max(1, total_seen),
                det=total_loss_det / max(1, total_seen),
                tr=total_loss_tr / max(1, total_seen),
                en=total_loss_en / max(1, total_seen),
                fc1=total_loss_fc1_geometry / max(1, total_seen),
                keep=total_loss_preserve / max(1, total_seen),
                field=total_loss_correction_field_preserve / max(1, total_seen),
                proj=total_loss_projection_geometry / max(1, total_seen),
                corr=total_loss_corrected_geometry / max(1, total_seen),
            )

        row = {
            "epoch": epoch,
            "active_correction_gamma_epoch": active_correction_gamma_epoch,
            "loss": total_loss / max(1, total_seen),
            "loss_det": total_loss_det / max(1, total_seen),
            "loss_transport": total_loss_tr / max(1, total_seen),
            "loss_energy": total_loss_en / max(1, total_seen),
            "loss_fc1_geometry": total_loss_fc1_geometry / max(1, total_seen),
            "loss_fc1_rank": total_loss_fc1_rank / max(1, total_seen),
            "loss_fc1_compact": total_loss_fc1_compact / max(1, total_seen),
            "loss_preserve": total_loss_preserve / max(1, total_seen),
            "loss_preserve_logits": total_loss_preserve_logits / max(1, total_seen),
            "loss_preserve_bbox": total_loss_preserve_bbox / max(1, total_seen),
            "loss_projection_geometry": total_loss_projection_geometry / max(1, total_seen),
            "loss_projection_intra": total_loss_projection_intra / max(1, total_seen),
            "loss_projection_proto_div": total_loss_projection_proto_div / max(1, total_seen),
            "loss_projection_inter": total_loss_projection_inter / max(1, total_seen),
            "loss_corrected_geometry": total_loss_corrected_geometry / max(1, total_seen),
            "loss_corrected_intra": total_loss_corrected_intra / max(1, total_seen),
            "loss_corrected_inter": total_loss_corrected_inter / max(1, total_seen),
            "loss_corrected_inter_preserve": total_loss_corrected_inter_preserve / max(1, total_seen),
            "loss_corrected_center_preserve": total_loss_corrected_center_preserve / max(1, total_seen),
            "loss_corrected_memory_center_preserve": (
                total_loss_corrected_memory_center_preserve / max(1, total_seen)
            ),
            "loss_corrected_memory_inter_preserve": (
                total_loss_corrected_memory_inter_preserve / max(1, total_seen)
            ),
            "loss_correction_field_preserve": total_loss_correction_field_preserve / max(1, total_seen),
            "avg_boxes_per_batch": total_boxes / max(1, len(train_loader)),
            "avg_fg_boxes_per_batch": total_fg_boxes / max(1, len(train_loader)),
        }

        # Feature-space geometry diagnostics on accumulated foreground proposals.
        if geometry_feat_buffer:
            with torch.no_grad():
                all_feats = torch.cat(geometry_feat_buffer, dim=0)
                all_labels = torch.cat(geometry_label_buffer, dim=0)
                all_corr = torch.cat(geometry_corr_buffer, dim=0)

                # Subsample to buffer size to keep ID estimation stable.
                if all_feats.shape[0] > geometry_buffer_size:
                    perm = torch.randperm(all_feats.shape[0], device=all_feats.device)[:geometry_buffer_size]
                    all_feats = all_feats[perm]
                    all_labels = all_labels[perm]
                    all_corr = all_corr[perm]

                etf_weight = None
                predictor = model.roi_heads.box_predictor
                if isinstance(predictor, ManifoldCorrectionPredictor):
                    predictor = predictor.base_predictor
                cls_score = getattr(predictor, "cls_score", None)
                if isinstance(cls_score, ETFClassifier):
                    etf_weight = cls_score.weight

                geometry = compute_manifold_geometry(
                    all_feats,
                    all_labels,
                    num_classes,
                    corrected_features=all_corr,
                    method=args.id_method,
                    normalize=normalize_features,
                    etf_weight=etf_weight,
                    class_frequency_weights=class_weight_per_class,
                )
                row.update(scalar_geometry_report(geometry))

        # Periodic geometry metrics on prototype bank.
        if geometry_every > 0 and global_step % max(1, geometry_every) == 0:
            with torch.no_grad():
                all_ids = []
                for c in range(num_classes):
                    protos = prototype_bank.prototypes[c]
                    id_est = id_estimator.estimate_id(protos)
                    all_ids.append(float(id_est.item()))
                row["prototype_id_per_class"] = all_ids
                row["prototype_id_mean"] = sum(all_ids) / max(1, len(all_ids))

        # Validation and checkpointing.
        should_stop = False
        if eval_every > 0 and epoch % eval_every == 0:
            val_metrics = eval_metrics(model, val_loader, device, config, num_classes)
            val_ap50 = float(val_metrics["ap50"])
            val_ap75 = float(val_metrics.get("ap75", 0.0))
            row["val_ap50"] = val_ap50
            row["val_ap75"] = val_ap75
            row["val_ece"] = val_metrics.get("ece")
            row["val_metrics"] = val_metrics
            for key in ("per_class_ap50", "per_class_ap75"):
                if key in val_metrics:
                    row[f"val_{key}"] = val_metrics[key]
            print(
                f"Epoch {epoch}: val AP50/AP75 = {val_ap50:.4f}/{val_ap75:.4f} "
                f"(active initial {initial_ap50:.4f}/{initial_ap75:.4f})"
            )

            best_ap50, best_epoch, best_metrics, improved_ap50 = update_best_metric(
                metric_name="ap50",
                candidate_metrics=val_metrics,
                epoch=epoch,
                best_value=best_ap50,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
            )
            if improved_ap50:
                stale_epochs = 0
                save_checkpoint(
                    model,
                    run_dir / "checkpoint_best.pth",
                    checkpoint_metadata(
                        context,
                        {"epoch": epoch, "val_ap50": val_ap50, "val_ap75": val_ap75, **row},
                    ),
                )
            else:
                stale_epochs += 1

            best_ap75, best_ap75_epoch, best_ap75_metrics, improved_ap75 = update_best_metric(
                metric_name="ap75",
                candidate_metrics=val_metrics,
                epoch=epoch,
                best_value=best_ap75,
                best_epoch=best_ap75_epoch,
                best_metrics=best_ap75_metrics,
            )
            if improved_ap75:
                save_checkpoint(
                    model,
                    run_dir / "checkpoint_best_ap75.pth",
                    checkpoint_metadata(
                        context,
                        {"epoch": epoch, "val_ap50": val_ap50, "val_ap75": val_ap75, **row},
                    ),
                )

            if stale_epochs >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}; best val AP50 {best_ap50:.4f} at epoch {best_epoch}")
                should_stop = True

        append_jsonl(row, run_dir / "metrics_train.jsonl")
        print(row)

        if should_stop:
            break

        save_checkpoint(
            model,
            run_dir / "checkpoint_last.pth",
            checkpoint_metadata(context, {"epoch": epoch, **row}),
        )

    result = {
        "best_epoch": best_epoch,
        "best_val_ap50": best_ap50,
        "best_val_ap75": float(best_metrics.get("ap75", 0.0)),
        "best_metrics": best_metrics,
        "best_ap75_epoch": best_ap75_epoch,
        "best_ap75_selection_ap50": float(best_ap75_metrics.get("ap50", 0.0)),
        "best_ap75_selection_ap75": best_ap75,
        "best_ap75_metrics": best_ap75_metrics,
        "initial_val_ap50": initial_ap50,
        "initial_val_ap75": initial_ap75,
        "initial_metrics": initial_metrics,
        "raw_baseline_val_ap50": raw_baseline_ap50,
        "raw_baseline_val_ap75": raw_baseline_ap75,
        "raw_baseline_metrics": raw_baseline_metrics,
        "active_initial_val_ap50": initial_ap50,
        "active_initial_val_ap75": initial_ap75,
        "active_initial_metrics": initial_metrics,
        "total_epochs": completed_epochs,
        "lambda_tr": args.lambda_tr,
        "lambda_en": args.lambda_en,
        "lr": args.lr,
        "lr_manifold": args.lr_manifold,
        "use_gt_boxes": args.use_gt_boxes,
        "freeze_box_head": args.freeze_box_head,
        "freeze_box_predictor": args.freeze_box_predictor,
        "active_manifold_correction": args.active_manifold_correction,
        "active_correction_gamma": args.active_correction_gamma,
        "active_correction_gamma_schedule": args.active_correction_gamma_schedule,
        "active_correction_gamma_final": args.active_correction_gamma_final,
        "active_correction_mode": active_correction_mode,
        "active_endpoint_gate_init": args.active_endpoint_gate_init,
        "freeze_prototypes_after_warmup": args.freeze_prototypes_after_warmup,
        "lambda_fc1_rank": args.lambda_fc1_rank,
        "lambda_fc1_compact": args.lambda_fc1_compact,
        "fc1_rank_target": args.fc1_rank_target,
        "lambda_logit_preserve": args.lambda_logit_preserve,
        "lambda_bbox_preserve": args.lambda_bbox_preserve,
        "preserve_temperature": args.preserve_temperature,
        "lambda_proj_intra": args.lambda_proj_intra,
        "lambda_proto_div": args.lambda_proto_div,
        "lambda_proj_inter": args.lambda_proj_inter,
        "projection_inter_margin": args.projection_inter_margin,
        "proto_div_temperature": args.proto_div_temperature,
        "lambda_corrected_intra": args.lambda_corrected_intra,
        "lambda_corrected_inter": args.lambda_corrected_inter,
        "lambda_corrected_inter_preserve": args.lambda_corrected_inter_preserve,
        "lambda_corrected_center_preserve": args.lambda_corrected_center_preserve,
        "lambda_corrected_memory_center_preserve": args.lambda_corrected_memory_center_preserve,
        "lambda_corrected_memory_inter_preserve": args.lambda_corrected_memory_inter_preserve,
        "lambda_correction_field_preserve": args.lambda_correction_field_preserve,
        "corrected_memory_momentum": args.corrected_memory_momentum,
        "corrected_inter_margin": args.corrected_inter_margin,
        "box_head_type": get_box_head_type(model),
        "box_head_rank": args.box_head_rank,
        "box_head_conv_channels": args.box_head_conv_channels,
        "box_head_bottleneck_dim": args.box_head_bottleneck_dim,
        "box_head_attention_channels": args.box_head_attention_channels,
    }
    save_json(result, run_dir / "manifold_result.json")
    print(result)

    # Save manifold modules separately for inspection.
    manifold_payload = {
        "prototype_bank": prototype_bank.state_dict(),
        "transport_head": transport_head.state_dict() if transport_head is not None else None,
        "active_transport_head": active_transport_head.state_dict() if active_transport_head is not None else None,
        "config": {
            "num_classes": num_classes,
            "num_prototypes": args.num_prototypes,
            "feature_dim": feature_dim,
            "ema_decay": args.ema_decay,
            "sinkhorn_eps": args.sinkhorn_eps,
            "tau": args.tau,
            "lambda_tr": args.lambda_tr,
            "lambda_en": args.lambda_en,
            "active_manifold_correction": args.active_manifold_correction,
            "active_correction_gamma": args.active_correction_gamma,
            "active_correction_gamma_schedule": args.active_correction_gamma_schedule,
            "active_correction_gamma_final": args.active_correction_gamma_final,
            "active_correction_mode": active_correction_mode,
            "active_endpoint_gate_init": args.active_endpoint_gate_init,
            "freeze_prototypes_after_warmup": args.freeze_prototypes_after_warmup,
            "lambda_fc1_rank": args.lambda_fc1_rank,
            "lambda_fc1_compact": args.lambda_fc1_compact,
            "fc1_rank_target": args.fc1_rank_target,
            "lambda_logit_preserve": args.lambda_logit_preserve,
            "lambda_bbox_preserve": args.lambda_bbox_preserve,
            "preserve_temperature": args.preserve_temperature,
            "lambda_proj_intra": args.lambda_proj_intra,
            "lambda_proto_div": args.lambda_proto_div,
            "lambda_proj_inter": args.lambda_proj_inter,
            "projection_inter_margin": args.projection_inter_margin,
            "proto_div_temperature": args.proto_div_temperature,
            "lambda_corrected_intra": args.lambda_corrected_intra,
            "lambda_corrected_inter": args.lambda_corrected_inter,
            "lambda_corrected_inter_preserve": args.lambda_corrected_inter_preserve,
            "lambda_corrected_center_preserve": args.lambda_corrected_center_preserve,
            "lambda_corrected_memory_center_preserve": args.lambda_corrected_memory_center_preserve,
            "lambda_corrected_memory_inter_preserve": args.lambda_corrected_memory_inter_preserve,
            "lambda_correction_field_preserve": args.lambda_correction_field_preserve,
            "corrected_memory_momentum": args.corrected_memory_momentum,
            "corrected_inter_margin": args.corrected_inter_margin,
            "box_head_type": get_box_head_type(model),
            "box_head_rank": args.box_head_rank,
            "box_head_conv_channels": args.box_head_conv_channels,
            "box_head_bottleneck_dim": args.box_head_bottleneck_dim,
            "box_head_attention_channels": args.box_head_attention_channels,
        },
    }
    torch.save(manifold_payload, run_dir / "manifold_modules.pth")
    save_json(manifold_payload["config"], run_dir / "manifold_config.json")
    print(f"Saved manifold modules to {run_dir / 'manifold_modules.pth'}")


if __name__ == "__main__":
    main()
