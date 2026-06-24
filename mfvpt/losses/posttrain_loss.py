from __future__ import annotations

import torch
import torch.nn.functional as F

from mfvpt.losses.confidence import high_confidence_error_penalty
from mfvpt.losses.view_consistency import kl_view_consistency_loss
from mfvpt.models import normalize_for_imagenet
from mfvpt.transforms.fourier import high_freq_perturb, low_pass_filter
from mfvpt.transforms.patch import add_patch


def compute_posttrain_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: dict,
) -> dict[str, torch.Tensor]:
    perturb_cfg = config["perturb"]
    posttrain_cfg = config["posttrain"]
    images_low = low_pass_filter(images, ratio=float(perturb_cfg.get("low_ratio", 0.25)))
    images_high = high_freq_perturb(
        images,
        strength=float(perturb_cfg.get("high_strength", 0.10)),
        ratio=float(perturb_cfg.get("high_ratio", 0.25)),
    )
    images_patch = add_patch(
        images,
        patch_type=str(perturb_cfg.get("patch_type", "random")),
        patch_size=int(perturb_cfg.get("patch_size", 32)),
    )

    logits_ori = model(normalize_for_imagenet(images))
    logits_low = model(normalize_for_imagenet(images_low))
    logits_high = model(normalize_for_imagenet(images_high))
    logits_patch = model(normalize_for_imagenet(images_patch))

    loss_ce = (
        F.cross_entropy(logits_ori, labels)
        + F.cross_entropy(logits_low, labels)
        + F.cross_entropy(logits_high, labels)
        + F.cross_entropy(logits_patch, labels)
    ) / 4.0
    loss_view_consistency = (
        kl_view_consistency_loss(logits_ori, logits_low)
        + kl_view_consistency_loss(logits_ori, logits_high)
        + kl_view_consistency_loss(logits_ori, logits_patch)
    ) / 3.0
    loss_confidence = high_confidence_error_penalty(logits_patch, labels)
    lambda_view_consistency = float(
        posttrain_cfg.get("lambda_view_consistency", posttrain_cfg.get("lambda_consistency", 1.0))
    )
    loss_total = (
        loss_ce
        + lambda_view_consistency * loss_view_consistency
        + float(posttrain_cfg.get("lambda_confidence", 0.5)) * loss_confidence
    )
    return {
        "loss_total": loss_total,
        "loss_ce": loss_ce,
        "loss_view_consistency": loss_view_consistency,
        "loss_confidence": loss_confidence,
    }
