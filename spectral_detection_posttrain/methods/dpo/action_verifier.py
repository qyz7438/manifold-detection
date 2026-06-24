from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionVerifierConfig:
    num_samples: int = 2
    sigma: float = 0.1
    seed: int | None = None
    include_identity_action: bool = False


@dataclass(frozen=True)
class ActionBatch:
    proposals: torch.Tensor
    deltas: torch.Tensor
    decoded_boxes: torch.Tensor
    log_probs: torch.Tensor


@dataclass(frozen=True)
class DpoPairs:
    chosen_indices: torch.Tensor
    rejected_indices: torch.Tensor
    valid: torch.Tensor


def gaussian_log_prob(deltas: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(deltas.device).clamp_min(1e-6)
    errors = (deltas - mu.unsqueeze(1).to(deltas.device)) / sigma.unsqueeze(1)
    return -0.5 * (errors.pow(2) + 2.0 * torch.log(sigma.unsqueeze(1)) + torch.log(deltas.new_tensor(2.0 * torch.pi))).sum(dim=-1)


def decode_box_actions(
    proposals: torch.Tensor,
    deltas: torch.Tensor,
    image_size: tuple[int, int],
    weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> torch.Tensor:
    if proposals.ndim != 2 or proposals.shape[-1] != 4:
        raise ValueError("proposals must have shape (N, 4)")
    if deltas.ndim != 3 or deltas.shape[-1] != 4 or deltas.shape[0] != proposals.shape[0]:
        raise ValueError("deltas must have shape (N, G, 4)")

    proposals = proposals.to(deltas.device).float()
    widths = (proposals[:, 2] - proposals[:, 0]).clamp_min(1e-6)
    heights = (proposals[:, 3] - proposals[:, 1]).clamp_min(1e-6)
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    dx = deltas[..., 0] / weights[0]
    dy = deltas[..., 1] / weights[1]
    dw = (deltas[..., 2] / weights[2]).clamp(max=4.135)
    dh = (deltas[..., 3] / weights[3]).clamp(max=4.135)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    height, width = image_size
    x1 = (pred_ctr_x - 0.5 * pred_w).clamp(min=0.0, max=float(width))
    y1 = (pred_ctr_y - 0.5 * pred_h).clamp(min=0.0, max=float(height))
    x2 = (pred_ctr_x + 0.5 * pred_w).clamp(min=0.0, max=float(width))
    y2 = (pred_ctr_y + 0.5 * pred_h).clamp(min=0.0, max=float(height))
    return torch.stack([x1, y1, x2, y2], dim=-1)


def build_action_batch(
    proposals: torch.Tensor,
    mu: torch.Tensor,
    image_size: tuple[int, int],
    cfg: ActionVerifierConfig | None = None,
) -> ActionBatch:
    cfg = cfg or ActionVerifierConfig()
    if proposals.shape != mu.shape:
        raise ValueError("proposals and mu must have matching shape (N, 4)")
    generator = torch.Generator(device=mu.device)
    if cfg.seed is not None:
        generator.manual_seed(int(cfg.seed))
    noise = torch.randn(
        (mu.shape[0], int(cfg.num_samples), 4),
        generator=generator,
        device=mu.device,
        dtype=mu.dtype,
    )
    sigma = torch.full_like(mu, float(cfg.sigma)).clamp_min(1e-6)
    deltas = mu.unsqueeze(1) + sigma.unsqueeze(1) * noise
    if bool(cfg.include_identity_action) and int(cfg.num_samples) > 0:
        deltas[:, 0, :] = 0.0
    elif int(cfg.num_samples) > 0:
        deltas[:, 0, :] = mu
    action_deltas = deltas.detach()
    decoded = decode_box_actions(proposals, action_deltas, image_size=image_size)
    log_probs = gaussian_log_prob(action_deltas, mu, sigma)
    return ActionBatch(
        proposals=proposals.detach(),
        deltas=action_deltas,
        decoded_boxes=decoded,
        log_probs=log_probs,
    )


def _crop_and_resize(image: torch.Tensor, box: torch.Tensor, crop_size: int) -> torch.Tensor:
    x1 = max(0, int(torch.floor(box[0]).item()))
    y1 = max(0, int(torch.floor(box[1]).item()))
    x2 = min(image.shape[-1], max(x1 + 1, int(torch.ceil(box[2]).item())))
    y2 = min(image.shape[-2], max(y1 + 1, int(torch.ceil(box[3]).item())))
    if x2 <= x1 or y2 <= y1:
        crop = torch.zeros((image.shape[0], 1, 1), dtype=image.dtype, device=image.device)
    else:
        crop = image[:, y1:y2, x1:x2]
    return F.interpolate(crop.unsqueeze(0).float(), size=(crop_size, crop_size), mode="bilinear", align_corners=False).squeeze(0)


def compute_fft_action_quality(image: torch.Tensor, decoded_boxes: torch.Tensor, crop_size: int = 32) -> torch.Tensor:
    if decoded_boxes.ndim != 3 or decoded_boxes.shape[-1] != 4:
        raise ValueError("decoded_boxes must have shape (N, G, 4)")
    flat_boxes = decoded_boxes.reshape(-1, 4).detach().cpu()
    image_cpu = image.detach().cpu()
    crops = torch.stack([_crop_and_resize(image_cpu, box, crop_size) for box in flat_boxes])
    amp = torch.abs(torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho"))
    energy = torch.log1p(amp.pow(2).mean(dim=(-3, -2, -1)))
    area = ((flat_boxes[:, 2] - flat_boxes[:, 0]).clamp_min(0) * (flat_boxes[:, 3] - flat_boxes[:, 1]).clamp_min(0))
    image_area = float(image.shape[-1] * image.shape[-2])
    area_score = (area / max(image_area, 1.0)).clamp(0.0, 1.0)
    quality = energy * area_score
    return quality.reshape(decoded_boxes.shape[0], decoded_boxes.shape[1]).to(decoded_boxes.device)


def compute_manifold_action_quality(features: torch.Tensor, reference_features: torch.Tensor, k: int = 5) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("features must have shape (N, G, D)")
    if reference_features.ndim != 2:
        raise ValueError("reference_features must have shape (M, D)")
    flat = features.reshape(-1, features.shape[-1]).float()
    reference = reference_features.to(flat.device).float()
    if reference.numel() == 0:
        return torch.zeros(features.shape[:2], dtype=features.dtype, device=features.device)
    distances = torch.cdist(flat, reference)
    k_eff = min(max(1, int(k)), reference.shape[0])
    nn_distance = distances.topk(k_eff, largest=False, dim=1).values.mean(dim=1)
    return (-nn_distance).reshape(features.shape[0], features.shape[1]).to(features.device)


def build_rlvr_rewards(
    iou: torch.Tensor,
    verifier_quality: torch.Tensor,
    matched: torch.Tensor,
    verifier_weight: float = 0.1,
    high_conf_fp_penalty: float = 0.0,
    scores: torch.Tensor | None = None,
    high_conf_threshold: float = 0.8,
) -> torch.Tensor:
    positive = iou.float() + float(verifier_weight) * verifier_quality.float()
    reward = positive * matched.float()
    if scores is not None and high_conf_fp_penalty:
        penalty = ((~matched.bool()) & (scores >= high_conf_threshold)).float()
        reward = reward - float(high_conf_fp_penalty) * penalty
    return reward


def normalize_group_advantage(rewards: torch.Tensor, eps: float = 1e-6, clamp: float = 3.0) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    return ((rewards - mean) / std).clamp(min=-float(clamp), max=float(clamp))


def build_dpo_pairs(quality: torch.Tensor, margin: float = 0.05) -> DpoPairs:
    if quality.ndim != 2:
        raise ValueError("quality must have shape (N, G)")
    if quality.shape[1] != 2:
        raise ValueError("DPO pair construction currently expects exactly two actions")
    diff = quality[:, 0] - quality[:, 1]
    chosen = torch.where(diff >= 0, torch.zeros_like(diff, dtype=torch.long), torch.ones_like(diff, dtype=torch.long))
    rejected = 1 - chosen
    valid = diff.abs() > float(margin)
    return DpoPairs(chosen_indices=chosen, rejected_indices=rejected, valid=valid)


def dpo_loss_from_log_probs(
    log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    pairs: DpoPairs,
    beta: float = 0.5,
) -> torch.Tensor:
    if not pairs.valid.any():
        return log_probs.sum() * 0.0
    row = torch.arange(log_probs.shape[0], device=log_probs.device)
    chosen = pairs.chosen_indices.to(log_probs.device)
    rejected = pairs.rejected_indices.to(log_probs.device)
    valid = pairs.valid.to(log_probs.device)
    policy_margin = log_probs[row, chosen] - log_probs[row, rejected]
    reference_margin = reference_log_probs.to(log_probs.device)[row, chosen] - reference_log_probs.to(log_probs.device)[row, rejected]
    return -F.logsigmoid(float(beta) * (policy_margin - reference_margin)[valid]).mean()
