from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


AMP_SIGNALS = {"ramp", "shuffled_ramp", "shuffled_amp", "amp_structure", "shuffled_amp_structure"}
STRUCTURE_SIGNALS = {"structure", "shuffled_structure", "amp_structure", "shuffled_amp_structure"}


def signal_uses_amp(signal: str) -> bool:
    return signal in AMP_SIGNALS


def signal_uses_structure(signal: str) -> bool:
    return signal in STRUCTURE_SIGNALS


@dataclass(frozen=True)
class DetectionVerifierConfig:
    signal: str = "none"
    temperature: float = 1.0
    w_iou: float = 1.0
    w_cls: float = 0.2
    w_amp: float = 0.1
    w_struct: float = 0.0
    w_hconf_fp: float = 0.5
    high_conf_threshold: float = 0.8


def compute_box_rewards(
    cfg: DetectionVerifierConfig,
    ious: torch.Tensor,
    class_correct: torch.Tensor,
    scores: torch.Tensor,
    matched: torch.Tensor,
    s_amp: torch.Tensor | None = None,
    s_struct: torch.Tensor | None = None,
) -> torch.Tensor:
    amp = torch.zeros_like(ious) if s_amp is None else s_amp.to(ious.device).float()
    struct = torch.zeros_like(ious) if s_struct is None else s_struct.to(ious.device).float()
    amp_weight = cfg.w_amp if signal_uses_amp(cfg.signal) else 0.0
    struct_weight = cfg.w_struct if signal_uses_structure(cfg.signal) else 0.0
    high_conf_fp = ((~matched) & (scores >= cfg.high_conf_threshold)).float()
    positive_reward = cfg.w_iou * ious + cfg.w_cls * class_correct + amp_weight * amp + struct_weight * struct
    reward = positive_reward * matched.float()
    reward = reward - cfg.w_hconf_fp * high_conf_fp
    return reward


def normalize_group_advantages(rewards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    std = rewards.std(unbiased=False)
    advantages = (rewards - rewards.mean()) / (std + 1e-6)
    return F.softplus(advantages / max(float(temperature), 1e-6)).clamp(0.05, 5.0)


def shuffle_tp_values(values: torch.Tensor, matched: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    out = values.clone()
    tp_idx = torch.where(matched)[0]
    if tp_idx.numel() < 2:
        out[~matched] = 0.0
        return out
    generator = torch.Generator(device=values.device)
    if seed is not None:
        generator.manual_seed(seed)
    perm = tp_idx[torch.randperm(tp_idx.numel(), generator=generator, device=values.device)]
    out[tp_idx] = values[perm]
    out[~matched] = 0.0
    return out


def shuffle_tp_ramp(values: torch.Tensor, matched: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    return shuffle_tp_values(values, matched, seed=seed)


def _box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1 = boxes1.cpu() if boxes1.device.type != "cpu" else boxes1
    boxes2 = boxes2.cpu() if boxes2.device.type != "cpu" else boxes2
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2 - inter).clamp_min(1e-6)


def build_rewarded_roi_actions(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    num_classes: int,
    max_candidates: int = 40,
    reward_score_threshold: float = 0.2,
    verifier_cfg: DetectionVerifierConfig | None = None,
    s_amp: torch.Tensor | None = None,
    s_struct: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cfg = verifier_cfg or DetectionVerifierConfig()
    boxes_all = prediction["boxes"].detach().cpu()
    pred_labels_all = prediction["labels"].detach().cpu()
    scores_all = prediction["scores"].detach().cpu()
    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    # unified keep mask and order (s_amp and s_struct must use same mask)
    keep = scores_all >= reward_score_threshold
    boxes = boxes_all[keep]
    pred_labels = pred_labels_all[keep]
    scores = scores_all[keep]
    if s_amp is not None:
        s_amp = s_amp[keep]
    if s_struct is not None:
        s_struct = s_struct[keep]
    if len(scores) > max_candidates:
        order = torch.argsort(scores, descending=True)[:max_candidates]
        boxes = boxes[order]
        pred_labels = pred_labels[order]
        scores = scores[order]
        if s_amp is not None:
            s_amp = s_amp[order]
        if s_struct is not None:
            s_struct = s_struct[order]

    if boxes.numel() == 0:
        empty_boxes = boxes.new_empty((0, 4))
        empty_float = scores.new_empty((0,), dtype=torch.float32)
        empty_long = pred_labels.new_empty((0,), dtype=torch.long)
        return {
            "boxes": empty_boxes,
            "labels": empty_long,
            "policy_labels": empty_long.clone(),
            "matched_gt_boxes": empty_boxes.clone(),
            "weights": empty_float,
            "advantages": empty_float.clone(),
            "rewards": empty_float.clone(),
            "matched": torch.empty((0,), dtype=torch.bool),
            "scores": empty_float.clone(),
            "amp_values": empty_float.clone(),
            "structure_values": empty_float.clone(),
        }

    if gt_boxes.numel() == 0:
        best_iou = torch.zeros((boxes.shape[0],), dtype=torch.float32)
        best_gt = torch.zeros((boxes.shape[0],), dtype=torch.long)
        matched = torch.zeros((boxes.shape[0],), dtype=torch.bool)
        matched_gt_boxes = torch.zeros((boxes.shape[0], 4), dtype=boxes.dtype)
    else:
        ious = _box_iou_matrix(boxes, gt_boxes)
        best_iou, best_gt = ious.max(dim=1)
        matched = best_iou >= 0.5
        matched_gt_boxes = gt_boxes[best_gt]

    # supervised labels: GT-matched truth for CE training
    supervised_labels = torch.zeros_like(pred_labels)
    if matched.any():
        supervised_labels[matched] = gt_labels[best_gt[matched]].clamp(max=num_classes - 1)

    # policy labels: the model's own action (what it predicted)
    policy_labels = pred_labels.clamp(min=0, max=num_classes - 1)

    class_correct = (pred_labels == supervised_labels).float() * matched.float()
    amp = torch.zeros_like(best_iou) if s_amp is None else s_amp.to(best_iou.device)
    struct = torch.zeros_like(best_iou) if s_struct is None else s_struct.to(best_iou.device)

    # shuffle controls for each verifier component
    if cfg.signal in {"shuffled_ramp", "shuffled_amp"}:
        amp = shuffle_tp_values(amp, matched)
    if cfg.signal == "shuffled_structure":
        struct = shuffle_tp_values(struct, matched)
    if cfg.signal == "shuffled_amp_structure":
        amp = shuffle_tp_values(amp, matched)
        struct = shuffle_tp_values(struct, matched)
    amp = amp * matched.float()
    struct = struct * matched.float()

    rewards = compute_box_rewards(cfg, best_iou, class_correct, scores, matched, s_amp=amp, s_struct=struct)

    # signed advantages for policy gradient (not softplus weights)
    std = rewards.std(unbiased=False)
    advantages = (rewards - rewards.mean()) / (std + 1e-6)
    advantages = advantages / max(float(cfg.temperature), 1e-6)
    advantages = advantages.clamp(-3.0, 3.0)

    # softplus weights kept for legacy weighted_ce
    policy_weights = normalize_group_advantages(rewards, temperature=cfg.temperature)

    # recovery boxes are NOT model actions — do not enter policy_labels or advantages
    # (Round 2.2 default: recovery_loss_weight=0, skip entirely)

    return {
        "boxes": boxes,
        "labels": supervised_labels.long(),
        "policy_labels": policy_labels.long(),
        "matched_gt_boxes": matched_gt_boxes,
        "weights": policy_weights.float(),
        "advantages": advantages.float(),
        "rewards": rewards.float(),
        "matched": matched.bool(),
        "scores": scores.float(),
        "amp_values": amp.float(),
        "structure_values": struct.float(),
    }


def _cat_action_tensor(actions: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    tensors = [action[key].detach().float().cpu() for action in actions if key in action and action[key].numel() > 0]
    if not tensors:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(tensors, dim=0)


def build_reward_component_summary(actions: list[dict[str, torch.Tensor]]) -> dict[str, float | int]:
    amp = _cat_action_tensor(actions, "amp_values")
    struct = _cat_action_tensor(actions, "structure_values")
    rewards = _cat_action_tensor(actions, "rewards")
    matched = _cat_action_tensor(actions, "matched").bool()

    def _mean(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else 0.0

    def _std(values: torch.Tensor) -> float:
        return float(values.std(unbiased=False).item()) if values.numel() else 0.0

    return {
        "candidate_count": int(rewards.numel()),
        "matched_count": int(matched.sum().item()) if matched.numel() else 0,
        "amp_mean": _mean(amp), "amp_std": _std(amp),
        "structure_mean": _mean(struct), "structure_std": _std(struct),
        "reward_mean": _mean(rewards), "reward_std": _std(rewards),
    }
