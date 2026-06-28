from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.signals.fft.fft_features import compute_fft_amplitude
from spectral_detection_posttrain.signals.fft.radial_profile import radial_profile
from spectral_detection_posttrain.signals.fft.roi_crop import crop_and_resize_roi


def spectral_reward(roi_pred: torch.Tensor, roi_gt: torch.Tensor, num_bins: int = 32) -> float:
    amp_pred = compute_fft_amplitude(roi_pred)
    amp_gt = compute_fft_amplitude(roi_gt)
    profile_pred = radial_profile(amp_pred, num_bins=num_bins)
    profile_gt = radial_profile(amp_gt, num_bins=num_bins)
    cosine = F.cosine_similarity(profile_pred, profile_gt, dim=0).clamp(-1.0, 1.0)
    distance = 1.0 - cosine
    return float(torch.exp(-distance).item())


def prediction_reward(
    image: torch.Tensor,
    pred_box: torch.Tensor,
    gt_box: torch.Tensor,
    iou: float,
    score: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    eta: float = 0.5,
) -> dict:
    roi_pred = crop_and_resize_roi(image, pred_box)
    roi_gt = crop_and_resize_roi(image, gt_box)
    r_amp = spectral_reward(roi_pred, roi_gt)
    raw = alpha * 1.0 + beta * iou + gamma * r_amp
    normalized = raw / (alpha + beta + gamma)
    reward = max(0.0, min(1.0, normalized - eta * 0.0 * score))
    return {"reward": reward, "r_amp": r_amp}


def compute_prediction_rewards(
    image: torch.Tensor,
    prediction: dict,
    target: dict,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> dict:
    matched = match_predictions_to_gt(prediction, target, iou_threshold=iou_threshold, score_threshold=score_threshold)
    tp_rewards = []
    fp_rewards = []
    pred_boxes = prediction.get("boxes", torch.empty((0, 4))).detach().cpu()
    pred_scores = prediction.get("scores", torch.ones((len(pred_boxes),))).detach().cpu()
    gt_boxes = target.get("boxes", torch.empty((0, 4))).detach().cpu()

    for match in matched["matches"]:
        pred_idx = match["pred_index"]
        gt_idx = match["gt_index"]
        values = prediction_reward(
            image.cpu(),
            pred_boxes[pred_idx],
            gt_boxes[gt_idx],
            match["iou"],
            match["score"],
        )
        tp_rewards.append(values["r_amp"])

    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        ious = box_iou(pred_boxes, gt_boxes)
        for pred_idx in matched["unmatched_predictions"]:
            best_gt = int(ious[pred_idx].argmax().item())
            roi_pred = crop_and_resize_roi(image.cpu(), pred_boxes[pred_idx])
            roi_gt = crop_and_resize_roi(image.cpu(), gt_boxes[best_gt])
            fp_rewards.append(spectral_reward(roi_pred, roi_gt))

    return {"tp_r_amp": tp_rewards, "fp_r_amp": fp_rewards}


def auc_tp_vs_fp(tp_values: list[float], fp_values: list[float]) -> float | None:
    if not tp_values or not fp_values:
        return None
    wins = 0.0
    total = len(tp_values) * len(fp_values)
    for tp in tp_values:
        for fp in fp_values:
            if tp > fp:
                wins += 1.0
            elif math.isclose(tp, fp):
                wins += 0.5
    return wins / total
