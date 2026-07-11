"""Candidate-local sign and top-focused rank diagnostics for frozen B3 heads."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


def median_sign_metrics(
    scores: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor
) -> dict[str, float | int]:
    prediction, utility, ids = _rows(scores, target, image_ids)
    image_aucs: list[float] = []
    correct_pairs = 0.0
    pair_count = 0
    true_positive = true_negative = positive_count = negative_count = 0
    valid_candidates = 0
    for image_id in torch.unique(ids, sorted=True):
        mask = ids == image_id
        image_scores, image_target = prediction[mask], utility[mask]
        target_median = torch.quantile(image_target, 0.5)
        positive = image_target > target_median
        negative = image_target < target_median
        if not positive.any() or not negative.any():
            continue
        comparisons = image_scores[positive, None] - image_scores[None, negative]
        image_correct = float(
            (
                comparisons.gt(0).float()
                + 0.5 * comparisons.eq(0).float()
            ).sum().item()
        )
        image_pairs = int(comparisons.numel())
        image_aucs.append(image_correct / image_pairs)
        correct_pairs += image_correct
        pair_count += image_pairs
        valid = positive | negative
        score_median = torch.quantile(image_scores, 0.5)
        predicted_positive = image_scores > score_median
        true_positive += int((predicted_positive & positive).sum().item())
        true_negative += int((~predicted_positive & negative).sum().item())
        positive_count += int(positive.sum().item())
        negative_count += int(negative.sum().item())
        valid_candidates += int(valid.sum().item())
    if not image_aucs:
        raise ValueError("median sign audit has no images with strict median pairs")
    recall = true_positive / max(1, positive_count)
    specificity = true_negative / max(1, negative_count)
    return {
        "valid_candidate_count": valid_candidates,
        "valid_image_count": len(image_aucs),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "within_image_pair_count": pair_count,
        "auroc_equal_image": sum(image_aucs) / len(image_aucs),
        "auroc_candidate_weighted": correct_pairs / pair_count,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "positive_recall": recall,
        "negative_recall": specificity,
    }


def top_focused_rank_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
    min_target_gap: float,
    lcb_z: float,
) -> dict[str, float | int]:
    prediction, utility, ids = _rows(scores, target, image_ids)
    if min(target_epsilon, min_target_gap, lcb_z) < 0.0:
        raise ValueError("top-focused thresholds must be non-negative")
    cross_aucs: list[float] = []
    cross_correct = 0.0
    cross_pairs = 0
    oracle_aucs: list[float] = []
    oracle_correct = 0.0
    oracle_pairs = 0
    top_values: list[float] = []
    regrets: list[float] = []
    top_positive = 0
    random_positive_rate = 0.0
    exact_oracle = 0
    unique_images = torch.unique(ids, sorted=True)
    for image_id in unique_images:
        mask = ids == image_id
        image_scores, image_target = prediction[mask], utility[mask]
        positive = image_target > float(target_epsilon)
        negative = ~positive
        if positive.any() and negative.any():
            comparisons = image_scores[positive, None] - image_scores[None, negative]
            correct = float(
                (
                    comparisons.gt(0).float()
                    + 0.5 * comparisons.eq(0).float()
                ).sum().item()
            )
            count = int(comparisons.numel())
            cross_aucs.append(correct / count)
            cross_correct += correct
            cross_pairs += count
        oracle_index = int(image_target.argmax().item())
        oracle_value = float(image_target[oracle_index].item())
        eligible = oracle_value - image_target > float(min_target_gap)
        if eligible.any():
            comparisons = image_scores[oracle_index] - image_scores[eligible]
            correct = float(
                (
                    comparisons.gt(0).float()
                    + 0.5 * comparisons.eq(0).float()
                ).sum().item()
            )
            count = int(comparisons.numel())
            oracle_aucs.append(correct / count)
            oracle_correct += correct
            oracle_pairs += count
        predicted_index = int(image_scores.argmax().item())
        selected_value = float(image_target[predicted_index].item())
        top_values.append(selected_value)
        top_positive += int(selected_value > float(target_epsilon))
        random_positive_rate += float(positive.float().mean().item())
        exact_oracle += int(oracle_value - selected_value <= float(min_target_gap))
        regrets.append(max(0.0, oracle_value) - selected_value)
    image_values = torch.tensor(top_values, dtype=torch.float64)
    mean_delta = float(image_values.mean().item())
    standard_error = (
        float(image_values.std(unbiased=True).item()) / math.sqrt(image_values.numel())
        if image_values.numel() > 1
        else 0.0
    )
    image_count = int(unique_images.numel())
    hit_rate = top_positive / image_count
    random_hit_rate = random_positive_rate / image_count
    return {
        "image_count": image_count,
        "cross_boundary_image_count": len(cross_aucs),
        "cross_boundary_pair_count": cross_pairs,
        "cross_boundary_pairwise_equal_image": sum(cross_aucs) / max(1, len(cross_aucs)),
        "cross_boundary_pairwise_candidate_weighted": cross_correct / max(1, cross_pairs),
        "oracle_top_image_count": len(oracle_aucs),
        "oracle_top_pair_count": oracle_pairs,
        "oracle_top_pairwise_equal_image": sum(oracle_aucs) / max(1, len(oracle_aucs)),
        "oracle_top_pairwise_candidate_weighted": oracle_correct / max(1, oracle_pairs),
        "predicted_top1_positive_count": top_positive,
        "predicted_top1_positive_hit_rate": hit_rate,
        "random_top1_positive_hit_rate": random_hit_rate,
        "top1_positive_hit_lift": hit_rate - random_hit_rate,
        "top1_exact_oracle_rate": exact_oracle / image_count,
        "top1_mean_delta_u": mean_delta,
        "top1_mean_delta_u_standard_error": standard_error,
        "top1_mean_delta_u_lcb": mean_delta - float(lcb_z) * standard_error,
        "top1_oracle_regret_mean": sum(regrets) / image_count,
    }


def evaluate_top_focused_gates(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    support = payload["support"]
    arms = payload["arms"]
    full = arms["local_full"]
    sign_keys = ("auroc_equal_image", "auroc_candidate_weighted")
    rank_keys = (
        "cross_boundary_pairwise_equal_image",
        "cross_boundary_pairwise_candidate_weighted",
        "oracle_top_pairwise_equal_image",
        "oracle_top_pairwise_candidate_weighted",
    )
    controls = ("within_image_feature_shuffle", "within_image_label_shuffle")
    sign_values = [float(full["median_sign"][key]) for key in sign_keys]
    rank_values = [float(full["top_rank"][key]) for key in rank_keys]
    sign_gains = [
        sign_values[index] - float(arms[control]["median_sign"][key])
        for control in controls
        for index, key in enumerate(sign_keys)
    ]
    rank_gains = [
        rank_values[index] - float(arms[control]["top_rank"][key])
        for control in controls
        for index, key in enumerate(rank_keys)
    ]
    top_rank = full["top_rank"]
    gate_values = {
        "G0_support": (
            int(support["candidate_count"]) >= int(gates["min_candidates"])
            and int(support["image_count"]) >= int(gates["min_images"])
            and int(full["median_sign"]["within_image_pair_count"])
            >= int(gates["min_median_pairs"])
            and int(top_rank["cross_boundary_pair_count"])
            >= int(gates["min_cross_boundary_pairs"])
            and int(top_rank["oracle_top_pair_count"])
            >= int(gates["min_oracle_top_pairs"])
        ),
        "G1_candidate_local_median_sign": (
            min(sign_values) >= float(gates["min_median_sign_auroc"])
            and min(sign_gains) >= float(gates["min_control_gain"])
        ),
        "G2_top_focused_rank": (
            min(rank_values) >= float(gates["min_rank_pairwise"])
            and min(rank_gains) >= float(gates["min_control_gain"])
        ),
        "G3_actionable_top1": (
            float(top_rank["top1_positive_hit_lift"])
            > float(gates["min_top1_positive_hit_lift"])
            and float(top_rank["top1_mean_delta_u_lcb"])
            > float(gates["min_top1_mean_delta_u_lcb"])
        ),
    }
    return {
        "all_passed": all(gate_values.values()),
        "gates": gate_values,
        "diagnostics": {
            "minimum_median_sign_control_gain": min(sign_gains),
            "minimum_top_rank_control_gain": min(rank_gains),
            "minimum_median_sign_auroc": min(sign_values),
            "minimum_top_rank_pairwise": min(rank_values),
        },
    }


def _rows(
    scores: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.as_tensor(scores).flatten()
    utility = torch.as_tensor(target, dtype=prediction.dtype, device=prediction.device).flatten()
    ids = torch.as_tensor(image_ids, dtype=torch.long, device=prediction.device).flatten()
    if prediction.numel() == 0 or prediction.shape != utility.shape or utility.shape != ids.shape:
        raise ValueError("scores, target, and image ids must share non-empty rows")
    if not torch.isfinite(prediction).all() or not torch.isfinite(utility).all():
        raise ValueError("scores and target must be finite")
    return prediction, utility, ids
