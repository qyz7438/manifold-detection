"""Train-cache diagnostics for decomposed actionability, rank, and abstention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class PairwiseRows:
    features: torch.Tensor
    target: torch.Tensor
    image_ids: torch.Tensor


def within_image_pairwise_rows(
    features: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    min_target_gap: float,
) -> PairwiseRows:
    x, y, ids = _rows(features, target, image_ids)
    if min_target_gap < 0.0:
        raise ValueError("min_target_gap must be non-negative")
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    image_rows: list[torch.Tensor] = []
    for image_id in torch.unique(ids, sorted=True):
        rows = torch.nonzero(ids == image_id, as_tuple=False).flatten()
        if rows.numel() < 2:
            continue
        left, right = torch.triu_indices(
            rows.numel(), rows.numel(), offset=1, device=rows.device
        )
        left_rows, right_rows = rows[left], rows[right]
        gaps = y[left_rows] - y[right_rows]
        keep = gaps.abs() > float(min_target_gap)
        if not keep.any():
            continue
        feature_rows.append(x[left_rows[keep]] - x[right_rows[keep]])
        target_rows.append(torch.where(gaps[keep] > 0.0, 1.0, -1.0))
        image_rows.append(
            torch.full(
                (int(keep.sum().item()),),
                int(image_id.item()),
                dtype=ids.dtype,
                device=ids.device,
            )
        )
    if not feature_rows:
        raise ValueError("no non-tied within-image pairs")
    return PairwiseRows(
        features=torch.cat(feature_rows),
        target=torch.cat(target_rows),
        image_ids=torch.cat(image_rows),
    )


def class_image_balanced_weights(
    sign_target: torch.Tensor, image_ids: torch.Tensor
) -> torch.Tensor:
    target = torch.as_tensor(sign_target).flatten()
    ids = torch.as_tensor(image_ids, dtype=torch.long, device=target.device).flatten()
    if target.shape != ids.shape or target.numel() == 0:
        raise ValueError("sign target and image ids must share non-empty rows")
    if not torch.all((target == 1.0) | (target == -1.0)):
        raise ValueError("sign target must contain only -1 and +1")
    weights = torch.empty_like(target, dtype=torch.float32)
    for image_id in torch.unique(ids, sorted=True):
        image_mask = ids == image_id
        weights[image_mask] = 1.0 / float(image_mask.sum().item())
    for value in (-1.0, 1.0):
        mask = target.eq(value)
        class_mass = weights[mask].sum()
        if class_mass <= 0.0:
            raise ValueError("sign target must contain both classes")
        weights[mask] *= 0.5 / class_mass
    return weights


def sign_classification_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
) -> dict[str, float | int]:
    prediction, utility, ids = _utility_scores(scores, target, image_ids)
    if target_epsilon < 0.0:
        raise ValueError("target_epsilon must be non-negative")
    positive = utility > float(target_epsilon)
    predicted_positive = prediction > 0.0
    true_positive = int((predicted_positive & positive).sum().item())
    true_negative = int((~predicted_positive & ~positive).sum().item())
    positive_count = int(positive.sum().item())
    negative_count = int((~positive).sum().item())
    recall = true_positive / max(1, positive_count)
    specificity = true_negative / max(1, negative_count)
    candidate_auc = _binary_auc(prediction, positive)
    image_aucs: list[float] = []
    for image_id in torch.unique(ids, sorted=True):
        mask = ids == image_id
        if positive[mask].any() and (~positive[mask]).any():
            image_aucs.append(_binary_auc(prediction[mask], positive[mask]))
    precision = true_positive / max(1, int(predicted_positive.sum().item()))
    prevalence = positive_count / utility.numel()
    return {
        "candidate_count": int(utility.numel()),
        "image_count": int(torch.unique(ids).numel()),
        "positive_count": positive_count,
        "positive_prevalence": prevalence,
        "always_negative_accuracy": negative_count / utility.numel(),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "positive_precision": precision,
        "positive_precision_lift": precision - prevalence,
        "positive_recall": recall,
        "auroc_equal_image": sum(image_aucs) / max(1, len(image_aucs)),
        "auroc_candidate_weighted": candidate_auc,
        "mixed_class_image_count": len(image_aucs),
    }


def ranked_abstention_metrics(
    rank_scores: torch.Tensor,
    sign_scores: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    sign_threshold: float,
    target_epsilon: float,
    lcb_z: float,
) -> dict[str, float | int]:
    rank, utility, ids = _utility_scores(rank_scores, target, image_ids)
    sign = torch.as_tensor(sign_scores, dtype=rank.dtype, device=rank.device).flatten()
    if sign.shape != rank.shape or not torch.isfinite(sign).all():
        raise ValueError("sign scores must be finite and align with rank scores")
    if target_epsilon < 0.0 or lcb_z < 0.0:
        raise ValueError("target_epsilon and lcb_z must be non-negative")
    per_image: list[float] = []
    selected: list[float] = []
    regrets: list[float] = []
    selected_positive = 0
    for image_id in torch.unique(ids, sorted=True):
        mask = ids == image_id
        image_rank, image_sign, image_target = rank[mask], sign[mask], utility[mask]
        best = int(image_rank.argmax().item())
        acts = float(image_sign[best].item()) > float(sign_threshold)
        delta = float(image_target[best].item()) if acts else 0.0
        if acts:
            selected.append(delta)
            selected_positive += int(delta > float(target_epsilon))
        per_image.append(delta)
        regrets.append(max(0.0, float(image_target.max().item())) - delta)
    image_values = torch.tensor(per_image, dtype=torch.float64)
    mean_delta = float(image_values.mean().item())
    standard_error = (
        float(image_values.std(unbiased=True).item()) / math.sqrt(image_values.numel())
        if image_values.numel() > 1
        else 0.0
    )
    positive_prevalence = float(utility.gt(float(target_epsilon)).float().mean().item())
    selected_count = len(selected)
    selected_precision = selected_positive / max(1, selected_count)
    return {
        "sign_threshold": float(sign_threshold),
        "candidate_count": int(utility.numel()),
        "image_count": int(torch.unique(ids).numel()),
        "selected_count": selected_count,
        "selected_positive_count": selected_positive,
        "selected_positive_precision": selected_precision,
        "candidate_positive_prevalence": positive_prevalence,
        "positive_precision_lift": selected_precision - positive_prevalence,
        "action_image_rate": selected_count / max(1, image_values.numel()),
        "mean_delta_u_per_image": mean_delta,
        "mean_delta_u_selected": sum(selected) / max(1, selected_count),
        "mean_delta_u_standard_error": standard_error,
        "mean_delta_u_lcb": mean_delta - float(lcb_z) * standard_error,
        "oracle_regret_mean": sum(regrets) / max(1, len(regrets)),
    }


def calibrate_ranked_abstention(
    rank_scores: torch.Tensor,
    sign_scores: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
    min_selected_actions: int,
    max_action_image_rate: float,
    min_positive_precision_lift: float,
    min_mean_delta_u_lcb: float,
    lcb_z: float,
) -> dict[str, Any]:
    rank, utility, ids = _utility_scores(rank_scores, target, image_ids)
    sign = torch.as_tensor(sign_scores, dtype=rank.dtype, device=rank.device).flatten()
    if sign.shape != rank.shape or not torch.isfinite(sign).all():
        raise ValueError("sign scores must be finite and align with rank scores")
    if min_selected_actions <= 0 or not 0.0 <= max_action_image_rate <= 1.0:
        raise ValueError("invalid abstention calibration constraints")
    image_sign_maxima: list[float] = []
    for image_id in torch.unique(ids, sorted=True):
        mask = ids == image_id
        best = int(rank[mask].argmax().item())
        image_sign_maxima.append(float(sign[mask][best].item()))
    unique_scores = sorted(set(image_sign_maxima))
    epsilon = max(1e-7, float(torch.finfo(torch.float32).eps))
    thresholds = [unique_scores[0] - epsilon, *unique_scores]
    feasible: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for threshold in thresholds:
        metrics = ranked_abstention_metrics(
            rank,
            sign,
            utility,
            ids,
            sign_threshold=threshold,
            target_epsilon=target_epsilon,
            lcb_z=lcb_z,
        )
        if (
            int(metrics["selected_count"]) >= int(min_selected_actions)
            and float(metrics["action_image_rate"]) <= float(max_action_image_rate)
            and float(metrics["positive_precision_lift"])
            >= float(min_positive_precision_lift)
            and float(metrics["mean_delta_u_lcb"]) > float(min_mean_delta_u_lcb)
        ):
            rank_key = (
                float(metrics["mean_delta_u_lcb"]),
                float(metrics["mean_delta_u_per_image"]),
                float(metrics["selected_positive_precision"]),
                float(threshold),
            )
            feasible.append((rank_key, float(threshold), metrics))
    if feasible:
        _, threshold, metrics = max(feasible, key=lambda value: value[0])
        return {
            "sign_threshold": threshold,
            "used_identity_fallback": False,
            "candidates_evaluated": len(thresholds),
            "metrics": metrics,
        }
    threshold = float("inf")
    return {
        "sign_threshold": threshold,
        "used_identity_fallback": True,
        "candidates_evaluated": len(thresholds),
        "metrics": ranked_abstention_metrics(
            rank,
            sign,
            utility,
            ids,
            sign_threshold=threshold,
            target_epsilon=target_epsilon,
            lcb_z=lcb_z,
        ),
    }


def evaluate_decomposed_actionability_gates(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    heldout = payload["heldout"]
    full = heldout["arms"]["local_full"]
    selection = payload["selection"]["local_full"]
    sign_keys = ("auroc_equal_image", "auroc_candidate_weighted")
    rank_keys = ("pairwise_accuracy", "pairwise_accuracy_candidate_weighted")
    final_sign = [float(full["sign"][key]) for key in sign_keys]
    final_rank = [float(full["rank"][key]) for key in rank_keys]
    fit_values = [
        *[float(selection["fit_sign"][key]) for key in sign_keys],
        *[float(selection["fit_rank"][key]) for key in rank_keys],
    ]
    tune_values = [
        *[float(selection["tune_sign"][key]) for key in sign_keys],
        *[float(selection["tune_rank"][key]) for key in rank_keys],
    ]
    final_values = [*final_sign, *final_rank]
    controls = ("within_image_feature_shuffle", "within_image_label_shuffle")
    control_gains = [
        final_sign[index]
        - float(heldout["arms"][control]["sign"][key])
        for control in controls
        for index, key in enumerate(sign_keys)
    ] + [
        final_rank[index]
        - float(heldout["arms"][control]["rank"][key])
        for control in controls
        for index, key in enumerate(rank_keys)
    ]
    safety = full["abstention"]
    gate_values = {
        "G0_heldout_support": (
            int(heldout["support"]["candidate_count"])
            >= int(gates["min_heldout_candidates"])
            and int(heldout["support"]["image_count"])
            >= int(gates["min_heldout_images"])
        ),
        "G1_sign_actionability": min(final_sign) >= float(gates["min_sign_auroc"]),
        "G2_within_image_rank": min(final_rank) >= float(gates["min_rank_pairwise"]),
        "G3_frozen_abstention_safety": (
            selection["used_identity_fallback"] is False
            and int(safety["selected_count"]) >= int(gates["min_selected_actions"])
            and float(safety["mean_delta_u_per_image"])
            >= float(gates["min_mean_delta_u"])
            and float(safety["mean_delta_u_lcb"])
            > float(gates["min_mean_delta_u_lcb"])
            and float(safety["positive_precision_lift"])
            >= float(gates["min_positive_precision_lift"])
            and float(safety["action_image_rate"])
            <= float(gates["max_action_image_rate"])
        ),
        "G4_generalization_gap": (
            max(fit - final for fit, final in zip(fit_values, final_values))
            <= float(gates["max_fit_to_heldout_gap"])
            and max(tune - final for tune, final in zip(tune_values, final_values))
            <= float(gates["max_tune_to_heldout_gap"])
        ),
        "G5_control_gain": min(control_gains)
        >= float(gates["min_gain_over_controls"]),
    }
    return {
        "all_passed": all(gate_values.values()),
        "gates": gate_values,
        "diagnostics": {
            "minimum_sign_or_rank_control_gain": min(control_gains),
            "maximum_fit_to_heldout_gap": max(
                fit - final for fit, final in zip(fit_values, final_values)
            ),
            "maximum_tune_to_heldout_gap": max(
                tune - final for tune, final in zip(tune_values, final_values)
            ),
        },
    }


def _binary_auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    positive_scores = scores[positive]
    negative_scores = scores[~positive]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return 0.5
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(
        (comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean().item()
    )


def _rows(
    features: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(features)
    y = torch.as_tensor(target, dtype=x.dtype, device=x.device).flatten()
    ids = torch.as_tensor(image_ids, dtype=torch.long, device=x.device).flatten()
    if x.ndim != 2 or x.shape[0] == 0 or y.shape != ids.shape or x.shape[0] != y.numel():
        raise ValueError("features, target, and image ids must share non-empty rows")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("features and target must be finite")
    return x, y, ids


def _utility_scores(
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
