"""Direct listwise candidate-or-no-op learning for oracle-pool diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ListwiseBatch:
    features: torch.Tensor
    candidate_mask: torch.Tensor
    target_indices: torch.Tensor
    image_ids: torch.Tensor


@dataclass(frozen=True)
class LinearListwiseModel:
    weights: torch.Tensor
    noop_logit: torch.Tensor
    l2: float


def build_listwise_batch(
    features: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
) -> ListwiseBatch:
    x, utility, ids = _feature_rows(features, target, image_ids)
    if target_epsilon < 0.0:
        raise ValueError("target_epsilon must be non-negative")
    unique_images = torch.unique(ids, sorted=True)
    row_groups = [torch.nonzero(ids == image_id, as_tuple=False).flatten() for image_id in unique_images]
    max_candidates = max(int(rows.numel()) for rows in row_groups)
    padded = torch.zeros(
        unique_images.numel(), max_candidates, x.shape[1], dtype=x.dtype, device=x.device
    )
    mask = torch.zeros(
        unique_images.numel(), max_candidates, dtype=torch.bool, device=x.device
    )
    targets = torch.empty(unique_images.numel(), dtype=torch.long, device=x.device)
    for index, rows in enumerate(row_groups):
        count = int(rows.numel())
        padded[index, :count] = x[rows]
        mask[index, :count] = True
        image_target = utility[rows]
        best = int(image_target.argmax().item())
        targets[index] = best if float(image_target[best].item()) > float(target_epsilon) else max_candidates
    return ListwiseBatch(
        features=padded,
        candidate_mask=mask,
        target_indices=targets,
        image_ids=unique_images,
    )


def fit_listwise_model(
    features: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
    l2: float,
    max_iter: int,
) -> LinearListwiseModel:
    if l2 < 0.0 or max_iter <= 0:
        raise ValueError("l2 and max_iter must be valid")
    batch = build_listwise_batch(
        features, target, image_ids, target_epsilon=target_epsilon
    )
    solve_dtype = torch.float64
    padded = batch.features.to(solve_dtype)
    weights = torch.zeros(
        padded.shape[2], dtype=solve_dtype, device=padded.device, requires_grad=True
    )
    noop_logit = torch.zeros((), dtype=solve_dtype, device=padded.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, noop_logit],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        candidate_logits = padded @ weights
        candidate_logits = candidate_logits.masked_fill(~batch.candidate_mask, -1e9)
        logits = torch.cat(
            (
                candidate_logits,
                noop_logit.expand(candidate_logits.shape[0], 1),
            ),
            dim=1,
        )
        loss = F.cross_entropy(logits, batch.target_indices)
        loss = loss + 0.5 * float(l2) * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return LinearListwiseModel(
        weights=weights.detach().to(features.dtype),
        noop_logit=noop_logit.detach().to(features.dtype),
        l2=float(l2),
    )


def predict_listwise_scores(
    model: LinearListwiseModel, features: torch.Tensor
) -> torch.Tensor:
    x = torch.as_tensor(
        features, dtype=model.weights.dtype, device=model.weights.device
    )
    if x.ndim != 2 or x.shape[1] != model.weights.numel():
        raise ValueError("listwise feature dimension mismatch")
    return x @ model.weights


def listwise_decision_metrics(
    scores: torch.Tensor,
    *,
    noop_logit: float | torch.Tensor,
    noop_margin: float,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    target_epsilon: float,
    lcb_z: float,
) -> dict[str, Any]:
    prediction, utility, ids = _score_rows(scores, target, image_ids)
    if min(target_epsilon, lcb_z) < 0.0:
        raise ValueError("decision thresholds must be non-negative")
    noop = float(torch.as_tensor(noop_logit).item()) + float(noop_margin)
    per_image: list[float] = []
    regrets: list[float] = []
    selected_positive = selected_count = correct_decision = 0
    noop_target_count = noop_prediction_count = correct_noop = 0
    for image_id in torch.unique(ids, sorted=True):
        mask = ids == image_id
        image_scores, image_target = prediction[mask], utility[mask]
        predicted_index = int(image_scores.argmax().item())
        acts = float(image_scores[predicted_index].item()) > noop
        selected_delta = float(image_target[predicted_index].item()) if acts else 0.0
        oracle_index = int(image_target.argmax().item())
        oracle_acts = float(image_target[oracle_index].item()) > float(target_epsilon)
        selected_count += int(acts)
        selected_positive += int(acts and selected_delta > float(target_epsilon))
        correct_decision += int(
            (acts and oracle_acts and predicted_index == oracle_index)
            or (not acts and not oracle_acts)
        )
        noop_target_count += int(not oracle_acts)
        noop_prediction_count += int(not acts)
        correct_noop += int(not acts and not oracle_acts)
        per_image.append(selected_delta)
        regrets.append(max(0.0, float(image_target[oracle_index].item())) - selected_delta)
    values = torch.tensor(per_image, dtype=torch.float64)
    mean_delta = float(values.mean().item())
    standard_error = (
        float(values.std(unbiased=True).item()) / math.sqrt(values.numel())
        if values.numel() > 1
        else 0.0
    )
    candidate_prevalence = float(utility.gt(float(target_epsilon)).float().mean().item())
    precision = selected_positive / max(1, selected_count)
    image_count = int(values.numel())
    return {
        "candidate_count": int(utility.numel()),
        "image_count": image_count,
        "noop_margin": float(noop_margin),
        "selected_count": selected_count,
        "selected_positive_count": selected_positive,
        "selected_positive_precision": precision,
        "candidate_positive_prevalence": candidate_prevalence,
        "positive_precision_lift": precision - candidate_prevalence,
        "action_image_rate": selected_count / image_count,
        "decision_accuracy": correct_decision / image_count,
        "noop_target_count": noop_target_count,
        "noop_prediction_count": noop_prediction_count,
        "noop_recall": correct_noop / max(1, noop_target_count),
        "mean_delta_u": mean_delta,
        "mean_delta_u_standard_error": standard_error,
        "mean_delta_u_lcb": mean_delta - float(lcb_z) * standard_error,
        "oracle_regret_mean": sum(regrets) / image_count,
        "per_image_delta_u": per_image,
    }


def calibrate_noop_margin(
    scores: torch.Tensor,
    *,
    noop_logit: float | torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    target_epsilon: float,
    min_selected_actions: int,
    max_action_image_rate: float,
    min_positive_precision_lift: float,
    min_mean_delta_u_lcb: float,
    lcb_z: float,
) -> dict[str, Any]:
    prediction, utility, ids = _score_rows(scores, target, image_ids)
    if min_selected_actions <= 0 or not 0.0 <= max_action_image_rate <= 1.0:
        raise ValueError("invalid no-op calibration constraints")
    noop = float(torch.as_tensor(noop_logit).item())
    advantages = []
    for image_id in torch.unique(ids, sorted=True):
        advantages.append(float(prediction[ids == image_id].max().item()) - noop)
    unique_advantages = sorted(set(advantages))
    epsilon = max(1e-7, float(torch.finfo(torch.float32).eps))
    margins = [unique_advantages[0] - epsilon, *unique_advantages]
    feasible: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for margin in margins:
        metrics = listwise_decision_metrics(
            prediction,
            noop_logit=noop,
            noop_margin=margin,
            target=utility,
            image_ids=ids,
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
            rank = (
                float(metrics["mean_delta_u_lcb"]),
                float(metrics["mean_delta_u"]),
                float(metrics["selected_positive_precision"]),
                float(margin),
            )
            feasible.append((rank, float(margin), metrics))
    if feasible:
        _, margin, metrics = max(feasible, key=lambda item: item[0])
        return {
            "noop_margin": margin,
            "used_identity_fallback": False,
            "candidates_evaluated": len(margins),
            "metrics": metrics,
        }
    margin = float("inf")
    return {
        "noop_margin": margin,
        "used_identity_fallback": True,
        "candidates_evaluated": len(margins),
        "metrics": listwise_decision_metrics(
            prediction,
            noop_logit=noop,
            noop_margin=margin,
            target=utility,
            image_ids=ids,
            target_epsilon=target_epsilon,
            lcb_z=lcb_z,
        ),
    }


def paired_gain_stats(
    left: list[float], right: list[float], *, lcb_z: float
) -> dict[str, float | int]:
    if len(left) != len(right) or not left:
        raise ValueError("paired values must have equal non-empty length")
    differences = torch.tensor(left, dtype=torch.float64) - torch.tensor(
        right, dtype=torch.float64
    )
    mean = float(differences.mean().item())
    standard_error = (
        float(differences.std(unbiased=True).item()) / math.sqrt(differences.numel())
        if differences.numel() > 1
        else 0.0
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lcb": mean - float(lcb_z) * standard_error,
        "pair_count": int(differences.numel()),
    }


def evaluate_listwise_gates(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    support = payload["support"]
    full = payload["arms"]["local_full"]
    selection = payload["selection"]["local_full"]
    gains = payload["paired_gains"]
    gate_values = {
        "G0_support": (
            int(support["candidate_count"]) >= int(gates["min_candidates"])
            and int(support["image_count"]) >= int(gates["min_images"])
        ),
        "G1_safe_listwise_decision": (
            selection["used_identity_fallback"] is False
            and int(full["selected_count"]) >= int(gates["min_selected_actions"])
            and float(full["action_image_rate"])
            <= float(gates["max_action_image_rate"])
            and float(full["positive_precision_lift"])
            >= float(gates["min_positive_precision_lift"])
            and float(full["mean_delta_u_lcb"])
            > float(gates["min_mean_delta_u_lcb"])
        ),
        "G2_baseline_gain": float(gains["vs_rate_matched_random"]["lcb"])
        > float(gates["min_paired_gain_lcb"]),
        "G3_control_gain": (
            float(gains["vs_feature_shuffle"]["lcb"])
            > float(gates["min_paired_gain_lcb"])
            and float(gains["vs_label_shuffle"]["lcb"])
            > float(gates["min_paired_gain_lcb"])
        ),
        "G4_generalization_gap": (
            float(selection["fit_metrics"]["mean_delta_u"])
            - float(full["mean_delta_u"])
            <= float(gates["max_fit_to_outer_mean_gap"])
            and float(selection["tune_metrics"]["mean_delta_u"])
            - float(full["mean_delta_u"])
            <= float(gates["max_tune_to_outer_mean_gap"])
        ),
    }
    return {
        "all_passed": all(gate_values.values()),
        "gates": gate_values,
        "diagnostics": {
            "fit_to_outer_mean_gap": float(selection["fit_metrics"]["mean_delta_u"])
            - float(full["mean_delta_u"]),
            "tune_to_outer_mean_gap": float(selection["tune_metrics"]["mean_delta_u"])
            - float(full["mean_delta_u"]),
            "minimum_paired_gain_lcb": min(float(value["lcb"]) for value in gains.values()),
        },
    }


def _feature_rows(
    features: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(features)
    utility = torch.as_tensor(target, dtype=x.dtype, device=x.device).flatten()
    ids = torch.as_tensor(image_ids, dtype=torch.long, device=x.device).flatten()
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[0] != utility.numel() or utility.shape != ids.shape:
        raise ValueError("features, target, and image ids must share non-empty rows")
    if not torch.isfinite(x).all() or not torch.isfinite(utility).all():
        raise ValueError("listwise rows must be finite")
    return x, utility, ids


def _score_rows(
    scores: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.as_tensor(scores).flatten()
    utility = torch.as_tensor(target, dtype=prediction.dtype, device=prediction.device).flatten()
    ids = torch.as_tensor(image_ids, dtype=torch.long, device=prediction.device).flatten()
    if prediction.numel() == 0 or prediction.shape != utility.shape or utility.shape != ids.shape:
        raise ValueError("scores, target, and image ids must share non-empty rows")
    if not torch.isfinite(prediction).all() or not torch.isfinite(utility).all():
        raise ValueError("decision rows must be finite")
    return prediction, utility, ids
