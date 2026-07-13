"""Calibration and control statistics for held-out joint Delta-U probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from spectral_detection_posttrain.methods.energy_transport.policy.joint_delta_u import (
    ProposalSetEdges,
)


@dataclass(frozen=True)
class ConservativeCalibration:
    """A fit-only action threshold and the evidence used to choose it."""

    threshold: float
    metrics: dict[str, float | int]
    used_identity_fallback: bool
    candidates_evaluated: int


def shuffle_edge_topology(
    edges: ProposalSetEdges,
    labels: torch.Tensor,
    image_indices: torch.Tensor,
    *,
    seed: int,
) -> ProposalSetEdges:
    """Break edge alignment while preserving image/class groups and edge rows."""
    if labels.shape != (edges.node_count,) or image_indices.shape != (edges.node_count,):
        raise ValueError("labels and image_indices must match edges.node_count")
    if edges.edge_index.ndim != 2 or edges.edge_index.shape[0] != 2:
        raise ValueError("edges.edge_index must have shape (2, E)")
    if edges.edge_features.shape != (edges.edge_index.shape[1], 8):
        raise ValueError("edges.edge_features must have shape (E, 8)")
    if edges.edge_index.numel() == 0:
        return edges

    source = edges.edge_index[0].long()
    target = edges.edge_index[1].long()
    if source.min() < 0 or target.min() < 0:
        raise ValueError("edge indices must be non-negative")
    if source.max() >= edges.node_count or target.max() >= edges.node_count:
        raise ValueError("edge indices exceed edges.node_count")
    if not torch.equal(image_indices[source], image_indices[target]):
        raise ValueError("topology shuffle cannot accept cross-image edges")
    if not torch.equal(labels[source], labels[target]):
        raise ValueError("topology shuffle requires same-class input edges")

    shuffled_target = target.clone()
    target_images = image_indices[target]
    target_labels = labels[target]
    groups = torch.unique(
        torch.stack((target_images.long(), target_labels.long()), dim=1), dim=0
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for image_id, label in groups.cpu().tolist():
        rows = torch.nonzero(
            target_images.eq(int(image_id)) & target_labels.eq(int(label)),
            as_tuple=False,
        ).flatten()
        if rows.numel() < 3:
            continue
        original = target[rows]
        replacement = None
        for _ in range(32):
            order = torch.randperm(rows.numel(), generator=generator, device="cpu").to(
                rows.device
            )
            candidate = original[order]
            if not torch.equal(candidate, original) and not candidate.eq(source[rows]).any():
                replacement = candidate
                break
        if replacement is None:
            for shift in range(1, rows.numel()):
                candidate = torch.roll(original, shifts=shift)
                if not candidate.eq(source[rows]).any():
                    replacement = candidate
                    break
        if replacement is not None:
            shuffled_target[rows] = replacement

    return ProposalSetEdges(
        edge_index=torch.stack((source, shuffled_target)),
        edge_features=edges.edge_features,
        node_count=edges.node_count,
    )


def imagewise_pairwise_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
) -> dict[int, float]:
    predicted, target, image_ids = _utility_rows(predicted, target, image_ids)
    if target_epsilon < 0.0:
        raise ValueError("target_epsilon must be non-negative")
    result: dict[int, float] = {}
    for image_id in torch.unique(image_ids, sorted=True).tolist():
        mask = image_ids.eq(int(image_id))
        image_prediction = predicted[mask]
        image_target = target[mask]
        correct = 0.0
        count = 0
        for left in range(image_target.numel()):
            for right in range(left + 1, image_target.numel()):
                target_gap = float((image_target[left] - image_target[right]).item())
                if abs(target_gap) <= float(target_epsilon):
                    continue
                predicted_gap = float(
                    (image_prediction[left] - image_prediction[right]).item()
                )
                count += 1
                if predicted_gap == 0.0:
                    correct += 0.5
                elif (predicted_gap > 0.0) == (target_gap > 0.0):
                    correct += 1.0
        if count:
            result[int(image_id)] = correct / count
    return result


def paired_bootstrap_mean_difference(
    left: Mapping[int, float],
    right: Mapping[int, float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap inputs must share non-empty image IDs")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    image_ids = sorted(left)
    differences = torch.tensor(
        [float(left[image_id]) - float(right[image_id]) for image_id in image_ids],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0,
        differences.numel(),
        (int(resamples), differences.numel()),
        generator=generator,
    )
    samples = differences[indices].mean(dim=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "estimate": float(differences.mean().item()),
        "ci_low": float(torch.quantile(samples, alpha).item()),
        "ci_high": float(torch.quantile(samples, 1.0 - alpha).item()),
        "n_images": int(differences.numel()),
        "resamples": int(resamples),
        "confidence": float(confidence),
    }


def calibrated_selection_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    threshold: float,
    target_epsilon: float,
    lcb_z: float,
) -> dict[str, float | int]:
    predicted, target, image_ids = _utility_rows(predicted, target, image_ids)
    if target_epsilon < 0.0 or lcb_z < 0.0:
        raise ValueError("target_epsilon and lcb_z must be non-negative")
    unique_images = torch.unique(image_ids, sorted=True)
    selected_deltas: list[float] = []
    per_image_deltas: list[float] = []
    regrets: list[float] = []
    selected_positive = 0
    for image_id in unique_images.tolist():
        mask = image_ids.eq(int(image_id))
        image_prediction = predicted[mask]
        image_target = target[mask]
        best = int(image_prediction.argmax().item())
        acts = float(image_prediction[best].item()) > float(threshold)
        selected_delta = float(image_target[best].item()) if acts else 0.0
        if acts:
            selected_deltas.append(selected_delta)
            selected_positive += int(selected_delta > float(target_epsilon))
        per_image_deltas.append(selected_delta)
        regrets.append(max(0.0, float(image_target.max().item())) - selected_delta)

    image_values = torch.tensor(per_image_deltas, dtype=torch.float64)
    mean_delta = float(image_values.mean().item())
    standard_error = (
        float(image_values.std(unbiased=True).item()) / math.sqrt(image_values.numel())
        if image_values.numel() > 1
        else 0.0
    )
    selected_count = len(selected_deltas)
    positive_count = int(target.gt(float(target_epsilon)).sum().item())
    positive_prevalence = positive_count / max(1, target.numel())
    selected_precision = selected_positive / max(1, selected_count)
    return {
        "threshold": float(threshold),
        "candidate_count": int(target.numel()),
        "image_count": int(unique_images.numel()),
        "selected_count": selected_count,
        "selected_positive_count": selected_positive,
        "selected_positive_precision": selected_precision,
        "candidate_positive_prevalence": positive_prevalence,
        "positive_precision_lift": selected_precision - positive_prevalence,
        "action_image_rate": selected_count / max(1, unique_images.numel()),
        "mean_delta_u_per_image": mean_delta,
        "mean_delta_u_selected": sum(selected_deltas) / max(1, selected_count),
        "mean_delta_u_standard_error": standard_error,
        "mean_delta_u_lcb": mean_delta - float(lcb_z) * standard_error,
        "oracle_regret_mean": sum(regrets) / max(1, len(regrets)),
    }


def calibrate_conservative_threshold(
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
    min_selected_actions: int,
    max_action_image_rate: float,
    min_positive_precision_lift: float,
    min_mean_delta_u_lcb: float,
    lcb_z: float,
) -> ConservativeCalibration:
    predicted, target, image_ids = _utility_rows(predicted, target, image_ids)
    if min_selected_actions <= 0:
        raise ValueError("min_selected_actions must be positive")
    if not 0.0 <= max_action_image_rate <= 1.0:
        raise ValueError("max_action_image_rate must be in [0, 1]")
    if min_positive_precision_lift < 0.0:
        raise ValueError("min_positive_precision_lift must be non-negative")

    image_maxima = []
    for image_id in torch.unique(image_ids, sorted=True).tolist():
        image_maxima.append(float(predicted[image_ids.eq(int(image_id))].max().item()))
    unique_maxima = sorted(set(image_maxima))
    epsilon = max(1e-7, torch.finfo(torch.float32).eps)
    thresholds = [unique_maxima[0] - epsilon, *unique_maxima]
    feasible: list[
        tuple[tuple[float, float, float, float], float, dict[str, float | int]]
    ] = []
    for threshold in thresholds:
        metrics = calibrated_selection_metrics(
            predicted,
            target,
            image_ids,
            threshold=threshold,
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
                float(metrics["mean_delta_u_per_image"]),
                float(metrics["selected_positive_precision"]),
                float(threshold),
            )
            feasible.append((rank, float(threshold), metrics))
    if feasible:
        _, threshold, metrics = max(feasible, key=lambda item: item[0])
        return ConservativeCalibration(
            threshold=threshold,
            metrics=metrics,
            used_identity_fallback=False,
            candidates_evaluated=len(thresholds),
        )

    identity_metrics = calibrated_selection_metrics(
        predicted,
        target,
        image_ids,
        threshold=float("inf"),
        target_epsilon=target_epsilon,
        lcb_z=lcb_z,
    )
    return ConservativeCalibration(
        threshold=float("inf"),
        metrics=identity_metrics,
        used_identity_fallback=True,
        candidates_evaluated=len(thresholds),
    )


def constant_utility_baselines(
    target: torch.Tensor,
    image_ids: torch.Tensor,
    *,
    target_epsilon: float,
) -> dict[str, float | int]:
    target = torch.as_tensor(target, dtype=torch.float32).flatten().cpu()
    image_ids = torch.as_tensor(image_ids, dtype=torch.long).flatten().cpu()
    if target.shape != image_ids.shape or target.numel() == 0:
        raise ValueError("target and image_ids must share a non-empty shape")
    positive = target.gt(float(target_epsilon))
    zero = torch.zeros_like(target)
    pairwise = imagewise_pairwise_accuracy(
        zero, target, image_ids, target_epsilon=target_epsilon
    )
    return {
        "candidate_count": int(target.numel()),
        "image_count": int(torch.unique(image_ids).numel()),
        "positive_prevalence": float(positive.float().mean().item()),
        "always_negative_sign_accuracy": float((~positive).float().mean().item()),
        "zero_utility_mae": float(target.abs().mean().item()),
        "zero_utility_pairwise_accuracy": sum(pairwise.values()) / max(1, len(pairwise)),
    }


def _utility_rows(
    predicted: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = torch.as_tensor(predicted, dtype=torch.float32).flatten().cpu()
    target = torch.as_tensor(target, dtype=torch.float32).flatten().cpu()
    image_ids = torch.as_tensor(image_ids, dtype=torch.long).flatten().cpu()
    if not (predicted.shape == target.shape == image_ids.shape) or predicted.numel() == 0:
        raise ValueError("predicted, target, and image_ids must share a non-empty shape")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("utility rows must be finite")
    return predicted, target, image_ids
