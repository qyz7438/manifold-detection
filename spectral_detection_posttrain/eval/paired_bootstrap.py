"""Image-paired bootstrap intervals for detector AP75 differences."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .detection_metrics import evaluate_detection_predictions


@dataclass(frozen=True)
class AP75BootstrapResult:
    """A percentile bootstrap summary for arm A AP75 minus arm B AP75."""

    estimate: float
    ci_low: float
    ci_high: float
    resamples: int
    n_images: int


@dataclass(frozen=True)
class HierarchicalAP75BootstrapResult:
    """A nested seed-and-image bootstrap summary for mean AP75 delta."""

    estimate: float
    ci_low: float
    ci_high: float
    n_seeds: int
    resamples: int


@dataclass(frozen=True)
class _SeedAP75Data:
    predictions_a: Mapping[Hashable, Any]
    predictions_b: Mapping[Hashable, Any]
    targets: Mapping[Hashable, Any]
    image_ids: tuple[Hashable, ...]


def paired_bootstrap_ap75(
    predictions_a: Mapping[Hashable, Any],
    predictions_b: Mapping[Hashable, Any],
    targets: Mapping[Hashable, Any] | None = None,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> AP75BootstrapResult:
    """Bootstrap the global AP75 difference between two detector arms.

    ``predictions_a`` and ``predictions_b`` are maps from image IDs to the
    prediction record for that image. ``targets`` is a common image-ID map.
    For callers that keep each arm together, each arm may instead be a map
    whose values contain ``prediction`` and ``target`` (or a two-item
    ``(prediction, target)`` pair); in that form ``targets`` is omitted.

    Each replicate samples exactly ``n_images`` image IDs with replacement.
    The same sampled positions are used for both arms, and the records are
    passed as duplicated list entries to the canonical global AP evaluator.
    ``estimate`` is the unresampled AP75 difference, ``A - B``.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    pred_a, embedded_targets_a = _unpack_arm(predictions_a, "predictions_a")
    pred_b, embedded_targets_b = _unpack_arm(predictions_b, "predictions_b")

    if targets is None:
        targets = embedded_targets_a or embedded_targets_b
    if targets is None:
        raise ValueError("targets are required unless both arms contain target records")
    if not isinstance(targets, Mapping):
        raise TypeError("targets must be a map keyed by image ID")

    _require_non_empty(pred_a, "predictions_a")
    _require_non_empty(pred_b, "predictions_b")
    _require_non_empty(targets, "targets")

    image_ids = tuple(pred_a.keys())
    expected_ids = set(image_ids)
    if set(pred_b) != expected_ids or set(targets) != expected_ids:
        raise ValueError("predictions_a, predictions_b, and targets must have the same image IDs")

    n_images = len(image_ids)
    estimate = _global_ap75_difference(pred_a, pred_b, targets, image_ids)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bootstrap_estimates: list[float] = []
    for _ in range(n_resamples):
        sampled_positions = torch.randint(
            n_images, (n_images,), generator=generator, device="cpu"
        ).tolist()
        sampled_ids = [image_ids[position] for position in sampled_positions]
        bootstrap_estimates.append(
            _global_ap75_difference(pred_a, pred_b, targets, sampled_ids)
        )

    values = torch.tensor(bootstrap_estimates, dtype=torch.float64)
    tail = (1.0 - confidence) / 2.0
    ci_low = float(torch.quantile(values, tail).item())
    ci_high = float(torch.quantile(values, 1.0 - tail).item())
    return AP75BootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        resamples=n_resamples,
        n_images=n_images,
    )


def hierarchical_paired_bootstrap_ap75(
    predictions_a_by_seed: Mapping[Hashable, Mapping[Hashable, Any]],
    predictions_b_by_seed: Mapping[Hashable, Mapping[Hashable, Any]],
    targets_by_seed: Mapping[Hashable, Mapping[Hashable, Any]] | None = None,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> HierarchicalAP75BootstrapResult:
    """Bootstrap the mean paired AP75 delta across training seeds and images.

    The outer maps are keyed by training seed and each inner map follows the
    same image-record contract as :func:`paired_bootstrap_ap75`. Every
    replicate samples ``n_seeds`` seeds with replacement. For each selected
    seed occurrence, it then samples that seed's images with replacement,
    recomputes global AP75 for both arms, and averages the paired seed deltas.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not isinstance(predictions_a_by_seed, Mapping):
        raise TypeError("predictions_a_by_seed must be a map keyed by training seed")
    if not isinstance(predictions_b_by_seed, Mapping):
        raise TypeError("predictions_b_by_seed must be a map keyed by training seed")

    training_seeds = tuple(predictions_a_by_seed.keys())
    if len(training_seeds) < 2:
        raise ValueError("hierarchical bootstrap requires at least 2 training seeds")
    expected_seeds = set(training_seeds)
    if set(predictions_b_by_seed) != expected_seeds:
        raise ValueError("both arms must have the same training seeds")
    if targets_by_seed is not None:
        if not isinstance(targets_by_seed, Mapping):
            raise TypeError("targets_by_seed must be a map keyed by training seed")
        if set(targets_by_seed) != expected_seeds:
            raise ValueError("both arms and targets must have the same training seeds")

    seed_data = tuple(
        _prepare_seed_data(
            training_seed,
            predictions_a_by_seed[training_seed],
            predictions_b_by_seed[training_seed],
            None if targets_by_seed is None else targets_by_seed[training_seed],
        )
        for training_seed in training_seeds
    )
    n_seeds = len(seed_data)
    estimate = sum(_seed_ap75_difference(data, data.image_ids) for data in seed_data) / n_seeds

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bootstrap_estimates: list[float] = []
    for _ in range(n_resamples):
        selected_seed_positions = torch.randint(
            n_seeds, (n_seeds,), generator=generator, device="cpu"
        ).tolist()
        seed_deltas: list[float] = []
        for seed_position in selected_seed_positions:
            data = seed_data[seed_position]
            n_images = len(data.image_ids)
            sampled_image_positions = torch.randint(
                n_images, (n_images,), generator=generator, device="cpu"
            ).tolist()
            sampled_image_ids = [data.image_ids[position] for position in sampled_image_positions]
            seed_deltas.append(_seed_ap75_difference(data, sampled_image_ids))
        bootstrap_estimates.append(sum(seed_deltas) / n_seeds)

    ci_low, ci_high = _percentile_interval(bootstrap_estimates, confidence)
    return HierarchicalAP75BootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_seeds=n_seeds,
        resamples=n_resamples,
    )


def _prepare_seed_data(
    training_seed: Hashable,
    predictions_a: Mapping[Hashable, Any],
    predictions_b: Mapping[Hashable, Any],
    targets: Mapping[Hashable, Any] | None,
) -> _SeedAP75Data:
    pred_a, embedded_targets_a = _unpack_arm(
        predictions_a, f"predictions_a_by_seed[{training_seed!r}]"
    )
    pred_b, embedded_targets_b = _unpack_arm(
        predictions_b, f"predictions_b_by_seed[{training_seed!r}]"
    )
    if targets is None:
        targets = embedded_targets_a or embedded_targets_b
    if targets is None:
        raise ValueError(f"targets are required for training seed {training_seed!r}")
    if not isinstance(targets, Mapping):
        raise TypeError(f"targets for training seed {training_seed!r} must be an image-ID map")

    _require_non_empty(pred_a, f"predictions_a for training seed {training_seed!r}")
    _require_non_empty(pred_b, f"predictions_b for training seed {training_seed!r}")
    _require_non_empty(targets, f"targets for training seed {training_seed!r}")
    image_ids = tuple(pred_a.keys())
    expected_ids = set(image_ids)
    if set(pred_b) != expected_ids or set(targets) != expected_ids:
        raise ValueError(
            "predictions_a, predictions_b, and targets must have the same image IDs "
            f"for training seed {training_seed!r}"
        )
    return _SeedAP75Data(pred_a, pred_b, targets, image_ids)


def _seed_ap75_difference(
    data: _SeedAP75Data, image_ids: tuple[Hashable, ...] | list[Hashable]
) -> float:
    return _global_ap75_difference(
        data.predictions_a, data.predictions_b, data.targets, image_ids
    )


def _global_ap75_difference(
    predictions_a: Mapping[Hashable, Any],
    predictions_b: Mapping[Hashable, Any],
    targets: Mapping[Hashable, Any],
    image_ids: tuple[Hashable, ...] | list[Hashable],
) -> float:
    sampled_targets = [targets[image_id] for image_id in image_ids]
    ap_a = evaluate_detection_predictions(
        [predictions_a[image_id] for image_id in image_ids],
        sampled_targets,
        iou_threshold=0.75,
        score_threshold=0.05,
    )["ap75"]
    ap_b = evaluate_detection_predictions(
        [predictions_b[image_id] for image_id in image_ids],
        sampled_targets,
        iou_threshold=0.75,
        score_threshold=0.05,
    )["ap75"]
    return float(ap_a) - float(ap_b)


def _percentile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    estimates = torch.tensor(values, dtype=torch.float64)
    tail = (1.0 - confidence) / 2.0
    return (
        float(torch.quantile(estimates, tail).item()),
        float(torch.quantile(estimates, 1.0 - tail).item()),
    )


def _unpack_arm(
    arm: Mapping[Hashable, Any], name: str
) -> tuple[Mapping[Hashable, Any], Mapping[Hashable, Any] | None]:
    if not isinstance(arm, Mapping):
        raise TypeError(f"{name} must be a map keyed by image ID")
    if not arm:
        return arm, None

    if set(arm.keys()) == {"predictions", "targets"}:
        prediction_map = arm["predictions"]
        target_map = arm["targets"]
        if not isinstance(prediction_map, Mapping) or not isinstance(target_map, Mapping):
            raise TypeError(f"{name} bundle must contain image-ID maps")
        return prediction_map, target_map

    predictions: dict[Hashable, Any] = {}
    embedded_targets: dict[Hashable, Any] = {}
    has_embedded_targets = False
    for image_id, record in arm.items():
        if isinstance(record, Mapping) and "prediction" in record:
            predictions[image_id] = record["prediction"]
            if "target" in record:
                embedded_targets[image_id] = record["target"]
                has_embedded_targets = True
        elif isinstance(record, (tuple, list)) and len(record) == 2:
            predictions[image_id], embedded_targets[image_id] = record
            has_embedded_targets = True
        else:
            predictions[image_id] = record

    return predictions, embedded_targets if has_embedded_targets else None


def _require_non_empty(values: Mapping[Hashable, Any], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")


__all__ = [
    "AP75BootstrapResult",
    "HierarchicalAP75BootstrapResult",
    "hierarchical_paired_bootstrap_ap75",
    "paired_bootstrap_ap75",
]
