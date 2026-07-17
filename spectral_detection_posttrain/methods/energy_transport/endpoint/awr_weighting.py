"""Pure utilities for oracle-utility-weighted box-head fine-tuning.

The historical AWR label is retained by the experiment protocol, but these
helpers implement only its locked train-only weighted-ERM bookkeeping.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import random
from typing import Any, TypeAlias


ImageId: TypeAlias = Hashable
CandidateId: TypeAlias = Hashable
DEFAULT_IDENTITY_FAMILY = "identity_permutation"
DEFAULT_LAMBDA = 1.0
DEFAULT_WEIGHT_MAX = 20.0


@dataclass(frozen=True)
class OracleUtilities:
    """Candidate and image utilities derived from one fit cache."""

    candidate_utilities: dict[tuple[ImageId, CandidateId], float]
    image_utilities: dict[ImageId, float]
    candidate_counts: dict[ImageId, int]
    positive_candidate_count: int
    positive_image_count: int


@dataclass(frozen=True)
class WeightDiagnostics:
    """Weight-health values required by the locked protocol."""

    raw_min: float
    raw_max: float
    raw_mean: float
    raw_std: float
    normalized_min: float
    normalized_max: float
    normalized_mean: float
    normalized_std: float
    saturation_fraction: float
    effective_sample_size: float


@dataclass(frozen=True)
class OracleUtilityWeights:
    """Raw and globally normalized image weights with their diagnostics."""

    raw_weights: dict[ImageId, float]
    weights: dict[ImageId, float]
    diagnostics: WeightDiagnostics


@dataclass(frozen=True)
class BudgetCounters:
    """Runtime counters which must match across the four training arms."""

    image_exposures: int
    logical_batches: int
    optimizer_steps: int
    scheduler_steps: int


def derive_oracle_utilities(
    fit_image_ids: Sequence[ImageId],
    cache_rows: Iterable[Mapping[str, Any]],
    *,
    identity_family: str = DEFAULT_IDENTITY_FAMILY,
) -> OracleUtilities:
    """Derive no-op-relative candidate and image utilities from cache rows.

    ``cache_rows`` accepts flattened action rows or the cache builder's nested
    ``{"image_id": ..., "actions": [...]}`` records.  Only ``q_teacher`` is
    consulted; identity rows are checked for exact zero and excluded.
    """
    image_ids = tuple(fit_image_ids)
    _validate_unique_ids(image_ids, "fit_image_ids")
    if not isinstance(identity_family, str) or not identity_family:
        raise ValueError("identity_family must be a non-empty string")

    known_images = set(image_ids)
    candidate_values: dict[tuple[ImageId, CandidateId], list[float]] = {}
    seen_rows: set[tuple[ImageId, CandidateId, Hashable]] = set()

    for image_id, row in _iter_action_rows(cache_rows):
        _validate_hashable(image_id, "image_id")
        if image_id not in known_images:
            raise ValueError(f"unknown image_id in cache row: {image_id!r}")
        candidate_id = _row_identifier(row, ("candidate_index", "candidate_id", "candidate"), "candidate")
        action_id = _row_identifier(row, ("family", "action_id", "action"), "action")
        row_key = (image_id, candidate_id, action_id)
        if row_key in seen_rows:
            raise ValueError(f"duplicate ambiguous cache row: {row_key!r}")
        seen_rows.add(row_key)

        q_teacher = _finite_float(row.get("q_teacher"), "q_teacher")
        identity = _is_identity(row, action_id, identity_family)
        if identity:
            if q_teacher != 0.0:
                raise ValueError("identity q_teacher must be exactly zero")
            continue
        candidate_values.setdefault((image_id, candidate_id), []).append(q_teacher)

    candidate_utilities = {
        key: max(values) for key, values in candidate_values.items()
    }
    candidate_counts = {image_id: 0 for image_id in image_ids}
    for image_id, _ in candidate_utilities:
        candidate_counts[image_id] += 1

    image_utilities = {image_id: 0.0 for image_id in image_ids}
    for (image_id, _), utility in candidate_utilities.items():
        image_utilities[image_id] = max(image_utilities[image_id], utility, 0.0)

    return OracleUtilities(
        candidate_utilities=candidate_utilities,
        image_utilities=image_utilities,
        candidate_counts=candidate_counts,
        positive_candidate_count=sum(value > 0.0 for value in candidate_utilities.values()),
        positive_image_count=sum(value > 0.0 for value in image_utilities.values()),
    )


def build_oracle_utility_weights(
    image_utilities: Mapping[ImageId, float],
    *,
    lambda_: float = DEFAULT_LAMBDA,
    weight_max: float = DEFAULT_WEIGHT_MAX,
) -> OracleUtilityWeights:
    """Apply the locked capped exponential transform and global fit mean."""
    if not image_utilities:
        raise ValueError("image_utilities must not be empty")
    lambda_value = _finite_float(lambda_, "lambda")
    cap = _finite_float(weight_max, "weight_max")
    if lambda_value <= 0.0:
        raise ValueError("lambda must be positive")
    if cap <= 0.0:
        raise ValueError("weight_max must be positive")

    cap_log = math.log(cap)
    raw_weights: dict[ImageId, float] = {}
    saturated: list[bool] = []
    for image_id, utility_value in image_utilities.items():
        _validate_hashable(image_id, "image_id")
        utility = _finite_float(utility_value, "image utility")
        if utility < 0.0:
            raise ValueError("image utility must be non-negative")
        exponent = utility / lambda_value
        is_saturated = exponent >= cap_log
        raw_weights[image_id] = cap if is_saturated else math.exp(exponent)
        saturated.append(is_saturated)

    raw_mean = _mean(tuple(raw_weights.values()))
    weights = {image_id: value / raw_mean for image_id, value in raw_weights.items()}
    raw_values = tuple(raw_weights.values())
    normalized_values = tuple(weights.values())
    diagnostics = WeightDiagnostics(
        raw_min=min(raw_values),
        raw_max=max(raw_values),
        raw_mean=raw_mean,
        raw_std=_population_std(raw_values, raw_mean),
        normalized_min=min(normalized_values),
        normalized_max=max(normalized_values),
        normalized_mean=_mean(normalized_values),
        normalized_std=_population_std(normalized_values, _mean(normalized_values)),
        saturation_fraction=sum(saturated) / len(saturated),
        effective_sample_size=_effective_sample_size(normalized_values),
    )
    return OracleUtilityWeights(
        raw_weights=raw_weights,
        weights=weights,
        diagnostics=diagnostics,
    )


def deterministic_weight_permutation(
    weights: Mapping[ImageId, float], *, seed: int
) -> dict[ImageId, float]:
    """Assign a seed-stable permutation of weights to the same image IDs."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    image_ids = tuple(weights)
    _validate_unique_ids(image_ids, "weight image IDs")
    values = [_finite_float(value, "weight") for value in weights.values()]
    random.Random(seed).shuffle(values)
    return dict(zip(image_ids, values, strict=True))


def assert_equal_budget(counters: Mapping[str, BudgetCounters]) -> None:
    """Hard-assert that U/W/F/S consumed identical runtime budgets."""
    required_arms = ("U", "W", "F", "S")
    missing = tuple(arm for arm in required_arms if arm not in counters)
    if missing:
        raise AssertionError(f"missing equal-budget arm counters: {missing!r}")

    baseline = _validated_counters(counters["U"], "U")
    for arm in required_arms[1:]:
        current = _validated_counters(counters[arm], arm)
        for field in ("image_exposures", "logical_batches", "optimizer_steps", "scheduler_steps"):
            if getattr(current, field) != getattr(baseline, field):
                raise AssertionError(
                    f"equal-budget mismatch for {field}: U={getattr(baseline, field)!r}, "
                    f"{arm}={getattr(current, field)!r}"
                )


def _iter_action_rows(
    cache_rows: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[ImageId, Mapping[str, Any]]]:
    for record in cache_rows:
        if not isinstance(record, Mapping):
            raise ValueError("cache rows must be mappings")
        if "actions" not in record:
            yield _required_value(record, "image_id"), record
            continue

        image_id = _required_value(record, "image_id")
        actions = record["actions"]
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Iterable):
            raise ValueError("cache record actions must be an iterable of mappings")
        for action in actions:
            if not isinstance(action, Mapping):
                raise ValueError("cache action rows must be mappings")
            if "image_id" in action and action["image_id"] != image_id:
                raise ValueError("nested action image_id conflicts with cache record")
            yield image_id, action


def _is_identity(
    row: Mapping[str, Any], action_id: Hashable, identity_family: str
) -> bool:
    from_family = action_id == identity_family
    if "identity" not in row:
        return from_family
    marked = row["identity"]
    if not isinstance(marked, bool):
        raise ValueError("identity marker must be boolean")
    if not marked and from_family:
        raise ValueError("ambiguous identity cache row")
    return marked


def _row_identifier(
    row: Mapping[str, Any], fields: tuple[str, ...], name: str
) -> Hashable:
    for field in fields:
        if field in row:
            value = row[field]
            _validate_hashable(value, name)
            return value
    raise ValueError(f"cache row is missing {name}")


def _required_value(row: Mapping[str, Any], field: str) -> Any:
    if field not in row:
        raise ValueError(f"cache row is missing {field}")
    return row[field]


def _validate_unique_ids(values: Sequence[ImageId], name: str) -> None:
    try:
        unique = set(values)
    except TypeError as error:
        raise ValueError(f"{name} must be hashable") from error
    if len(unique) != len(values):
        raise ValueError(f"{name} contains duplicate IDs")


def _validate_hashable(value: Any, name: str) -> None:
    if not isinstance(value, Hashable):
        raise ValueError(f"{name} must be hashable")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _population_std(values: Sequence[float], mean: float) -> float:
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))


def _effective_sample_size(values: Sequence[float]) -> float:
    numerator = math.fsum(values) ** 2
    denominator = math.fsum(value * value for value in values)
    return numerator / denominator


def _validated_counters(value: BudgetCounters, arm: str) -> BudgetCounters:
    if not isinstance(value, BudgetCounters):
        raise AssertionError(f"{arm} budget counters must be BudgetCounters")
    for field in ("image_exposures", "logical_batches", "optimizer_steps", "scheduler_steps"):
        count = getattr(value, field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise AssertionError(f"{arm} {field} must be a non-negative integer")
    return value


__all__ = [
    "BudgetCounters",
    "DEFAULT_IDENTITY_FAMILY",
    "DEFAULT_LAMBDA",
    "DEFAULT_WEIGHT_MAX",
    "OracleUtilities",
    "OracleUtilityWeights",
    "WeightDiagnostics",
    "assert_equal_budget",
    "build_oracle_utility_weights",
    "derive_oracle_utilities",
    "deterministic_weight_permutation",
]
