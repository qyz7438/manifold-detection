"""Deterministic set search primitives for postprocessed detector actions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Callable, Sequence

from spectral_detection_posttrain.core.matching import match_predictions_to_gt


@dataclass(frozen=True, order=True)
class ActionCandidate:
    """Immutable identity and energy cost for one postprocessing action."""

    action_id: str
    action_energy: float = 0.0

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if float(self.action_energy) < 0.0:
            raise ValueError("action_energy must be non-negative")


@dataclass(frozen=True)
class SetOutcome:
    """Detection counts and action accounting for one selected action set."""

    tp50: int = 0
    tp75: int = 0
    fp50: int = 0
    fp75: int = 0
    prediction_count: int = 0
    action_energy: float = 0.0
    action_count: int = 0

    @property
    def utility(self) -> float:
        return (
            float(self.tp75)
            - 0.25 * float(self.fp75)
            - 0.10 * float(self.fp50)
            - 0.05 * float(self.action_energy)
            - 0.02 * float(self.action_count)
        )


@dataclass(frozen=True)
class SetSearchResult:
    """A canonical action set with its callback-evaluated outcome."""

    actions: tuple[ActionCandidate, ...]
    outcome: SetOutcome

    @property
    def utility(self) -> float:
        return self.outcome.utility


@dataclass(frozen=True)
class PairedBootstrapSummary:
    """Deterministic bootstrap summary for beam-minus-local utilities."""

    n_pairs: int
    mean_delta: float
    ci_low: float
    ci_high: float
    positive_fraction: float
    sign: int


SetEvaluator = Callable[[tuple[ActionCandidate, ...]], SetOutcome]


def set_outcome_from_prediction(
    prediction: dict,
    target: dict,
    score_threshold: float = 0.05,
) -> SetOutcome:
    """Score native predictions with the canonical class-aware matcher."""
    matched50 = match_predictions_to_gt(
        prediction,
        target,
        iou_threshold=0.50,
        score_threshold=float(score_threshold),
    )
    matched75 = match_predictions_to_gt(
        prediction,
        target,
        iou_threshold=0.75,
        score_threshold=float(score_threshold),
    )
    return SetOutcome(
        tp50=len(matched50["matches"]),
        tp75=len(matched75["matches"]),
        fp50=len(matched50["unmatched_predictions"]),
        fp75=len(matched75["unmatched_predictions"]),
        prediction_count=len(matched50["matches"]) + len(matched50["unmatched_predictions"]),
    )


def local_top_b(
    candidates: Sequence[ActionCandidate],
    evaluate: SetEvaluator,
    *,
    top_b: int,
    max_action_energy: float | None = None,
) -> tuple[SetSearchResult, ...]:
    """Return the best singleton actions, ordered deterministically by utility."""
    if top_b < 0:
        raise ValueError("top_b must be non-negative")
    ordered = _canonical_candidates(candidates)
    _validate_energy_budget(max_action_energy)
    results = [
        _evaluate((candidate,), evaluate)
        for candidate in ordered
        if _within_energy((candidate,), max_action_energy)
    ]
    return tuple(sorted(results, key=_result_order)[:top_b])


def greedy_positive_marginal_selection(
    candidates: Sequence[ActionCandidate],
    evaluate: SetEvaluator,
    *,
    max_actions: int | None = None,
    max_action_energy: float | None = None,
) -> SetSearchResult:
    """Select actions while the best feasible marginal utility is positive."""
    _validate_budgets(max_actions, max_action_energy)
    ordered = _canonical_candidates(candidates)
    current = _evaluate((), evaluate)
    while max_actions is None or len(current.actions) < max_actions:
        choices: list[tuple[float, SetSearchResult]] = []
        selected_ids = {action.action_id for action in current.actions}
        for candidate in ordered:
            if candidate.action_id in selected_ids:
                continue
            actions = _canonical_actions((*current.actions, candidate))
            if not _within_energy(actions, max_action_energy):
                continue
            result = _evaluate(actions, evaluate)
            choices.append((result.utility - current.utility, result))
        if not choices:
            break
        marginal, next_result = min(choices, key=lambda item: (-item[0], _action_ids(item[1].actions)))
        if marginal <= 0.0:
            break
        current = next_result
    return current


def beam_search(
    candidates: Sequence[ActionCandidate],
    evaluate: SetEvaluator,
    *,
    beam_width: int,
    max_depth: int,
    max_action_energy: float | None = None,
) -> SetSearchResult:
    """Search coordinated action sets with bounded deterministic beam expansion."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    _validate_energy_budget(max_action_energy)
    ordered = _canonical_candidates(candidates)
    best = _evaluate((), evaluate)
    frontier = (best,)
    for _ in range(max_depth):
        expanded: dict[tuple[str, ...], SetSearchResult] = {}
        for result in frontier:
            selected_ids = {action.action_id for action in result.actions}
            for candidate in ordered:
                if candidate.action_id in selected_ids:
                    continue
                actions = _canonical_actions((*result.actions, candidate))
                if not _within_energy(actions, max_action_energy):
                    continue
                expanded[_action_ids(actions)] = _evaluate(actions, evaluate)
        if not expanded:
            break
        frontier = tuple(sorted(expanded.values(), key=_result_order)[:beam_width])
        best = min((best, *frontier), key=_result_order)
    return best


def deterministic_delta_permutation(deltas: Sequence[float], *, seed: int = 0) -> tuple[float, ...]:
    """Return a seeded permutation without changing the paired delta values."""
    permuted = tuple(float(delta) for delta in deltas)
    indices = list(range(len(permuted)))
    Random(seed).shuffle(indices)
    return tuple(permuted[index] for index in indices)


def paired_bootstrap_summary(
    beam_utilities: Sequence[float],
    local_utilities: Sequence[float],
    *,
    num_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedBootstrapSummary:
    """Summarize paired beam-minus-local utility deltas by bootstrap resampling."""
    if len(beam_utilities) != len(local_utilities) or not beam_utilities:
        raise ValueError("beam_utilities and local_utilities must have equal non-zero length")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    deltas = tuple(float(beam) - float(local) for beam, local in zip(beam_utilities, local_utilities))
    n_pairs = len(deltas)
    rng = Random(seed)
    bootstrap_means = sorted(
        sum(deltas[rng.randrange(n_pairs)] for _ in range(n_pairs)) / n_pairs
        for _ in range(num_resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = bootstrap_means[int(tail * (num_resamples - 1))]
    upper = bootstrap_means[int((1.0 - tail) * (num_resamples - 1))]
    mean_delta = sum(deltas) / n_pairs
    positive_fraction = sum(value > 0.0 for value in bootstrap_means) / num_resamples
    return PairedBootstrapSummary(
        n_pairs=n_pairs,
        mean_delta=mean_delta,
        ci_low=lower,
        ci_high=upper,
        positive_fraction=positive_fraction,
        sign=1 if mean_delta > 0.0 else -1 if mean_delta < 0.0 else 0,
    )


def _canonical_candidates(candidates: Sequence[ActionCandidate]) -> tuple[ActionCandidate, ...]:
    ordered = _canonical_actions(candidates)
    if len({candidate.action_id for candidate in ordered}) != len(ordered):
        raise ValueError("action_id values must be unique")
    return ordered


def _canonical_actions(actions: Sequence[ActionCandidate]) -> tuple[ActionCandidate, ...]:
    return tuple(sorted(actions, key=lambda action: action.action_id))


def _evaluate(actions: tuple[ActionCandidate, ...], evaluate: SetEvaluator) -> SetSearchResult:
    outcome = evaluate(actions)
    if not isinstance(outcome, SetOutcome):
        raise TypeError("evaluate must return SetOutcome")
    return SetSearchResult(
        actions=actions,
        outcome=replace(
            outcome,
            action_energy=sum(float(action.action_energy) for action in actions),
            action_count=len(actions),
        ),
    )


def _within_energy(actions: Sequence[ActionCandidate], max_action_energy: float | None) -> bool:
    return max_action_energy is None or sum(float(action.action_energy) for action in actions) <= max_action_energy


def _validate_budgets(max_actions: int | None, max_action_energy: float | None) -> None:
    if max_actions is not None and max_actions < 0:
        raise ValueError("max_actions must be non-negative")
    _validate_energy_budget(max_action_energy)


def _validate_energy_budget(max_action_energy: float | None) -> None:
    if max_action_energy is not None and max_action_energy < 0.0:
        raise ValueError("max_action_energy must be non-negative")


def _action_ids(actions: Sequence[ActionCandidate]) -> tuple[str, ...]:
    return tuple(action.action_id for action in actions)


def _result_order(result: SetSearchResult) -> tuple[float, tuple[str, ...]]:
    return (-result.utility, _action_ids(result.actions))
