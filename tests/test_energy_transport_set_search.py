from dataclasses import FrozenInstanceError

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport.set_search import (
    ActionCandidate,
    SetOutcome,
    beam_search,
    deterministic_delta_permutation,
    greedy_positive_marginal_selection,
    local_top_b,
    paired_bootstrap_summary,
    set_outcome_from_prediction,
)


def test_action_candidate_is_an_immutable_identity_and_outcome_has_fixed_utility() -> None:
    candidate = ActionCandidate("rescue-7", action_energy=2.0)
    with pytest.raises(FrozenInstanceError):
        candidate.action_id = "other"

    outcome = SetOutcome(tp50=9, tp75=1, fp50=4, fp75=2, prediction_count=7, action_energy=2.0, action_count=3)
    assert outcome.utility == pytest.approx(-0.06)


def test_set_outcome_from_prediction_uses_one_to_one_score_ordered_matching() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1, 1]),
        "scores": torch.tensor([0.9, 0.8, 0.01]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}

    outcome = set_outcome_from_prediction(prediction, target)

    assert outcome == SetOutcome(tp50=1, tp75=1, fp50=1, fp75=1, prediction_count=2)


def test_local_top_actions_can_lose_to_coordinated_beam_search() -> None:
    candidates = (ActionCandidate("a"), ActionCandidate("b"), ActionCandidate("c"))
    outcomes = {
        (): SetOutcome(),
        ("a",): SetOutcome(tp75=1),
        ("b",): SetOutcome(),
        ("c",): SetOutcome(),
        ("a", "b"): SetOutcome(tp75=1),
        ("a", "c"): SetOutcome(tp75=1),
        ("b", "c"): SetOutcome(tp75=3),
    }

    def evaluate(actions: tuple[ActionCandidate, ...]) -> SetOutcome:
        return outcomes[tuple(action.action_id for action in actions)]

    local = local_top_b(candidates, evaluate, top_b=1)
    coordinated = beam_search(candidates, evaluate, beam_width=3, max_depth=2)

    assert local[0].actions == (ActionCandidate("a"),)
    assert coordinated.actions == (ActionCandidate("b"), ActionCandidate("c"))
    assert coordinated.utility > local[0].utility


def test_searches_resolve_ties_by_action_identity() -> None:
    candidates = (ActionCandidate("z"), ActionCandidate("a"), ActionCandidate("m"))

    def evaluate(actions: tuple[ActionCandidate, ...]) -> SetOutcome:
        return SetOutcome(tp75=len(actions))

    assert local_top_b(candidates, evaluate, top_b=2)[0].actions == (ActionCandidate("a"),)
    assert greedy_positive_marginal_selection(candidates, evaluate, max_actions=1).actions == (ActionCandidate("a"),)
    assert beam_search(candidates, evaluate, beam_width=1, max_depth=1).actions == (ActionCandidate("a"),)


def test_search_accounts_for_candidate_energy_and_respects_budget() -> None:
    candidates = (ActionCandidate("cheap", action_energy=1.0), ActionCandidate("costly", action_energy=2.0))

    def evaluate(actions: tuple[ActionCandidate, ...]) -> SetOutcome:
        return SetOutcome(tp75=10 * len(actions), action_energy=99.0, action_count=99)

    result = greedy_positive_marginal_selection(candidates, evaluate, max_actions=2, max_action_energy=1.5)

    assert result.actions == (ActionCandidate("cheap", action_energy=1.0),)
    assert result.outcome.action_energy == pytest.approx(1.0)
    assert result.outcome.action_count == 1


def test_deterministic_delta_permutation_preserves_values() -> None:
    deltas = (3.0, -2.0, 1.0, 0.0, 5.0)

    first = deterministic_delta_permutation(deltas, seed=17)
    second = deterministic_delta_permutation(deltas, seed=17)

    assert first == second
    assert sorted(first) == sorted(deltas)


def test_paired_bootstrap_reports_the_direction_of_beam_local_deltas() -> None:
    positive = paired_bootstrap_summary((2.0, 3.0, 4.0), (1.0, 1.0, 1.0), num_resamples=100, seed=4)
    negative = paired_bootstrap_summary((0.0, 1.0, 2.0), (1.0, 2.0, 3.0), num_resamples=100, seed=4)

    assert positive.mean_delta == pytest.approx(2.0)
    assert positive.sign == 1
    assert positive.positive_fraction == 1.0
    assert positive.ci_low > 0.0
    assert negative.sign == -1
    assert negative.positive_fraction == 0.0
    assert negative.ci_high < 0.0
