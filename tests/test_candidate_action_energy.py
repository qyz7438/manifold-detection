from __future__ import annotations

import pytest
import torch

from spectral_detection_posttrain.methods.energy_transport import (
    CandidateEnergyLossConfig,
    CandidateGainLossConfig,
    ContextOnlyCandidateEnergyHead,
    ROIActionState,
    SpatialCandidateEnergyHead,
    build_candidate_quality_targets,
    build_symmetric_box_candidates,
    candidate_action_energy_loss,
    candidate_action_gain_loss,
    select_min_energy_box_actions,
)


def _state() -> ROIActionState:
    return ROIActionState(
        features=torch.randn(3, 5),
        boxes=torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
        scores=torch.tensor([0.9, 0.8, 0.7]),
        labels=torch.tensor([1, 1, 2]),
        image_indices=torch.tensor([0, 0, 1]),
        proposal_indices=torch.tensor([0, 1, 0]),
        logits=torch.tensor([[0.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
        matched_gt_indices=torch.tensor([0, 1, 0]),
    )


def test_candidate_library_is_unique_symmetric_and_starts_with_identity() -> None:
    candidates = build_symmetric_box_candidates((0.1,))

    assert candidates.shape == (25, 4)
    assert candidates[0].count_nonzero().item() == 0
    assert torch.unique(candidates, dim=0).shape[0] == candidates.shape[0]
    rows = {tuple(row.tolist()) for row in candidates}
    for row in rows:
        assert tuple(-value for value in row) in rows


def test_default_candidate_library_has_expected_three_scale_count() -> None:
    candidates = build_symmetric_box_candidates()

    assert candidates.shape == (73, 4)


def test_quality_targets_identify_improving_translation_and_preserve_wrong_class() -> None:
    state = _state()
    candidates = build_symmetric_box_candidates((0.1,))
    targets = build_candidate_quality_targets(
        state,
        candidates,
        matched_gt_boxes=torch.tensor(
            [
                [1.0, 0.0, 11.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
        matched_gt_labels=torch.tensor([1, 2, 2]),
        image_sizes=[(20, 20), (40, 40)],
        min_iou_gain=0.001,
    )

    assert targets.target_indices[0].item() != 0
    best_delta = candidates[targets.target_indices[0]]
    assert best_delta[0].item() == pytest.approx(0.1)
    assert targets.target_indices[1].item() == 0
    assert targets.target_indices[2].item() == 0
    assert targets.class_correct.tolist() == [True, False, True]


def test_candidate_loss_prefers_low_energy_on_oracle_index() -> None:
    qualities = torch.tensor(
        [
            [0.6, 0.8, 0.5],
            [0.9, 0.7, 0.6],
        ]
    )
    targets = torch.tensor([1, 0])
    scores = torch.tensor([0.8, 0.9])
    config = CandidateEnergyLossConfig(temperature=0.1)

    aligned = candidate_action_energy_loss(
        torch.tensor([[1.0, -1.0, 2.0], [-1.0, 1.0, 2.0]]),
        candidate_quality=qualities,
        target_indices=targets,
        scores=scores,
        config=config,
    )
    inverted = candidate_action_energy_loss(
        torch.tensor([[-1.0, 1.0, 2.0], [1.0, -1.0, 2.0]]),
        candidate_quality=qualities,
        target_indices=targets,
        scores=scores,
        config=config,
    )

    assert aligned["loss_total"].item() < inverted["loss_total"].item()


def test_dense_gain_loss_fits_identity_relative_candidate_quality() -> None:
    quality = torch.tensor([[0.5, 0.7, 0.3], [0.8, 0.75, 0.85]])
    scores = torch.tensor([0.9, 0.8])
    target_gain = quality - quality[:, :1]
    aligned_energy = -target_gain

    aligned = candidate_action_gain_loss(
        aligned_energy,
        candidate_quality=quality,
        scores=scores,
        config=CandidateGainLossConfig(energy_weight=0.0),
    )
    inverted = candidate_action_gain_loss(
        target_gain,
        candidate_quality=quality,
        scores=scores,
        config=CandidateGainLossConfig(energy_weight=0.0),
    )
    shifted = candidate_action_gain_loss(
        aligned_energy + 7.0,
        candidate_quality=quality,
        scores=scores,
        config=CandidateGainLossConfig(energy_weight=0.0),
    )

    assert aligned["loss_total"].item() == pytest.approx(0.0, abs=1e-8)
    assert aligned["loss_total"].item() < inverted["loss_total"].item()
    assert shifted["loss_total"].item() == pytest.approx(aligned["loss_total"].item(), abs=1e-8)
    assert aligned["gain_sign_accuracy"].item() == pytest.approx(1.0)


def test_ap75_utility_gain_loss_emphasizes_threshold_crossing() -> None:
    quality = torch.tensor([[0.74, 0.76, 0.70, 0.80]])
    config = CandidateGainLossConfig(
        target_mode="ap75_utility",
        utility_temperature=0.05,
        energy_weight=0.0,
    )
    utility = torch.sigmoid((quality - 0.75) / 0.05)
    target_gain = utility - utility[:, :1]

    aligned = candidate_action_gain_loss(
        -target_gain,
        candidate_quality=quality,
        scores=torch.tensor([0.9]),
        config=config,
    )

    assert target_gain[0, 1].item() > 0.0
    assert target_gain[0, 2].item() < 0.0
    assert aligned["loss_total"].item() == pytest.approx(0.0, abs=1e-8)
    assert aligned["gain_sign_accuracy"].item() == pytest.approx(1.0)


def test_spatial_candidate_head_preserves_batch_shape_and_zero_init() -> None:
    head = SpatialCandidateEnergyHead(
        in_channels=4,
        num_classes=3,
        hidden_dim=8,
        spatial_size=3,
    )
    energies = head(
        torch.randn(2, 4, 3, 3),
        torch.randn(2, 3),
        torch.tensor([1, 2]),
        torch.tensor([0.8, 0.7]),
        torch.randn(2, 4),
    )

    assert energies.shape == (2,)
    assert energies.count_nonzero().item() == 0


def test_context_only_candidate_head_uses_empty_feature_code() -> None:
    head = ContextOnlyCandidateEnergyHead(num_classes=3, hidden_dim=8)
    energies = head(
        torch.empty(2, 0),
        torch.randn(2, 3),
        torch.tensor([1, 2]),
        torch.tensor([0.8, 0.7]),
        torch.randn(2, 4),
    )

    assert energies.shape == (2,)
    assert energies.count_nonzero().item() == 0


def test_spatial_and_box_candidate_heads_have_comparable_capacity() -> None:
    from spectral_detection_posttrain.methods.energy_transport import ActionBenefitEnergyHead

    box_head = ActionBenefitEnergyHead(feature_dim=1024, num_classes=11, hidden_dim=256)
    spatial_head = SpatialCandidateEnergyHead(
        in_channels=256,
        num_classes=11,
        hidden_dim=256,
        spatial_size=7,
    )
    box_parameters = sum(parameter.numel() for parameter in box_head.parameters())
    spatial_parameters = sum(parameter.numel() for parameter in spatial_head.parameters())

    assert spatial_parameters / box_parameters == pytest.approx(1.0372, rel=0.01)


def test_selector_keeps_identity_without_energy_drop_and_respects_topk() -> None:
    state = _state()
    candidates = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0, 0.0],
        ]
    )
    energies = torch.tensor(
        [
            [0.0, -0.3, 0.2],
            [0.0, -0.2, 0.1],
            [0.0, 0.1, 0.2],
        ]
    )

    actions, selected, move = select_min_energy_box_actions(
        state,
        candidates,
        energies,
        min_energy_drop=0.0,
        min_score=0.05,
        require_foreground_dominant=True,
        max_actions_per_image=1,
    )

    assert selected.tolist() == [1, 0, 0]
    assert move.tolist() == [True, False, False]
    assert torch.allclose(actions.box_delta[0], candidates[1])
    assert actions.box_delta[1:].count_nonzero().item() == 0
