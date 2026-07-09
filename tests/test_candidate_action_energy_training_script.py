from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from spectral_detection_posttrain.methods.energy_transport import (
    ActionBenefitEnergyHead,
    ROIActionState,
    SpatialCandidateEnergyHead,
    build_symmetric_box_candidates,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_candidate_action_energy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_candidate_action_energy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state() -> ROIActionState:
    return ROIActionState(
        features=torch.randn(4, 6),
        boxes=torch.zeros((4, 4)),
        scores=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        labels=torch.tensor([1, 1, 2, 2]),
        image_indices=torch.tensor([0, 0, 1, 1]),
        proposal_indices=torch.tensor([0, 1, 0, 1]),
        logits=torch.tensor(
            [
                [0.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    )


def test_step_parser_and_zero_initialized_candidate_energy_shape() -> None:
    module = _load_module()
    state = _state()
    candidates = build_symmetric_box_candidates((0.1,))
    head = ActionBenefitEnergyHead(feature_dim=6, num_classes=3, hidden_dim=8)

    energies = module.candidate_energies(head, state, candidates)

    assert module.parse_step_sizes(".2,.05,.1,.1") == (0.05, 0.1, 0.2)
    assert energies.shape == (4, 25)
    assert energies.count_nonzero().item() == 0


def test_candidate_energy_encode_once_matches_direct_vector_scoring() -> None:
    module = _load_module()
    state = _state()
    candidates = build_symmetric_box_candidates((0.1,))
    head = ActionBenefitEnergyHead(feature_dim=6, num_classes=3, hidden_dim=8)
    torch.nn.init.normal_(head.energy_head[-1].weight)

    encoded_once = module.candidate_energies(head, state, candidates)
    count, candidate_count = encoded_once.shape
    direct = head(
        state.features[:, None, :].expand(count, candidate_count, -1).reshape(-1, 6),
        state.logits[:, None, :].expand(count, candidate_count, -1).reshape(-1, 3),
        state.labels[:, None].expand(count, candidate_count).reshape(-1),
        state.scores[:, None].expand(count, candidate_count).reshape(-1),
        candidates[None, :, :].expand(count, candidate_count, 4).reshape(-1, 4),
    ).reshape(count, candidate_count)

    assert torch.allclose(encoded_once, direct, atol=1e-6)


def test_candidate_energy_encodes_spatial_roi_once_per_proposal() -> None:
    module = _load_module()
    state = _state()
    candidates = build_symmetric_box_candidates((0.1,))
    head = SpatialCandidateEnergyHead(
        in_channels=4,
        num_classes=3,
        hidden_dim=8,
        spatial_size=3,
    )

    energies = module.candidate_energies(
        head,
        state,
        candidates,
        feature_values=torch.randn(4, 4, 3, 3),
    )

    assert energies.shape == (4, 25)
    assert energies.count_nonzero().item() == 0


def test_energy_shuffle_only_reassigns_eligible_rows_within_each_image() -> None:
    module = _load_module()
    energies = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )
    image_indices = torch.tensor([0, 0, 1, 1])
    eligible = torch.tensor([True, True, True, False])

    shuffled = module._permute_energy_rows(
        energies,
        image_indices,
        eligible,
        generator=torch.Generator().manual_seed(5),
    )

    assert sorted(map(tuple, shuffled[:2].tolist())) == [(1.0, 2.0), (3.0, 4.0)]
    assert torch.equal(shuffled[2:], energies[2:])


def test_roi_feature_shuffle_preserves_label_and_score_context_groups() -> None:
    module = _load_module()
    state = _state()
    state.features.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6))

    shuffled, changed = module._permute_roi_features_within_context(
        state,
        torch.ones(4, dtype=torch.bool),
        generator=torch.Generator().manual_seed(7),
        score_bins=5,
    )

    assert changed.tolist() == [True, True, True, True]
    assert torch.equal(shuffled[0], state.features[1])
    assert torch.equal(shuffled[1], state.features[0])
    assert torch.equal(shuffled[2], state.features[3])
    assert torch.equal(shuffled[3], state.features[2])
