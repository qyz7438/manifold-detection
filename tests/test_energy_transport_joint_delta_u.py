"""TDD coverage for action-local joint Delta-U proposal selection."""

from __future__ import annotations

import importlib

import pytest
import torch


def _joint_delta_u_module():
    return importlib.import_module(
        "spectral_detection_posttrain.methods.energy_transport.joint_delta_u"
    )


def _probe_inputs() -> dict[str, torch.Tensor]:
    return {
        "spatial_features": torch.randn(3, 4, 3, 3),
        "class_logits": torch.tensor(
            [[2.0, 0.0, -1.0], [0.5, 1.5, -1.0], [0.0, 1.0, 0.5]]
        ),
        "labels": torch.tensor([1, 1, 1]),
        "scores": torch.tensor([0.9, 0.6, 0.4]),
        "boxes": torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0], [0.0, 0.0, 8.0, 8.0]]
        ),
        "image_indices": torch.tensor([0, 0, 1]),
        "image_sizes": torch.tensor([[20.0, 20.0], [20.0, 20.0]]),
    }


def test_joint_delta_u_public_api_is_exported() -> None:
    package = importlib.import_module(
        "spectral_detection_posttrain.methods.energy_transport"
    )

    assert package.JointDeltaUProbe is _joint_delta_u_module().JointDeltaUProbe
    assert package.build_proposal_set_edges is _joint_delta_u_module().build_proposal_set_edges


def test_proposal_edges_never_cross_images_and_expose_eight_features() -> None:
    api = _joint_delta_u_module()
    values = _probe_inputs()

    edges = api.build_proposal_set_edges(
        values["boxes"],
        values["labels"],
        values["scores"],
        values["image_indices"],
        values["image_sizes"],
        max_neighbors=32,
        same_class_only=True,
    )

    assert edges.node_count == 3
    assert edges.edge_index.shape[0] == 2
    assert edges.edge_index.shape[1] == 2
    assert edges.edge_features.shape == (edges.edge_index.shape[1], 8)
    source, target = edges.edge_index
    assert torch.equal(values["image_indices"][source], values["image_indices"][target])
    assert torch.all(edges.edge_features[:, 7] == 1.0)
    assert torch.all(edges.edge_features[:, 0] > 0.0)
    forward_edge = torch.nonzero((source == 0) & (target == 1), as_tuple=False).item()
    assert torch.allclose(
        edges.edge_features[forward_edge],
        torch.tensor([2.0 / 3.0, 0.1, 0.0, 0.0, 0.0, 0.3, 1.0, 1.0]),
        atol=1e-6,
    )


def test_probe_anchors_identity_candidate_at_exact_zero() -> None:
    api = _joint_delta_u_module()
    values = _probe_inputs()
    edges = api.build_proposal_set_edges(
        values["boxes"],
        values["labels"],
        values["scores"],
        values["image_indices"],
        values["image_sizes"],
    )
    candidate_deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
        ]
    )
    probe = api.JointDeltaUProbe(
        in_channels=4,
        num_classes=3,
        candidate_deltas=candidate_deltas,
        hidden_dim=8,
    )

    delta_u = probe(edges=edges, **values)

    assert delta_u.shape == (3, 4)
    assert torch.equal(delta_u[:, 0], torch.zeros(3, dtype=delta_u.dtype))

    invalid = candidate_deltas.clone()
    invalid[0, 0] = 0.1
    with pytest.raises(ValueError, match="identity"):
        api.JointDeltaUProbe(
            in_channels=4,
            num_classes=3,
            candidate_deltas=invalid,
            hidden_dim=8,
        )


def test_group_heldout_split_is_stable_exclusive_and_covers_rows() -> None:
    api = _joint_delta_u_module()
    group_ids = torch.tensor([10, 10, 3, 3, 8, 8, 5, 5, 7])

    first = api.group_heldout_split(group_ids, heldout_fraction=0.5, seed=123)
    second = api.group_heldout_split(group_ids, heldout_fraction=0.5, seed=123)

    assert torch.equal(first.train_mask, second.train_mask)
    assert torch.equal(first.heldout_mask, second.heldout_mask)
    assert not torch.any(first.train_mask & first.heldout_mask)
    assert torch.all(first.train_mask | first.heldout_mask)
    assert not set(first.train_groups.tolist()) & set(first.heldout_groups.tolist())
    for group in torch.unique(group_ids):
        in_group = group_ids == group
        assert bool(first.train_mask[in_group].all() or first.heldout_mask[in_group].all())


def test_selector_applies_per_image_budget_and_rejects_nonidentity_zero() -> None:
    api = _joint_delta_u_module()
    candidate_deltas = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]
    )
    delta_u = torch.tensor(
        [[0.0, 0.8, 0.1], [0.0, 0.7, 0.2], [0.0, -0.4, 0.2], [0.0, 0.1, 0.3]]
    )
    image_indices = torch.tensor([0, 0, 0, 1])

    selected = api.select_joint_delta_u_actions(
        candidate_deltas,
        delta_u,
        image_indices,
        max_actions_per_image=1,
    )

    assert selected.selected_mask.tolist() == [True, False, False, True]
    assert selected.candidate_indices.tolist() == [1, 0, 0, 2]
    assert torch.equal(selected.box_delta, candidate_deltas[selected.candidate_indices])
    assert torch.allclose(selected.delta_u, torch.tensor([0.8, 0.0, 0.0, 0.3]))

    invalid = candidate_deltas.clone()
    invalid[0, 0] = 1e-6
    with pytest.raises(ValueError, match="identity"):
        api.select_joint_delta_u_actions(
            invalid,
            delta_u,
            image_indices,
            max_actions_per_image=1,
        )


def test_joint_delta_u_loss_backpropagates_through_all_terms() -> None:
    api = _joint_delta_u_module()
    prediction = torch.tensor(
        [[0.0, 0.1, -0.2], [0.0, -0.1, 0.3]], requires_grad=True
    )
    target = torch.tensor([[0.0, 0.4, -0.3], [0.0, -0.2, 0.5]])
    mask = torch.tensor([[True, True, True], [True, True, False]])

    monitor = api.joint_delta_u_loss(
        prediction,
        target,
        mask=mask,
        config=api.JointDeltaULossConfig(pairwise_epsilon=0.05),
    )
    monitor["loss"].backward()

    assert monitor["loss"].ndim == 0
    assert monitor["smooth_l1"].item() > 0.0
    assert monitor["sign_bce"].item() > 0.0
    assert monitor["pairwise_ranking"].item() > 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_joint_delta_u_loss_ranks_sparse_candidates_across_rows_in_same_image() -> None:
    api = _joint_delta_u_module()
    prediction = torch.tensor(
        [[0.0, -0.5, 0.0], [0.0, 0.0, 0.5]], requires_grad=True
    )
    target = torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.0, -0.3]])
    mask = torch.tensor([[False, True, False], [False, False, True]])

    monitor = api.joint_delta_u_loss(
        prediction,
        target,
        mask=mask,
        group_ids=torch.tensor([7, 7]),
        config=api.JointDeltaULossConfig(pairwise_epsilon=0.05),
    )

    assert monitor["pair_count"].item() == 1
    assert monitor["pairwise_ranking"].item() > 1.0


def test_joint_delta_u_metrics_are_perfect_for_perfect_prediction() -> None:
    api = _joint_delta_u_module()
    target = torch.tensor([[0.0, 0.5, -0.2], [0.0, 0.2, 0.1]])

    metrics = api.joint_delta_u_metrics(target, target, epsilon=0.01)

    assert metrics["masked_mae"] == 0.0
    assert metrics["sign_accuracy"] == 1.0
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["oracle_regret"] == 0.0
    assert metrics["positive_precision"] == 1.0
    assert metrics["positive_recall"] == 1.0
    assert metrics["selected_true_delta"] == pytest.approx(0.35)
