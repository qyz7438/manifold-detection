from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport.dense_endpoint import (
    DenseSetEnergyEndpoint,
    RobustTeacherStats,
    build_sparse_pair_features,
    reduced_teacher_values,
)
from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import DenseTeacherComponents


def test_reduced_teacher_uses_natural_support_and_fit_only_stats():
    components = DenseTeacherComponents(
        coverage=torch.tensor(8.0),
        background_risk=torch.tensor(4.0),
        class_risk=torch.tensor(2.0),
        duplicate_risk=torch.tensor(3.0),
        calibration_error=torch.tensor(9.0),
        prediction_count=2,
        ground_truth_count=4,
        duplicate_edge_count=3,
    )
    values = reduced_teacher_values(components)
    stats = RobustTeacherStats.fit(torch.tensor([[1.0, 2.0, 0.0], [2.0, 3.0, 1.0], [3.0, 4.0, 2.0]]))

    assert torch.allclose(values, torch.tensor([2.0, 3.0, 1.0]))
    assert torch.allclose(stats.standardize(values), torch.zeros(3))
    assert stats.quality(values) == 0


def test_sparse_pair_features_are_symmetric_and_permutation_stable():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.5, 2.5], [3.0, 3.0, 4.0, 4.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([1, 1, 2])
    first = build_sparse_pair_features(boxes, scores, labels, (4, 4), min_iou=0.1)
    permutation = torch.tensor([2, 0, 1])
    second = build_sparse_pair_features(
        boxes[permutation], scores[permutation], labels[permutation], (4, 4), min_iou=0.1
    )

    assert first.shape == (1, 5)
    assert torch.allclose(first, second)


def test_dense_endpoint_is_permutation_invariant_and_handles_empty_pairs():
    torch.manual_seed(3)
    endpoint = DenseSetEnergyEndpoint(node_dim=4, pair_dim=5, hidden_dim=8)
    nodes = torch.randn(3, 4)
    pairs = torch.randn(2, 5)
    original = endpoint(nodes, pairs)
    permuted = endpoint(nodes[torch.tensor([2, 0, 1])], pairs[torch.tensor([1, 0])])
    no_pairs = endpoint(nodes, torch.empty(0, 5))
    empty = endpoint(torch.empty(0, 4), torch.empty(0, 5))

    assert torch.allclose(original.quality, permuted.quality, atol=1e-7)
    assert torch.allclose(original.unary_contribution, permuted.unary_contribution, atol=1e-7)
    assert torch.allclose(original.pair_contribution, permuted.pair_contribution, atol=1e-7)
    assert torch.isfinite(no_pairs.quality)
    assert torch.isfinite(empty.quality)
