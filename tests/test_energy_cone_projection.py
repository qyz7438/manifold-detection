from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.energy_transport import (
    compute_class_prototypes,
    cone_dpog_regularizer,
    cone_residual_alignment_loss,
    cross_entropy_energy,
    decompose_cone_features,
    local_tangent_energy_endpoint,
)


def test_cone_decomposition_residual_is_tangent_to_class_axis() -> None:
    features = torch.tensor([[1.0, 2.0, 0.0], [0.5, 0.0, 1.0]])
    labels = torch.tensor([0, 1])
    prototypes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    decomposition = decompose_cone_features(features, labels, prototypes)

    dot = (decomposition.residual * decomposition.prototypes).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-6)
    reconstructed = decomposition.axial + decomposition.residual
    assert torch.allclose(reconstructed, decomposition.normalized_features, atol=1e-6)


def test_compute_class_prototypes_normalizes_non_empty_classes() -> None:
    features = torch.tensor([[2.0, 0.0], [0.0, 3.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1, 1])

    prototypes = compute_class_prototypes(features, labels, num_classes=3)

    assert torch.allclose(prototypes[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(prototypes[1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(prototypes[2], torch.tensor([0.0, 0.0]))


def test_cone_residual_alignment_loss_is_low_for_matching_residuals() -> None:
    features = torch.tensor([[1.0, 0.4, 0.2]])
    labels = torch.tensor([0])
    prototypes = torch.tensor([[1.0, 0.0, 0.0]])
    same = features.clone()
    flipped = torch.tensor([[1.0, -0.4, -0.2]])

    same_loss = cone_residual_alignment_loss(features, same, labels, prototypes)
    flipped_loss = cone_residual_alignment_loss(features, flipped, labels, prototypes)

    assert same_loss.item() < 1e-5
    assert flipped_loss.item() > 1.9


def test_local_tangent_endpoint_reduces_classifier_energy() -> None:
    features = torch.tensor([[1.0, -0.2, -0.7]], dtype=torch.float32)
    labels = torch.tensor([0])
    prototypes = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    classifier = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[0.0, 3.0, 0.0], [0.0, -3.0, 0.0]]))

    endpoint = local_tangent_energy_endpoint(
        features,
        labels,
        prototypes,
        cross_entropy_energy(classifier),
        steps=30,
        step_size=0.25,
    )

    assert endpoint.endpoint_energy.item() <= endpoint.current_energy.item()
    assert endpoint.dpog.item() > 0.0
    assert endpoint.action_energy.item() > 0.0


def test_cone_dpog_regularizer_is_differentiable() -> None:
    features = torch.tensor([[1.0, -0.2, -0.7]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0])
    prototypes = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    classifier = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[0.0, 3.0, 0.0], [0.0, -3.0, 0.0]]))

    loss, diagnostics = cone_dpog_regularizer(
        features,
        labels,
        prototypes,
        cross_entropy_energy(classifier),
        steps=10,
        step_size=0.2,
        gap_weight=0.5,
        alignment_weight=1.0,
    )
    loss.backward()

    assert loss.ndim == 0
    assert features.grad is not None
    assert diagnostics["roi_dpog"] >= 0.0
    assert diagnostics["roi_cone_action_energy"] >= 0.0
