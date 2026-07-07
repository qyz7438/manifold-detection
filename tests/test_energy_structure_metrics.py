from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.energy_transport import (
    basin_leakage_graph,
    centered_relation_matrix,
    inter_class_relation_energy,
    prototype_basin_geometry,
    relation_cka,
    roi_basin_energy,
    roi_basin_retention,
    roi_compactness_energy,
    roi_dual_energy,
    roi_structure_signature,
    simplex_energy,
)


def test_roi_compactness_energy_is_lower_for_tight_class_clusters() -> None:
    prototypes = torch.eye(3)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    tight = torch.tensor(
        [
            [1.0, 0.02, 0.00],
            [1.0, -0.01, 0.01],
            [0.01, 1.0, 0.02],
            [0.00, 1.0, -0.02],
            [0.02, 0.01, 1.0],
            [-0.02, 0.00, 1.0],
        ]
    )
    loose = torch.tensor(
        [
            [0.4, 0.8, 0.1],
            [0.5, 0.1, 0.7],
            [0.7, 0.4, 0.2],
            [0.1, 0.5, 0.8],
            [0.8, 0.2, 0.4],
            [0.2, 0.8, 0.4],
        ]
    )

    assert roi_compactness_energy(tight, labels, prototypes) < roi_compactness_energy(loose, labels, prototypes)


def test_roi_basin_retention_rewards_large_prototype_margin() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 1])
    stable = torch.tensor([[1.0, 0.1], [-1.0, -0.1]])
    boundary = torch.tensor([[0.02, 1.0], [-0.02, 1.0]])

    stable_retention = roi_basin_retention(
        stable,
        labels,
        prototypes,
        perturb_radius=0.0,
        num_perturbations=0,
    )
    boundary_retention = roi_basin_retention(
        boundary,
        labels,
        prototypes,
        perturb_radius=0.0,
        num_perturbations=0,
    )

    assert stable_retention.item() > 0.95
    assert stable_retention > boundary_retention


def test_roi_basin_energy_is_lower_for_large_prototype_margin() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 1])
    stable = torch.tensor([[1.0, 0.1], [-1.0, -0.1]])
    boundary = torch.tensor([[0.02, 1.0], [-0.02, 1.0]])

    stable_energy = roi_basin_energy(
        stable,
        labels,
        prototypes,
        perturb_radius=0.0,
        num_perturbations=0,
    )
    boundary_energy = roi_basin_energy(
        boundary,
        labels,
        prototypes,
        perturb_radius=0.0,
        num_perturbations=0,
    )

    assert stable_energy < boundary_energy


def test_simplex_energy_is_near_zero_for_etf_triangle() -> None:
    prototypes = torch.tensor(
        [
            [1.0, 0.0],
            [-0.5, 0.8660254],
            [-0.5, -0.8660254],
        ]
    )

    assert simplex_energy(prototypes).item() < 1e-6


def test_inter_class_relation_energy_penalizes_wrong_class_relation() -> None:
    reference = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ]
    )
    aligned = reference.clone()
    wrong = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.8, 0.6],
        ]
    )

    aligned_energy, aligned_components = inter_class_relation_energy(
        aligned,
        reference_prototypes=reference,
        separation_weight=0.0,
    )
    wrong_energy, wrong_components = inter_class_relation_energy(
        wrong,
        reference_prototypes=reference,
        separation_weight=0.0,
    )

    assert relation_cka(centered_relation_matrix(reference), centered_relation_matrix(reference)).item() > 0.99
    assert aligned_components["inter_reference_alignment"] > wrong_components["inter_reference_alignment"]
    assert aligned_energy < wrong_energy


def test_basin_leakage_graph_tracks_off_diagonal_confusion() -> None:
    logits = torch.tensor(
        [
            [4.0, 2.0, -1.0],
            [3.0, 1.8, -1.0],
            [2.0, 4.0, -1.0],
            [1.8, 3.0, -1.0],
            [-1.0, 2.0, 4.0],
            [-1.0, 1.8, 3.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    graph = basin_leakage_graph(logits, labels, num_classes=3)

    assert graph.shape == (3, 3)
    assert torch.allclose(torch.diag(graph), torch.zeros(3))
    assert graph[0, 1] > graph[0, 2]
    assert graph[1, 0] > graph[1, 2]
    assert graph[2, 1] > graph[2, 0]


def test_prototype_basin_geometry_is_higher_when_class_graphs_align() -> None:
    prototypes = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    aligned_features = torch.tensor(
        [
            [1.0, 0.02],
            [0.98, -0.01],
            [0.78, 0.62],
            [0.82, 0.58],
            [-1.0, 0.01],
            [-0.98, -0.02],
        ]
    )
    scrambled_features = torch.tensor(
        [
            [1.0, 0.02],
            [0.98, -0.01],
            [-1.0, 0.01],
            [-0.98, -0.02],
            [0.78, 0.62],
            [0.82, 0.58],
        ]
    )

    aligned = prototype_basin_geometry(aligned_features, labels, prototypes=prototypes, num_classes=3, topk=1)
    scrambled = prototype_basin_geometry(scrambled_features, labels, prototypes=prototypes, num_classes=3, topk=1)

    assert aligned.score > scrambled.score
    assert aligned.components["j_proto_sample"] > scrambled.components["j_proto_sample"]


def test_roi_structure_signature_bundles_basic_style_components() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    features = torch.tensor(
        [
            [1.0, 0.02],
            [0.98, -0.01],
            [0.78, 0.62],
            [0.82, 0.58],
            [-1.0, 0.01],
            [-0.98, -0.02],
        ]
    )
    logits = features @ prototypes.t()

    signature = roi_structure_signature(
        features,
        labels,
        prototypes=prototypes,
        logits=logits,
        classifier_weight=prototypes,
        num_classes=3,
        topk=1,
        perturb_radius=0.0,
        num_perturbations=0,
    )

    assert signature.compactness_energy.item() < 0.01
    assert signature.basin_retention.item() > 0.5
    assert 0.0 <= signature.prototype_basin_geometry.item() <= 1.0
    assert "j_proto_weight" in signature.components


def test_roi_dual_energy_combines_intra_and_inter_constraints() -> None:
    reference = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    compact_features = torch.tensor(
        [
            [1.0, 0.02],
            [0.98, -0.01],
            [0.78, 0.62],
            [0.82, 0.58],
            [-1.0, 0.01],
            [-0.98, -0.02],
        ],
        requires_grad=True,
    )
    loose_features = torch.tensor(
        [
            [0.4, 0.8],
            [0.6, 0.7],
            [-0.7, 0.4],
            [-0.9, 0.1],
            [0.7, 0.5],
            [0.8, 0.4],
        ]
    )
    wrong_inter = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.8, 0.6],
        ]
    )

    good = roi_dual_energy(
        compact_features,
        labels,
        prototypes=reference,
        reference_prototypes=reference,
        perturb_radius=0.0,
        num_perturbations=0,
        separation_weight=0.0,
    )
    loose = roi_dual_energy(
        loose_features,
        labels,
        prototypes=reference,
        reference_prototypes=reference,
        perturb_radius=0.0,
        num_perturbations=0,
        separation_weight=0.0,
    )
    bad_inter = roi_dual_energy(
        compact_features.detach(),
        labels,
        prototypes=wrong_inter,
        reference_prototypes=reference,
        perturb_radius=0.0,
        num_perturbations=0,
        separation_weight=0.0,
    )

    assert good.intra_energy < loose.intra_energy
    assert good.inter_energy < bad_inter.inter_energy
    good.dual_energy.backward()
    assert compact_features.grad is not None
