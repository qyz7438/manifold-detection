"""ROI structure diagnostics adapted from the basic SCSI project.

The basic project did not validate a single prototype-attraction loss as the
core mechanism.  Its strongest evidence is a transition signature: compact
class representations, stable decision basins, and aligned prototype/basin
geometry.  This module provides the detection-side analogue for foreground ROI
features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.cone_projection import (
    compute_class_prototypes,
    normalize_l2,
)


@dataclass(frozen=True)
class PrototypeBasinGeometry:
    """Prototype/basin graph alignment diagnostics for ROI features."""

    score: torch.Tensor
    components: dict[str, torch.Tensor]
    prototype_graph: torch.Tensor
    sample_graph: torch.Tensor
    basin_graph: torch.Tensor | None = None
    weight_graph: torch.Tensor | None = None


@dataclass(frozen=True)
class ROIStructureSignature:
    """Detection analogue of the basic project's compactness/PBG signature."""

    compactness_energy: torch.Tensor
    basin_retention: torch.Tensor
    prototype_basin_geometry: torch.Tensor
    simplex_energy: torch.Tensor
    components: dict[str, torch.Tensor]


def roi_compactness_energy(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return normalized distance from ROI features to their class prototypes.

    Lower is better.  The numerator is mean squared distance to the matching
    class prototype.  The denominator is mean squared inter-prototype distance,
    so the metric remains comparable when feature scales change.
    """
    _check_feature_labels(features, labels)
    if prototypes.ndim != 2 or prototypes.shape[1] != features.shape[1]:
        raise ValueError("prototypes must have shape (C, D)")
    valid = _valid_label_mask(labels, prototypes.shape[0])
    if not bool(valid.any()):
        return features.new_tensor(0.0)

    z = normalize_l2(features[valid])
    proto = normalize_l2(prototypes)
    target_proto = proto[labels[valid].long()]
    intra = (z - target_proto).square().sum(dim=-1).mean()

    nonzero_proto = proto[prototypes.norm(dim=-1) > eps]
    if nonzero_proto.shape[0] < 2:
        return intra
    spacing = torch.pdist(nonzero_proto, p=2).square().mean().clamp_min(float(eps))
    return intra / spacing


def roi_basin_retention(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    perturb_radius: float = 0.05,
    num_perturbations: int = 4,
    margin: float = 0.0,
    temperature: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estimate whether ROI features remain in the correct class basin.

    Higher is better.  Each feature is lightly perturbed in normalized feature
    space and scored by the prototype margin between the true class and the
    nearest competing class.
    """
    _check_feature_labels(features, labels)
    if prototypes.ndim != 2 or prototypes.shape[1] != features.shape[1]:
        raise ValueError("prototypes must have shape (C, D)")
    if perturb_radius < 0.0:
        raise ValueError("perturb_radius must be non-negative")
    if num_perturbations < 0:
        raise ValueError("num_perturbations must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    valid = _valid_label_mask(labels, prototypes.shape[0])
    if not bool(valid.any()):
        return features.new_tensor(0.0)
    if prototypes.shape[0] < 2:
        return features.new_tensor(1.0)

    z = normalize_l2(features[valid])
    y = labels[valid].long()
    proto = normalize_l2(prototypes)
    probes = [z]
    for _ in range(int(num_perturbations)):
        noise = torch.randn(
            z.shape,
            device=z.device,
            dtype=z.dtype,
            generator=generator,
        )
        probes.append(normalize_l2(z + float(perturb_radius) * noise))
    all_probes = torch.cat(probes, dim=0)
    all_labels = y.repeat(len(probes))

    scores = all_probes @ proto.t()
    true_scores = scores.gather(1, all_labels[:, None]).squeeze(1)
    other_scores = scores.masked_fill(
        F.one_hot(all_labels, num_classes=prototypes.shape[0]).bool(),
        float("-inf"),
    ).max(dim=1).values
    margins = true_scores - other_scores
    return torch.sigmoid((margins - float(margin)) / float(temperature)).mean()


def simplex_energy(prototypes: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Measure deviation from an equiangular class simplex.

    Lower is better.  This is diagnostic rather than a direct objective for all
    detection settings, especially when only a subset of classes is present.
    """
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape (C, D)")
    proto = normalize_l2(prototypes)
    proto = proto[prototypes.norm(dim=-1) > eps]
    class_count = proto.shape[0]
    if class_count < 2:
        return prototypes.new_tensor(0.0)
    sim = proto @ proto.t()
    off_diag = sim[~torch.eye(class_count, dtype=torch.bool, device=sim.device)]
    target = -1.0 / float(class_count - 1)
    return (off_diag - target).square().mean()


def class_topk_adjacency(
    scores: torch.Tensor,
    *,
    topk: int = 2,
    require_positive: bool = False,
) -> torch.Tensor:
    """Convert a class-class score matrix to a directed top-k adjacency matrix."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must have shape (C, C)")
    class_count = scores.shape[0]
    adjacency = torch.zeros_like(scores)
    if class_count <= 1 or topk <= 0:
        return adjacency

    masked = scores.clone()
    masked.fill_diagonal_(float("-inf"))
    k = min(int(topk), class_count - 1)
    values, indices = masked.topk(k=k, dim=1)
    keep = torch.isfinite(values)
    if require_positive:
        keep = keep & (values > 0)
    rows = torch.arange(class_count, device=scores.device)[:, None].expand_as(indices)
    adjacency[rows[keep], indices[keep]] = 1.0
    return adjacency


def graph_jaccard(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Mean row-wise Jaccard overlap for binary class graphs."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("graphs must have the same 2D shape")
    a_bool = a > 0
    b_bool = b > 0
    intersection = (a_bool & b_bool).sum(dim=1).to(dtype=a.dtype)
    union = (a_bool | b_bool).sum(dim=1).to(dtype=a.dtype)
    row_score = torch.where(union > 0, intersection / union.clamp_min(float(eps)), torch.ones_like(union))
    return row_score.mean()


def basin_leakage_graph(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    """Build a class leakage matrix from ROI classifier logits.

    Row ``i`` contains the average probability mass from true class ``i`` to
    all competing classes.  The diagonal is forced to zero.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if labels.shape != (logits.shape[0],):
        raise ValueError("labels must have shape (N,)")
    if logits.shape[1] < num_classes:
        raise ValueError("logits must contain at least num_classes columns")
    probs = logits[:, :num_classes].softmax(dim=-1)
    graph = logits.new_zeros((num_classes, num_classes))
    valid = _valid_label_mask(labels, num_classes)
    for class_idx in range(int(num_classes)):
        mask = valid & (labels == class_idx)
        if bool(mask.any()):
            graph[class_idx] = probs[mask].mean(dim=0)
    graph.fill_diagonal_(0.0)
    row_sum = graph.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return graph / row_sum


def prototype_basin_geometry(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    prototypes: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    classifier_weight: torch.Tensor | None = None,
    num_classes: int | None = None,
    topk: int = 2,
) -> PrototypeBasinGeometry:
    """Compute a detection-side PrototypeBasinGeometry diagnostic.

    This mirrors the basic project: class prototypes, probe sample prototypes,
    basin leakage, and classifier weights should describe compatible class
    neighborhoods.  Missing optional pieces are skipped and the weighted score
    is renormalized over available components.
    """
    _check_feature_labels(features, labels)
    if num_classes is None:
        if prototypes is not None:
            num_classes = prototypes.shape[0]
        elif logits is not None:
            num_classes = logits.shape[1]
        elif labels.numel() > 0:
            num_classes = int(labels.max().item()) + 1
        else:
            num_classes = 0
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if prototypes is None:
        prototypes = compute_class_prototypes(features, labels, num_classes=int(num_classes))
    if prototypes.ndim != 2 or prototypes.shape[0] != num_classes or prototypes.shape[1] != features.shape[1]:
        raise ValueError("prototypes must have shape (num_classes, D)")

    sample_prototypes = compute_class_prototypes(features, labels, num_classes=int(num_classes))
    prototype_graph = class_topk_adjacency(normalize_l2(prototypes) @ normalize_l2(prototypes).t(), topk=topk)
    sample_graph = class_topk_adjacency(
        normalize_l2(sample_prototypes) @ normalize_l2(sample_prototypes).t(),
        topk=topk,
    )
    components: dict[str, torch.Tensor] = {
        "j_proto_sample": graph_jaccard(prototype_graph, sample_graph),
        "simplex_score": 1.0 / (1.0 + simplex_energy(prototypes)),
    }
    weighted_terms = [
        (0.25, components["j_proto_sample"]),
        (0.20, components["simplex_score"]),
    ]

    basin_graph = None
    if logits is not None:
        basin_graph = class_topk_adjacency(
            basin_leakage_graph(logits, labels, num_classes=int(num_classes)),
            topk=topk,
            require_positive=True,
        )
        components["j_basin_proto"] = graph_jaccard(basin_graph, prototype_graph)
        components["j_basin_sample"] = graph_jaccard(basin_graph, sample_graph)
        weighted_terms.extend(
            [
                (0.25, components["j_basin_proto"]),
                (0.15, components["j_basin_sample"]),
            ]
        )

    weight_graph = None
    if classifier_weight is not None:
        if classifier_weight.ndim != 2 or classifier_weight.shape[0] < num_classes:
            raise ValueError("classifier_weight must have shape (>=num_classes, D)")
        if classifier_weight.shape[1] != features.shape[1]:
            raise ValueError("classifier_weight feature dimension must match features")
        weight = normalize_l2(classifier_weight[:num_classes])
        weight_graph = class_topk_adjacency(weight @ weight.t(), topk=topk)
        components["j_proto_weight"] = graph_jaccard(prototype_graph, weight_graph)
        weighted_terms.append((0.15, components["j_proto_weight"]))

    total_weight = sum(weight for weight, _ in weighted_terms)
    score = sum(weight * value for weight, value in weighted_terms) / total_weight
    return PrototypeBasinGeometry(
        score=score,
        components=components,
        prototype_graph=prototype_graph,
        sample_graph=sample_graph,
        basin_graph=basin_graph,
        weight_graph=weight_graph,
    )


def roi_structure_signature(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    prototypes: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    classifier_weight: torch.Tensor | None = None,
    num_classes: int | None = None,
    topk: int = 2,
    perturb_radius: float = 0.05,
    num_perturbations: int = 4,
) -> ROIStructureSignature:
    """Bundle compactness, basin retention, and PBG into one ROI signature."""
    _check_feature_labels(features, labels)
    if num_classes is None:
        if prototypes is not None:
            num_classes = prototypes.shape[0]
        elif logits is not None:
            num_classes = logits.shape[1]
        elif labels.numel() > 0:
            num_classes = int(labels.max().item()) + 1
        else:
            num_classes = 0
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if prototypes is None:
        prototypes = compute_class_prototypes(features, labels, num_classes=int(num_classes))

    pbg = prototype_basin_geometry(
        features,
        labels,
        prototypes=prototypes,
        logits=logits,
        classifier_weight=classifier_weight,
        num_classes=int(num_classes),
        topk=topk,
    )
    compactness = roi_compactness_energy(features, labels, prototypes)
    retention = roi_basin_retention(
        features,
        labels,
        prototypes,
        perturb_radius=perturb_radius,
        num_perturbations=num_perturbations,
    )
    simplex = simplex_energy(prototypes)
    components = dict(pbg.components)
    components.update(
        {
            "roi_compactness_energy": compactness,
            "roi_basin_retention": retention,
            "roi_pbg": pbg.score,
            "roi_simplex_energy": simplex,
        }
    )
    return ROIStructureSignature(
        compactness_energy=compactness,
        basin_retention=retention,
        prototype_basin_geometry=pbg.score,
        simplex_energy=simplex,
        components=components,
    )


def _check_feature_labels(features: torch.Tensor, labels: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have shape (N,)")


def _valid_label_mask(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (labels >= 0) & (labels < int(num_classes))
