"""ROI dual-energy diagnostics adapted from the basic SCSI project.

The basic project frames useful representation structure as a simultaneous
intra-class / inter-class constraint: foreground samples should collapse into
stable class basins, while class prototypes keep a valid relation structure
instead of collapsing independently.  This module provides the detection-side
analogue for foreground ROI features.
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


@dataclass(frozen=True)
class ROIDualEnergy:
    """Intra/inter dual-energy objective for ROI representations."""

    compact_energy: torch.Tensor
    basin_energy: torch.Tensor
    intra_energy: torch.Tensor
    inter_energy: torch.Tensor
    dual_energy: torch.Tensor
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


def roi_basin_energy(
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
    """Softplus energy for leaving the correct class basin.

    Lower is better.  This is the energy counterpart of
    :func:`roi_basin_retention`, matching the basic project's ``E_basin``:
    perturbed ROI embeddings should keep a positive prototype margin against
    all competing classes.
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
    if not bool(valid.any()) or prototypes.shape[0] < 2:
        return features.new_tensor(0.0)

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
    return F.softplus((float(margin) - margins) / float(temperature)).mean()


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


def centered_relation_matrix(prototypes: torch.Tensor) -> torch.Tensor:
    """Centered cosine relation matrix for class prototypes."""
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape (C, D)")
    relation = normalize_l2(prototypes) @ normalize_l2(prototypes).t()
    return _center_square_matrix(relation)


def relation_cka(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """CKA alignment between two square class-relation matrices."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("relation matrices must have the same 2D shape")
    a_centered = _center_square_matrix(a)
    b_centered = _center_square_matrix(b)
    numerator = (a_centered * b_centered).sum()
    denominator = (a_centered.square().sum() * b_centered.square().sum()).sqrt().clamp_min(float(eps))
    return numerator / denominator


def inter_class_separation_energy(
    prototypes: torch.Tensor,
    *,
    cosine_margin: float = 0.0,
) -> torch.Tensor:
    """Penalize class prototypes that are too close to each other."""
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape (C, D)")
    proto = normalize_l2(prototypes)
    class_count = proto.shape[0]
    if class_count < 2:
        return prototypes.new_tensor(0.0)
    sim = proto @ proto.t()
    off_diag = sim[~torch.eye(class_count, dtype=torch.bool, device=sim.device)]
    return (off_diag - float(cosine_margin)).clamp_min(0.0).square().mean()


def inter_class_relation_energy(
    prototypes: torch.Tensor,
    *,
    reference_prototypes: torch.Tensor | None = None,
    classifier_weight: torch.Tensor | None = None,
    target_relation: torch.Tensor | None = None,
    relation_weight: float = 1.0,
    separation_weight: float = 0.25,
    simplex_weight: float = 0.0,
    cosine_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the inter-class half of the ROI dual-energy objective.

    The object-detection class manifold is usually not a known algebraic group,
    so the maintained default uses relation alignment anchors that are actually
    available in a detector: frozen/reference prototypes, classifier weights,
    or a future semantic/text relation matrix.  A separation term keeps the
    energy from being minimized by collapsing all class prototypes together.
    """
    if prototypes.ndim != 2:
        raise ValueError("prototypes must have shape (C, D)")
    if relation_weight < 0.0 or separation_weight < 0.0 or simplex_weight < 0.0:
        raise ValueError("energy weights must be non-negative")
    class_count, feature_dim = prototypes.shape
    relation = centered_relation_matrix(prototypes)
    components: dict[str, torch.Tensor] = {}
    relation_terms: list[torch.Tensor] = []

    if reference_prototypes is not None:
        if reference_prototypes.shape != prototypes.shape:
            raise ValueError("reference_prototypes must match prototypes")
        align = relation_cka(relation, centered_relation_matrix(reference_prototypes))
        components["inter_reference_alignment"] = align
        relation_terms.append(1.0 - align.clamp(0.0, 1.0))

    if classifier_weight is not None:
        if classifier_weight.ndim != 2 or classifier_weight.shape[0] < class_count:
            raise ValueError("classifier_weight must have shape (>=C, D)")
        if classifier_weight.shape[1] != feature_dim:
            raise ValueError("classifier_weight feature dimension must match prototypes")
        weight_relation = centered_relation_matrix(classifier_weight[:class_count])
        align = relation_cka(relation, weight_relation)
        components["inter_classifier_alignment"] = align
        relation_terms.append(1.0 - align.clamp(0.0, 1.0))

    if target_relation is not None:
        if target_relation.shape != (class_count, class_count):
            raise ValueError("target_relation must have shape (C, C)")
        align = relation_cka(relation, target_relation)
        components["inter_target_alignment"] = align
        relation_terms.append(1.0 - align.clamp(0.0, 1.0))

    if relation_terms:
        relation_energy = torch.stack(relation_terms).mean()
    else:
        relation_energy = prototypes.new_tensor(0.0)

    separation = inter_class_separation_energy(prototypes, cosine_margin=cosine_margin)
    simplex = simplex_energy(prototypes)
    energy = (
        float(relation_weight) * relation_energy
        + float(separation_weight) * separation
        + float(simplex_weight) * simplex
    )
    components.update(
        {
            "inter_relation_energy": relation_energy,
            "inter_separation_energy": separation,
            "inter_simplex_energy": simplex,
            "e_inter": energy,
        }
    )
    return energy, components


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


def roi_dual_energy(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    prototypes: torch.Tensor | None = None,
    reference_prototypes: torch.Tensor | None = None,
    classifier_weight: torch.Tensor | None = None,
    target_relation: torch.Tensor | None = None,
    num_classes: int | None = None,
    compact_weight: float = 0.5,
    basin_weight: float = 0.5,
    intra_weight: float = 0.5,
    inter_weight: float = 0.5,
    relation_weight: float = 1.0,
    separation_weight: float = 0.25,
    simplex_weight: float = 0.0,
    cosine_margin: float = 0.0,
    perturb_radius: float = 0.05,
    num_perturbations: int = 4,
    basin_margin: float = 0.0,
    basin_temperature: float = 0.05,
) -> ROIDualEnergy:
    """Compute the ROI intra/inter dual energy.

    This is the detection-side analogue of the basic project's GCRE form:

    ``E_intra = compactness + basin stability``
    ``E_inter = class-relation alignment + class separation``

    The returned tensors are differentiable and can be logged first, then used
    as a small auxiliary objective once full detector metrics support it.
    """
    _check_feature_labels(features, labels)
    if compact_weight < 0.0 or basin_weight < 0.0 or intra_weight < 0.0 or inter_weight < 0.0:
        raise ValueError("energy weights must be non-negative")
    if num_classes is None:
        if prototypes is not None:
            num_classes = prototypes.shape[0]
        elif classifier_weight is not None:
            num_classes = classifier_weight.shape[0]
        elif target_relation is not None:
            num_classes = target_relation.shape[0]
        elif labels.numel() > 0:
            num_classes = int(labels.max().item()) + 1
        else:
            num_classes = 0
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if prototypes is None:
        prototypes = compute_class_prototypes(features, labels, num_classes=int(num_classes))

    compact = roi_compactness_energy(features, labels, prototypes)
    basin = roi_basin_energy(
        features,
        labels,
        prototypes,
        perturb_radius=perturb_radius,
        num_perturbations=num_perturbations,
        margin=basin_margin,
        temperature=basin_temperature,
    )
    intra_norm = float(compact_weight) + float(basin_weight)
    if intra_norm <= 0.0:
        intra = features.new_tensor(0.0)
    else:
        intra = (float(compact_weight) * compact + float(basin_weight) * basin) / intra_norm

    inter, inter_components = inter_class_relation_energy(
        prototypes,
        reference_prototypes=reference_prototypes,
        classifier_weight=classifier_weight,
        target_relation=target_relation,
        relation_weight=relation_weight,
        separation_weight=separation_weight,
        simplex_weight=simplex_weight,
        cosine_margin=cosine_margin,
    )
    dual_norm = float(intra_weight) + float(inter_weight)
    if dual_norm <= 0.0:
        dual = features.new_tensor(0.0)
    else:
        dual = (float(intra_weight) * intra + float(inter_weight) * inter) / dual_norm

    components = dict(inter_components)
    components.update(
        {
            "e_compact": compact,
            "e_basin": basin,
            "e_intra": intra,
            "e_inter": inter,
            "dual_energy": dual,
        }
    )
    return ROIDualEnergy(
        compact_energy=compact,
        basin_energy=basin,
        intra_energy=intra,
        inter_energy=inter,
        dual_energy=dual,
        components=components,
    )


def _check_feature_labels(features: torch.Tensor, labels: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have shape (N,)")


def _valid_label_mask(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (labels >= 0) & (labels < int(num_classes))


def _center_square_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must have shape (C, C)")
    return matrix - matrix.mean(dim=0, keepdim=True) - matrix.mean(dim=1, keepdim=True) + matrix.mean()
