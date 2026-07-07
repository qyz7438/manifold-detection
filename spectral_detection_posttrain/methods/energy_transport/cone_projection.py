"""Cone-internal projection objectives for ROI manifold structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


EnergyFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ConeDecomposition:
    """Feature decomposition into class-axis and cone-internal residual parts."""

    normalized_features: torch.Tensor
    prototypes: torch.Tensor
    axial: torch.Tensor
    residual: torch.Tensor
    axial_coeff: torch.Tensor
    residual_norm: torch.Tensor


@dataclass(frozen=True)
class ConeProjectionEndpoint:
    """Local low-energy endpoint inside a class cone."""

    endpoint_features: torch.Tensor
    endpoint_residual: torch.Tensor
    current_energy: torch.Tensor
    endpoint_energy: torch.Tensor
    dpog: torch.Tensor
    residual_alignment: torch.Tensor
    action_energy: torch.Tensor


def normalize_l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize vectors along the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def compute_class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute class prototypes from features."""
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have shape (N,)")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    prototypes = []
    for class_idx in range(int(num_classes)):
        mask = labels == class_idx
        if mask.any():
            proto = features[mask].mean(dim=0)
        else:
            proto = torch.zeros(features.shape[1], device=features.device, dtype=features.dtype)
        prototypes.append(normalize_l2(proto) if normalize else proto)
    return torch.stack(prototypes, dim=0)


def decompose_cone_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    normalize_prototypes: bool = True,
) -> ConeDecomposition:
    """Decompose ROI features around their class prototype axis."""
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have shape (N,)")
    if prototypes.ndim != 2 or prototypes.shape[1] != features.shape[1]:
        raise ValueError("prototypes must have shape (C, D)")
    if labels.numel() and (labels.min() < 0 or labels.max() >= prototypes.shape[0]):
        raise ValueError("labels contain class indices outside prototypes")

    z = normalize_l2(features)
    proto = normalize_l2(prototypes) if normalize_prototypes else prototypes
    mu = proto[labels.long()]
    axial_coeff = (z * mu).sum(dim=-1, keepdim=True)
    axial = axial_coeff * mu
    residual = project_to_tangent(z, mu)
    residual_norm = residual.norm(dim=-1, keepdim=True)
    return ConeDecomposition(
        normalized_features=z,
        prototypes=mu,
        axial=axial,
        residual=residual,
        axial_coeff=axial_coeff,
        residual_norm=residual_norm,
    )


def project_to_tangent(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``v`` along ``axis``."""
    return v - (v * axis).sum(dim=-1, keepdim=True) * axis


def cone_residual_alignment_loss(
    features: torch.Tensor,
    reference_features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    detach_reference: bool = True,
) -> torch.Tensor:
    """DPA-style loss between cone-internal residual directions."""
    current = decompose_cone_features(features, labels, prototypes).residual
    reference = decompose_cone_features(reference_features, labels, prototypes).residual
    current = normalize_l2(current)
    reference = normalize_l2(reference)
    if detach_reference:
        reference = reference.detach()
    return (1.0 - (current * reference).sum(dim=-1)).mean()


def local_tangent_energy_endpoint(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    energy_fn: EnergyFn,
    *,
    steps: int = 5,
    step_size: float = 0.1,
    normalize_prototypes: bool = True,
) -> ConeProjectionEndpoint:
    """Search for a low-energy endpoint by rotating residual directions.

    The search preserves the sample's class-axis coefficient and residual norm.
    This mirrors the basic project's DPOG idea: after a feature has entered a
    class cone, the unconstrained degree of freedom is the tangential residual
    direction, not the class endpoint itself.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    decomposition = decompose_cone_features(
        features,
        labels,
        prototypes,
        normalize_prototypes=normalize_prototypes,
    )
    current_energy = _check_energy_shape(
        energy_fn(decomposition.normalized_features, labels),
        features.shape[0],
        "current_energy",
    )

    mu = decomposition.prototypes.detach()
    axial = decomposition.axial.detach()
    residual_norm = decomposition.residual_norm.detach()
    q = decomposition.residual.detach().clone()
    if q.numel() == 0:
        endpoint_energy = current_energy.detach()
        zero = current_energy * 0.0
        return ConeProjectionEndpoint(
            endpoint_features=decomposition.normalized_features.detach(),
            endpoint_residual=decomposition.residual.detach(),
            current_energy=current_energy,
            endpoint_energy=endpoint_energy,
            dpog=zero,
            residual_alignment=zero,
            action_energy=zero,
        )

    q.requires_grad_(True)
    for _ in range(int(steps)):
        q_tangent = project_to_tangent(q, mu)
        q_tangent = normalize_l2(q_tangent) * residual_norm
        z_candidate = normalize_l2(axial + q_tangent)
        energy = _check_energy_shape(energy_fn(z_candidate, labels), features.shape[0], "candidate_energy").mean()
        grad = torch.autograd.grad(energy, q, create_graph=False)[0]
        with torch.no_grad():
            q -= float(step_size) * grad
            q.copy_(project_to_tangent(q, mu))
            q.copy_(normalize_l2(q) * residual_norm)
        q.requires_grad_(True)

    endpoint_residual = project_to_tangent(q.detach(), mu)
    endpoint_residual = normalize_l2(endpoint_residual) * residual_norm
    endpoint_features = normalize_l2(axial + endpoint_residual)
    endpoint_energy = _check_energy_shape(
        energy_fn(endpoint_features.detach(), labels),
        features.shape[0],
        "endpoint_energy",
    ).detach()
    dpog = (current_energy - endpoint_energy).clamp_min(0.0)

    current_residual_unit = normalize_l2(decomposition.residual)
    endpoint_residual_unit = normalize_l2(endpoint_residual).detach()
    residual_alignment = 1.0 - (current_residual_unit * endpoint_residual_unit).sum(dim=-1)
    action_energy = (endpoint_features.detach() - decomposition.normalized_features).square().sum(dim=-1)
    return ConeProjectionEndpoint(
        endpoint_features=endpoint_features.detach(),
        endpoint_residual=endpoint_residual.detach(),
        current_energy=current_energy,
        endpoint_energy=endpoint_energy,
        dpog=dpog,
        residual_alignment=residual_alignment,
        action_energy=action_energy,
    )


def cone_dpog_regularizer(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    energy_fn: EnergyFn,
    *,
    steps: int = 5,
    step_size: float = 0.1,
    gap_weight: float = 1.0,
    alignment_weight: float = 1.0,
    action_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a scalar RoI-DPOG regularizer and detached diagnostics."""
    if gap_weight < 0.0 or alignment_weight < 0.0 or action_weight < 0.0:
        raise ValueError("regularizer weights must be non-negative")
    endpoint = local_tangent_energy_endpoint(
        features,
        labels,
        prototypes,
        energy_fn,
        steps=steps,
        step_size=step_size,
    )
    loss = (
        float(gap_weight) * endpoint.dpog.mean()
        + float(alignment_weight) * endpoint.residual_alignment.mean()
        + float(action_weight) * endpoint.action_energy.mean()
    )
    diagnostics = {
        "roi_dpog": float(endpoint.dpog.detach().mean().item()),
        "roi_cone_alignment": float(endpoint.residual_alignment.detach().mean().item()),
        "roi_cone_action_energy": float(endpoint.action_energy.detach().mean().item()),
        "roi_energy_current": float(endpoint.current_energy.detach().mean().item()),
        "roi_energy_endpoint": float(endpoint.endpoint_energy.detach().mean().item()),
    }
    return loss, diagnostics


def cross_entropy_energy(classifier: torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]) -> EnergyFn:
    """Wrap a classifier into a per-sample cross-entropy energy function."""

    def energy(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = classifier(features)
        return F.cross_entropy(logits, labels.long(), reduction="none")

    return energy


def _check_energy_shape(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape ({batch_size},), got {value.shape}")
    return value
