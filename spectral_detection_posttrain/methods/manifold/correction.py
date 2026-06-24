"""Manifold-aware wrapper for detector box predictors."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.prototype_bank import PrototypeBank
from spectral_detection_posttrain.methods.manifold.transport_head import TransportHead


class ManifoldCorrectionPredictor(nn.Module):
    """Apply a learnable manifold correction before a detector predictor.

    The module keeps the original predictor interface intact. Given a
    penultimate ROI feature vector, it computes distances to all class
    prototypes, asks ``TransportHead`` for a low-energy feature displacement,
    and forwards the corrected feature into the wrapped predictor.
    """

    def __init__(
        self,
        predictor: nn.Module,
        prototype_bank: PrototypeBank,
        transport_head: TransportHead,
        gamma: float = 1.0,
        tau: float | None = None,
        normalize_features: bool = True,
        background_index: int = 0,
        correction_mode: str = "residual",
        endpoint_gate_init: float = 0.25,
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        correction_mode = correction_mode.replace("-", "_")
        if correction_mode not in {"residual", "endpoint", "gated_endpoint"}:
            raise ValueError(
                "correction_mode must be one of: residual, endpoint, gated_endpoint"
            )
        if not 0.0 < endpoint_gate_init < 1.0:
            raise ValueError("endpoint_gate_init must be in (0, 1)")
        expected = prototype_bank.num_classes * prototype_bank.num_prototypes_per_class
        if transport_head.num_prototypes != expected:
            raise ValueError(
                "transport_head.num_prototypes must match flattened prototype "
                f"count {expected}, got {transport_head.num_prototypes}"
            )
        if transport_head.feature_dim != prototype_bank.feature_dim:
            raise ValueError("transport_head feature_dim must match prototype_bank")

        self.base_predictor = predictor
        self.prototype_bank = prototype_bank
        self.transport_head = transport_head
        self.gamma = gamma
        self.normalize_features = normalize_features
        self.background_index = int(background_index)
        self.correction_mode = correction_mode
        self.endpoint_gate: nn.Linear | None = None
        if self.correction_mode == "gated_endpoint":
            self.endpoint_gate = nn.Linear(prototype_bank.feature_dim, 1)
            gate_logit = math.log(endpoint_gate_init / (1.0 - endpoint_gate_init))
            nn.init.zeros_(self.endpoint_gate.weight)
            nn.init.constant_(self.endpoint_gate.bias, gate_logit)

        if tau is not None:
            if tau <= 0.0:
                raise ValueError("tau must be positive")
            self.transport_head.tau = tau

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gamma == 0.0 or features.numel() == 0:
            return self.base_predictor(features)

        preliminary_logits, _ = self.base_predictor(features)
        corrected = features + self.gamma * self.correction_field(
            features, preliminary_logits
        )
        return self.base_predictor(corrected)

    def correction_field(
        self, features: torch.Tensor, class_logits: torch.Tensor
    ) -> torch.Tensor:
        class_probs = self.foreground_class_probs(class_logits).detach()
        return self.correction_field_from_class_weights(features, class_probs)

    def foreground_class_probs(self, class_logits: torch.Tensor) -> torch.Tensor:
        """Convert detector logits to foreground-only class weights."""
        if class_logits.ndim != 2 or class_logits.shape[1] != self.prototype_bank.num_classes:
            raise ValueError(
                "class_logits must have shape "
                f"(B, {self.prototype_bank.num_classes}), got {class_logits.shape}"
            )

        class_probs = F.softmax(class_logits, dim=-1)
        if 0 <= self.background_index < class_probs.shape[1]:
            class_probs = class_probs.clone()
            class_probs[:, self.background_index] = 0.0
        return self._normalize_class_weights(class_probs)

    def correction_field_from_class_weights(
        self, features: torch.Tensor, class_weights: torch.Tensor
    ) -> torch.Tensor:
        """Predict the correction field using explicit class gates.

        ``class_weights`` has shape ``(B, C)`` and is multiplied with the
        per-class prototype weights before selecting residual slots from the
        flattened ``C * K`` transport head.  This is the same field used by
        active inference correction and by the auxiliary active loss.
        """
        if features.ndim != 2:
            raise ValueError(f"features must have shape (B, D), got {features.shape}")
        if features.shape[1] != self.prototype_bank.feature_dim:
            raise ValueError(
                f"features must have dim {self.prototype_bank.feature_dim}, "
                f"got {features.shape[1]}"
            )
        if class_weights.shape != (features.shape[0], self.prototype_bank.num_classes):
            raise ValueError(
                "class_weights must have shape "
                f"(B, {self.prototype_bank.num_classes}), got {class_weights.shape}"
            )

        class_probs, weights = self._class_and_prototype_weights(features, class_weights)
        if self.correction_mode in {"endpoint", "gated_endpoint"}:
            target = self.endpoint_from_weights(features, weights)
            field = target - features
            if self.endpoint_gate is not None:
                field = torch.sigmoid(self.endpoint_gate(features)) * field
            return field

        residuals = self.transport_head.mlp(features).view(
            features.shape[0],
            self.prototype_bank.num_classes,
            self.prototype_bank.num_prototypes_per_class,
            self.transport_head.feature_dim,
        )
        return torch.einsum("bck,bckd->bd", weights, residuals)

    def endpoint_from_weights(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = self.prototype_bank.get_prototypes().to(
            device=features.device,
            dtype=features.dtype,
        )
        return torch.einsum("bck,ckd->bd", weights, prototypes)

    def endpoint_gate_parameters(self) -> list[nn.Parameter]:
        if self.endpoint_gate is None:
            return []
        return list(self.endpoint_gate.parameters())

    def _class_and_prototype_weights(
        self,
        features: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = self.prototype_bank.get_prototypes().to(
            device=features.device,
            dtype=features.dtype,
        )

        distance_features = features
        distance_prototypes = prototypes
        if self.normalize_features:
            distance_features = F.normalize(distance_features, dim=-1)
            distance_prototypes = F.normalize(distance_prototypes, dim=-1)

        squared_distances = (
            distance_features[:, None, None, :] - distance_prototypes[None, :, :, :]
        ).pow(2).sum(dim=-1)

        class_probs = self._normalize_class_weights(
            class_weights.to(device=features.device, dtype=features.dtype)
        )
        proto_probs = F.softmax(-squared_distances / self.transport_head.tau, dim=-1)
        weights = class_probs[:, :, None] * proto_probs
        return class_probs, weights

    def _normalize_class_weights(self, class_weights: torch.Tensor) -> torch.Tensor:
        denom = class_weights.sum(dim=-1, keepdim=True)
        if self.prototype_bank.num_classes > 1:
            fallback = torch.ones_like(class_weights)
            if 0 <= self.background_index < fallback.shape[1]:
                fallback[:, self.background_index] = 0.0
            fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            fallback = torch.ones_like(class_weights)
        return torch.where(
            denom > 1e-12,
            class_weights / denom.clamp_min(1e-12),
            fallback,
        )
