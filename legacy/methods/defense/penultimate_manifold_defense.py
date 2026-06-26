r"""Penultimate-layer manifold defense for adversarial patch detection.

This defense combines two validated signals:

1.  **Phase-only / high-frequency AFM** at the ROI feature level
    (``box_roi_pool`` output → ``box_head`` input).  The block operates on the
    256-channel ROI feature map in the Fourier domain, adding a phase residual
    and suppressing anomalous high-frequency magnitude.

2.  **Penultimate-layer manifold gate** at the box-head output
    (``box_head`` output → ``box_predictor`` input).  A data-driven manifold is
    fit on clean training penultimate features; at inference anomalous
    penultimate vectors are projected back toward the nearest clean prototype.

Both components are inserted into the detector forward graph so the defense is
in-network and differentiable w.r.t. the AFM parameters.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from spectral_detection_posttrain.methods.afm.micro_afm import (
    PhaseOnlyAFMBlock,
    build_afm_block,
)


class PhaseHighFreqAFMBlock(nn.Module):
    r"""Phase-residual + hard high-frequency magnitude gate.

    This block captures the two signals we selected:

    * **Phase residual**: learned phase modulation (the critical AFM mechanism).
    * **High-frequency suppression**: a fixed radial mask that suppresses the
      highest 30% of frequency bins, which is where adversarial patches
      introduce the most anomalous energy.

    The block is identity-preserving: at init the phase network outputs zero
    and the high-frequency gate is inactive, so ``forward(x) ≈ x``.

    Args:
        channels: number of input channels (e.g. 256 for Faster R-CNN ROI).
        gate_strength: multiplier for both phase and magnitude gates.
        high_freq_ratio: fraction of highest radial frequencies to suppress.
    """

    def __init__(self, channels: int, gate_strength: float = 0.6, high_freq_ratio: float = 0.3):
        super().__init__()
        self.gate_strength = gate_strength
        self.high_freq_ratio = high_freq_ratio
        mid = max(channels // 4, 8)

        # Phase residual network: near-identity at init.
        self.phase_res = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.InstanceNorm2d(mid),
            nn.Tanh(),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid),
            nn.Tanh(),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.ones(1))
        self._eps = 1e-3

        for m in self.phase_res.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)

    def _high_freq_mask(self, shape: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        """Return a radial mask that is 1.0 for the highest frequencies."""
        _, _, h, w_rfft = shape
        # rfft2 width = original_width // 2 + 1  -> reconstruct even original width.
        spatial_w = max(2 * (w_rfft - 1), 2)
        freq_h = torch.fft.fftfreq(h, device=device)
        freq_w = torch.fft.rfftfreq(spatial_w, device=device)
        grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
        radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        radius = radius / radius.max().clamp_min(1e-6)
        threshold = 1.0 - self.high_freq_ratio
        return (radius > threshold).float().view(1, 1, h, w_rfft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"PhaseHighFreqAFMBlock expects 4-D input, got {x.ndim}-D")

        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + self._eps)

        # Phase residual.
        pha = pha + self.gate_strength * self.phase_res(pha)

        # Hard high-frequency magnitude suppression.
        hf_mask = self._high_freq_mask(mag.shape, mag.device)
        mag = mag * (1.0 - self.gate_strength * hf_mask)

        fr = mag * torch.exp(1j * pha)
        freq_out = torch.fft.irfft2(fr, s=x.shape[-2:], norm="ortho")
        freq_out = F.relu(freq_out, inplace=False)
        return x + self.residual_scale * freq_out


class PenultimateManifold:
    r"""Data-driven manifold on detector penultimate features.

    The manifold is fit on clean training examples using:

    * z-score normalization
    * PCA whitening
    * k-NN reference bank

    At inference it returns a per-vector anomaly score (distance to the k-th
    nearest clean neighbor) and can project anomalous vectors back toward the
    local clean mean.
    """

    def __init__(self, n_components: int = 50, n_neighbors: int = 5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.nn: NearestNeighbors | None = None
        self.ref_pca: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(self, features: np.ndarray, threshold_percentile: float = 95.0) -> "PenultimateManifold":
        """Fit manifold on clean penultimate features.

        Args:
            features: array of shape ``(N, D)``.
            threshold_percentile: percentile of training distances used as the
                anomaly threshold.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got {features.ndim}-D")

        n_samples = features.shape[0]
        n_comp = min(self.n_components, n_samples - 1, features.shape[1])

        self.scaler.fit(features)
        X = self.scaler.transform(features)
        self.pca = PCA(n_components=n_comp, whiten=True, random_state=42)
        X_pca = self.pca.fit_transform(X)
        self.ref_pca = X_pca

        k = min(self.n_neighbors, n_samples)
        self.nn = NearestNeighbors(n_neighbors=max(1, k), metric="euclidean")
        self.nn.fit(X_pca)

        dist, _ = self.nn.kneighbors(X_pca)
        dist_mean = dist.mean(axis=1)
        self.threshold = float(np.percentile(dist_mean, threshold_percentile))
        return self

    def _to_pca(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        X = self.scaler.transform(features)
        return self.pca.transform(X)  # type: ignore[union-attr]

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """Return per-vector distance to clean manifold."""
        X_pca = self._to_pca(features)
        dist, _ = self.nn.kneighbors(X_pca)  # type: ignore[union-attr]
        return dist.mean(axis=1)

    def project(self, features: np.ndarray) -> np.ndarray:
        """Project features back toward the clean manifold if anomalous.

        Returns the original features for in-manifold vectors and the local
        clean prototype mean for anomalous vectors.
        """
        X_pca = self._to_pca(features)
        dist, idx = self.nn.kneighbors(X_pca)  # type: ignore[union-attr]
        dist_mean = dist.mean(axis=1)
        mask = dist_mean > self.threshold  # type: ignore[operator]

        # Local clean prototype: mean of k nearest neighbors in PCA space.
        prototypes = self.ref_pca[idx].mean(axis=1)  # type: ignore[index]
        X_pca_out = np.where(mask[:, None], prototypes, X_pca)

        X_out = self.pca.inverse_transform(X_pca_out)  # type: ignore[union-attr]
        return self.scaler.inverse_transform(X_out)


class AFMBoxHead(nn.Module):
    """Wraps a detector box_head with an AFM purification block."""

    def __init__(self, box_head: nn.Module, afm: nn.Module):
        super().__init__()
        self.afm = afm
        self.head = box_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.afm(x))


class ManifoldBoxPredictor(nn.Module):
    """Wraps a detector box_predictor with a penultimate manifold gate."""

    def __init__(
        self,
        predictor: nn.Module,
        manifold: PenultimateManifold,
        enabled: bool = True,
    ):
        super().__init__()
        self.predictor = predictor
        self.manifold = manifold
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is penultimate feature from box_head.
        if self.enabled and self.manifold.ref_pca is not None and x.shape[0] > 0:
            x_np = x.detach().cpu().numpy()
            x_proj = self.manifold.project(x_np)
            x = torch.from_numpy(x_proj).to(x.dtype).to(x.device)
        return self.predictor(x)


class PenultimateManifoldDefense(nn.Module):
    r"""In-network defense combining AFM and a penultimate-layer manifold gate.

    The defense wraps an existing Faster R-CNN-style detector:

    * Inserts ``PhaseHighFreqAFMBlock`` between ``box_roi_pool`` and ``box_head``.
    * Inserts ``ManifoldBoxPredictor`` between ``box_head`` and the final
      classification/regression heads.

    The manifold must be fit on clean training data before inference.

    Args:
        detector: a Faster R-CNN (or compatible) detector.
        afm_channels: number of channels in the ROI feature map.
        gate_strength: AFM gate strength.
        high_freq_ratio: fraction of highest frequencies suppressed by AFM.
        manifold_components: PCA components for the penultimate manifold.
        manifold_neighbors: k-NN neighbors for the penultimate manifold.
    """

    def __init__(
        self,
        detector: nn.Module,
        afm_channels: int = 256,
        afm_type: str = "mplseg_phase_only",
        gate_strength: float = 0.6,
        high_freq_ratio: float = 0.3,
        manifold_components: int = 50,
        manifold_neighbors: int = 5,
        enable_manifold_gate: bool = True,
    ):
        super().__init__()
        self.detector = detector

        if afm_type == "phase_high_freq":
            self.afm = PhaseHighFreqAFMBlock(
                channels=afm_channels,
                gate_strength=gate_strength,
                high_freq_ratio=high_freq_ratio,
            )
        elif afm_type == "none":
            self.afm = nn.Identity()
        else:
            # Use validated MPLSeg variants (phase_only, mplseg, etc.).
            block = build_afm_block(afm_type, channels=afm_channels)
            if block is None:
                raise ValueError(f"Unsupported afm_type: {afm_type}")
            self.afm = block

        self.manifold = PenultimateManifold(
            n_components=manifold_components,
            n_neighbors=manifold_neighbors,
        )

        # Wrap box_head and box_predictor in place.
        original_box_head = detector.roi_heads.box_head
        original_predictor = detector.roi_heads.box_predictor
        detector.roi_heads.box_head = AFMBoxHead(original_box_head, self.afm)
        detector.roi_heads.box_predictor = ManifoldBoxPredictor(
            original_predictor, self.manifold, enabled=enable_manifold_gate
        )

    def fit_manifold(
        self,
        penultimate_features: np.ndarray,
        threshold_percentile: float = 95.0,
    ) -> None:
        """Fit the penultimate manifold on clean training features."""
        self.manifold.fit(penultimate_features, threshold_percentile=threshold_percentile)

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None):
        """Forward through the wrapped detector."""
        return self.detector(images, targets)
