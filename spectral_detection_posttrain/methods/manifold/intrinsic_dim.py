r"""Intrinsic dimension and local geometry estimators for prototype neighborhoods.

These estimators operate on a buffer of feature points that fall near a
prototype.  The intrinsic dimension is treated as a computed statistic, not a
hard-coded architectural dimension.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class IntrinsicDimEstimator(nn.Module):
    r"""Estimate local intrinsic dimension and geometry of a feature neighborhood.

    Args:
        method: one of ``"pca"`` or ``"twonn"``.
        pca_variance_threshold: cumulative variance threshold for the PCA
            estimator. Ignored when ``method != "pca"``.
        twonn_k: number of nearest neighbors for the TwoNN estimator (the
            second neighbor is used; ``k`` controls how many candidates are
            considered). Ignored when ``method != "twonn"``.
    """

    def __init__(
        self,
        method: str = "pca",
        pca_variance_threshold: float = 0.90,
        twonn_k: int = 2,
    ):
        super().__init__()
        if method not in {"pca", "twonn"}:
            raise ValueError("method must be 'pca' or 'twonn'")
        if not 0.0 < pca_variance_threshold < 1.0:
            raise ValueError("pca_variance_threshold must be in (0, 1)")
        if twonn_k < 2:
            raise ValueError("twonn_k must be at least 2")

        self.method = method
        self.pca_variance_threshold = pca_variance_threshold
        self.twonn_k = twonn_k

    def _pca_id(self, features: torch.Tensor) -> torch.Tensor:
        """Intrinsic dimension via cumulative explained variance."""
        n, d = features.shape
        if n <= 1:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        # Center and compute covariance.
        centered = features - features.mean(dim=0, keepdim=True)
        # torch.linalg.svd is more stable than eig on covariance.
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        variance = (s ** 2) / max(1, n - 1)
        total = variance.sum()
        if total <= 0:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        cumulative = variance.cumsum(dim=0) / total
        threshold = self.pca_variance_threshold
        id_est = (cumulative < threshold).sum().float() + 1.0
        return id_est

    def _twonn_id(self, features: torch.Tensor) -> torch.Tensor:
        """Intrinsic dimension via the TwoNN estimator (Facco et al. 2017)."""
        n, d = features.shape
        if n <= self.twonn_k:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        # Pairwise Euclidean distances.
        diff = features.unsqueeze(1) - features.unsqueeze(0)  # (n, n, d)
        dist = torch.linalg.norm(diff, dim=-1)  # (n, n)
        dist.fill_diagonal_(float("inf"))

        # Distances to the first and second nearest neighbors.
        sorted_dist, _ = torch.topk(
            dist, k=min(self.twonn_k, n - 1), largest=False, dim=-1
        )
        r1 = sorted_dist[:, 0].clamp_min(1e-12)
        r2 = sorted_dist[:, 1].clamp_min(1e-12)

        mu = r2 / r1
        # MLE for a D-dimensional uniform density: D = N / sum_i log(mu_i).
        log_mu = torch.log(mu.clamp_min(1e-12))
        id_est = 1.0 / log_mu.mean()
        return id_est.clamp_min(1.0)

    def estimate_id(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate the intrinsic dimension of a point cloud.

        Args:
            features: tensor of shape ``(N, D)``.

        Returns:
            Scalar intrinsic-dimension estimate.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        if features.shape[0] <= 1:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        if self.method == "pca":
            return self._pca_id(features)
        return self._twonn_id(features)

    def local_geometry(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        center: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute local geometry statistics for a prototype neighborhood.

        Args:
            features: tensor of shape ``(N, D)``.
            labels: optional bool/0-1 tensor of shape ``(N,)`` where ``1``
                indicates a true positive (close to the prototype) and ``0``
                indicates a false positive (far from it). If provided, a
                separability score is returned.
            center: optional prototype center of shape ``(D,)``. Defaults to
                the empirical mean of ``features``.

        Returns:
            Dictionary with keys:
                - ``intrinsic_dim``: estimated ID.
                - ``radius``: mean distance to the center.
                - ``separability``: TP/FP separability score if ``labels`` is
                  provided, otherwise ``NaN``.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")

        if center is None:
            center = features.mean(dim=0)

        distances = torch.linalg.norm(features - center.unsqueeze(0), dim=-1)
        radius = distances.mean()

        id_est = self.estimate_id(features)

        result = {
            "intrinsic_dim": id_est,
            "radius": radius,
        }

        if labels is not None:
            if labels.shape[0] != features.shape[0]:
                raise ValueError("labels must have the same length as features")
            labels = labels.to(dtype=torch.bool, device=features.device)
            pos_mask = labels
            neg_mask = ~labels
            pos_count = int(pos_mask.sum().item())
            neg_count = int(neg_mask.sum().item())

            if pos_count > 0 and neg_count > 0:
                # Score: lower distance should correlate with positive label.
                # Use negative distance as score and compute a simple AUC
                # equivalent for two groups (Mann-Whitney U).
                pos_dists = distances[pos_mask]
                neg_dists = distances[neg_mask]
                # AUC = P(pos_dist < neg_dist)
                auc = self._auc_from_sorted_distances(pos_dists, neg_dists)
                result["separability"] = auc
            else:
                result["separability"] = torch.tensor(
                    float("nan"),
                    device=features.device,
                    dtype=features.dtype,
                )
        else:
            result["separability"] = torch.tensor(
                float("nan"),
                device=features.device,
                dtype=features.dtype,
            )

        return result

    @staticmethod
    def _auc_from_sorted_distances(
        pos_dists: torch.Tensor, neg_dists: torch.Tensor
    ) -> torch.Tensor:
        """Compute AUC for the binary score ``-distance``.

        Equivalent to the Mann-Whitney U statistic scaled to [0, 1].
        Concordant pairs satisfy ``pos_dist < neg_dist``.
        """
        pos = pos_dists.sort().values
        neg = neg_dists.sort().values
        n_pos = pos.numel()
        n_neg = neg.numel()

        j = 0  # number of negatives with distance <= current positive distance
        concordant = 0.0
        ties = 0.0
        for d in pos:
            # Count negatives strictly smaller (incorrectly ranked above pos).
            smaller = j
            while j < n_neg and neg[j] <= d:
                j += 1
            # Negatives equal to d are ties.
            equal = j - smaller
            # Negatives strictly larger are correctly ranked below pos.
            concordant += n_neg - j
            ties += equal

        auc = (concordant + 0.5 * ties) / (n_pos * n_neg)
        return torch.tensor(auc, device=pos_dists.device, dtype=pos_dists.dtype)

    def extra_repr(self) -> str:
        return (
            f"method={self.method}, "
            f"pca_variance_threshold={self.pca_variance_threshold}, "
            f"twonn_k={self.twonn_k}"
        )
