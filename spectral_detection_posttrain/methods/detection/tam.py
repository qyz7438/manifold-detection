from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAlignedManifold(nn.Module):
    """Task-aligned residual manifold on box-head embeddings.

    The module can operate as a plain residual adapter (the default) or as a
    reward-guided latent manifold when the auxiliary heads are enabled:

    * ``use_spectral_quality=True`` adds a small MLP head that predicts a
      per-ROI spectral quality score from the latent code.  The target is
      derived from the FFT magnitude of the input ROI feature, so the latent
      space is explicitly shaped by frequency-domain structure.
    * ``use_contrastive=True`` maintains learnable class prototypes in latent
      space and applies a supervised contrastive / prototype loss when labels
      are supplied.  This pushes the latent representation to form class-aware
      clusters.

    Both auxiliary objectives are computed inside :meth:`forward` and exposed
    through :meth:`get_aux_loss` so the training loop can add them to the
    detection loss without changing the box-head signature.
    """

    def __init__(
        self,
        in_features: int,
        latent_dim: int = 256,
        use_spectral_quality: bool = False,
        use_contrastive: bool = False,
        num_classes: int = 2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_spectral_quality = use_spectral_quality
        self.use_contrastive = use_contrastive
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, in_features),
        )
        self.scale = nn.Parameter(torch.zeros(1))

        # Initialise decoder as zero so the module starts as an identity mapping.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

        if use_spectral_quality:
            self.quality_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
            )

        if use_contrastive:
            self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
            nn.init.xavier_uniform_(self.prototypes)

        self._latest_aux_loss = 0.0

    def _spectral_quality_target(self, roi_feature: torch.Tensor) -> torch.Tensor:
        """Compute a scalar spectral quality target per ROI.

        The target is the log of the mean FFT magnitude.  It is a coarse but
        cheap proxy for "how much structural frequency content" the ROI
        contains, which tends to correlate with objectness / boundary clarity.
        """
        fr = torch.fft.rfft2(roi_feature, norm="ortho")
        mag = torch.abs(fr).mean(dim=(1, 2, 3))
        return torch.log1p(mag)

    def _contrastive_loss(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Supervised prototype loss in latent space."""
        latent_norm = F.normalize(latent, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        logits = torch.matmul(latent_norm, proto_norm.t()) / self.temperature
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        z: torch.Tensor,
        roi_feature: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.encoder(z)
        residual = self.decoder(latent)
        z_out = z + self.scale * residual

        aux_loss = 0.0
        if self.use_spectral_quality and roi_feature is not None:
            pred_q = self.quality_head(latent).squeeze(-1)
            target_q = self._spectral_quality_target(roi_feature)
            aux_loss = aux_loss + F.mse_loss(pred_q, target_q)

        if self.use_contrastive and labels is not None:
            aux_loss = aux_loss + self._contrastive_loss(latent, labels)

        self._latest_aux_loss = aux_loss
        return z_out

    def get_aux_loss(self) -> torch.Tensor:
        """Return and reset the auxiliary loss accumulated in the last forward."""
        loss = self._latest_aux_loss
        self._latest_aux_loss = 0.0
        return loss
