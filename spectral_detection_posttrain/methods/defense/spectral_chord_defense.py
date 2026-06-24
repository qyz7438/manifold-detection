r"""Spectral Chord defense: project adversarial frequencies back onto a natural manifold.

The defense detects anomalous frequency bins in the rFFT domain and replaces
them with values obtained by Chord-transporting the spectrum towards a natural
image prototype.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.chord_transport import (
    ChordTransport,
)
from spectral_detection_posttrain.methods.defense.manifold_natural import (
    NaturalSpectrumModel,
)


class SpectralChordDefense(nn.Module):
    r"""Defense-as-preprocessing using a learned spectral manifold.

    For an adversarial image :math:`x_{\mathrm{adv}}` the module computes

    .. math::
        F = \mathcal{F}(x_{\mathrm{adv}})
        m = \mathrm{detect\_anomaly}(|F|)
        F_{\mathrm{clean}} = \mathrm{ChordTransport}(F, F_{\mathrm{natural}})
        F_{\mathrm{out}} = (1 - m) \, F + m \, F_{\mathrm{clean}}
        x_{\mathrm{clean}} = \mathcal{F}^{-1}(F_{\mathrm{out}})

    The manifold and transport operate on vectorised per-channel spectra.

    Args:
        manifold: learned complex spectral manifold.
        transport: Chord transport module.
        natural_model: fitted natural spectrum prototype model.
        anomaly_gate_threshold: z-score threshold for anomaly detection.
        window_size: size of the local window used to compute mean/std.
        preserve_dc: if ``True``, never modify the DC frequency bin.
    """

    def __init__(
        self,
        manifold: ComplexSpectralManifold,
        transport: ChordTransport,
        natural_model: NaturalSpectrumModel,
        anomaly_gate_threshold: float = 3.0,
        window_size: int = 5,
        preserve_dc: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.transport = transport
        self.natural_model = natural_model
        self.anomaly_gate_threshold = anomaly_gate_threshold
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.window_size = window_size
        self.preserve_dc = preserve_dc
        self._eps = 1e-8

    def _flatten_spectrum(self, F: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten spatial frequency dimensions to vector form.

        Args:
            F: complex tensor of shape ``(B, C, H, Wf)`` or ``(C, H, Wf)``.

        Returns:
            ``(F_vec, spatial_shape)`` where ``F_vec`` has shape
            ``(B*C, H*Wf)`` (or ``(C, H*Wf)`` for unbatched input).
        """
        spatial_shape = F.shape[-2:]
        if F.dim() == 3:
            return F.reshape(F.shape[0], -1), spatial_shape
        if F.dim() == 4:
            b, c, h, wf = F.shape
            return F.reshape(b * c, h * wf), spatial_shape
        raise ValueError(f"Expected 3-D or 4-D spectrum, got {F.dim()}-D")

    def _unflatten_spectrum(
        self,
        F_vec: torch.Tensor,
        spatial_shape: tuple[int, ...],
        batch_size: int | None,
        channels: int | None,
    ) -> torch.Tensor:
        """Reshape a vectorised spectrum back to its original shape."""
        h, wf = spatial_shape
        if batch_size is None:
            return F_vec.reshape(channels, h, wf)
        return F_vec.reshape(batch_size, channels, h, wf)

    def detect_anomaly(self, spectrum: torch.Tensor) -> torch.Tensor:
        r"""Detect anomalous frequency bins via local z-score.

        For each frequency bin the score is

        .. math::
            s(u,v) = \frac{|F|(u,v) - \mathrm{local\_mean}}{\mathrm{local\_std} + \epsilon}

        and the anomaly mask is ``s > threshold``.

        Args:
            spectrum: complex spectrum of shape ``(..., H, Wf)``.

        Returns:
            Real-valued binary mask of the same spatial shape as ``|spectrum|``.
        """
        mag = spectrum.abs()
        pad = self.window_size // 2
        # Pad with replicate so that boundaries are handled gracefully.
        mag_padded = F.pad(mag, (pad, pad, pad, pad), mode="reflect")
        # Unfold over the last two spatial dims.  The first unfold reduces the
        # height dimension; the second unfold reduces the width dimension.
        patches = mag_padded.unfold(-2, self.window_size, 1)
        patches = patches.unfold(-2, self.window_size, 1)
        # patches shape: (..., H, Wf, window, window)
        local_mean = patches.mean(dim=(-2, -1))
        local_std = patches.std(dim=(-2, -1), unbiased=False).clamp_min(self._eps)
        score = (mag - local_mean) / local_std
        mask = (score > self.anomaly_gate_threshold).to(mag.dtype)
        if self.preserve_dc:
            # DC bin is at spatial frequency (0, 0).
            mask[..., 0, 0] = 0.0
        return mask

    def forward(self, x_adv: torch.Tensor) -> torch.Tensor:
        r"""Purify an adversarial image.

        Args:
            x_adv: input image tensor of shape ``(B, C, H, W)`` or
                ``(C, H, W)``.  Values are assumed to be in :math:`[0, 1]`.

        Returns:
            Purified image with the same shape as ``x_adv``.
        """
        if x_adv.dim() not in (3, 4):
            raise ValueError(f"Expected 3-D or 4-D image, got {x_adv.dim()}-D")

        batched = x_adv.dim() == 4
        if not batched:
            x_adv = x_adv.unsqueeze(0)

        b, c, h, w = x_adv.shape
        # 1. DFT
        spectrum = torch.fft.rfft2(x_adv)
        # 2. Anomaly detection
        mask = self.detect_anomaly(spectrum)
        # 3. Vectorise for the manifold.
        spectrum_vec, spatial_shape = self._flatten_spectrum(spectrum)
        # Natural prototype matches per-channel spectrum shape.
        prototype = self.natural_model.prototype()
        if prototype.dim() == 2:
            # Single-channel prototype, broadcast to all channels.
            proto = prototype.unsqueeze(0).expand(c, -1)
        elif prototype.dim() == 3:
            # (C, H, Wf)
            if prototype.shape[0] != c:
                raise ValueError(
                    f"Natural prototype has {prototype.shape[0]} channels, "
                    f"image has {c} channels"
                )
            proto = prototype
        else:
            raise ValueError(f"Unexpected prototype shape {prototype.shape}")

        proto_vec, _ = self._flatten_spectrum(proto)
        # Broadcast prototype to the batch.
        proto_vec = proto_vec.unsqueeze(0).expand(b, -1, -1).reshape(b * c, -1)

        # 4. Chord transport towards the natural prototype.
        spectrum_clean_vec = self.transport(spectrum_vec, proto_vec)
        spectrum_clean = self._unflatten_spectrum(
            spectrum_clean_vec, spatial_shape, batch_size=b, channels=c
        )

        # 5. Blend: keep normal frequencies, replace anomalous ones.
        spectrum_out = (1.0 - mask) * spectrum + mask * spectrum_clean
        # 6. iDFT
        x_clean = torch.fft.irfft2(spectrum_out, s=(h, w))

        if not batched:
            x_clean = x_clean.squeeze(0)
        return x_clean
