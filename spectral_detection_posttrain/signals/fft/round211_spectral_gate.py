from __future__ import annotations

import torch
import torch.nn.functional as F


def radial_amplitude_profile(roi: torch.Tensor, bins: int = 16) -> torch.Tensor:
    if roi.ndim == 3:
        roi = roi.float().mean(dim=0)
    fft = torch.fft.fftshift(torch.fft.fft2(roi, norm="ortho"))
    amp = torch.log1p(torch.abs(fft))
    h, w = amp.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=amp.device), torch.arange(w, device=amp.device), indexing="ij")
    radius = torch.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    radius = radius / radius.max().clamp_min(1e-6)
    values = []
    for idx in range(bins):
        mask = (radius >= idx / bins) & (radius < (idx + 1) / bins)
        values.append(amp[mask].mean() if mask.any() else amp.new_tensor(0.0))
    return torch.stack(values)


def spectral_gate_score(pred_roi: torch.Tensor, gt_roi: torch.Tensor) -> torch.Tensor:
    pred = radial_amplitude_profile(pred_roi)
    gt = radial_amplitude_profile(gt_roi)
    return F.cosine_similarity(pred.flatten(), gt.flatten(), dim=0).clamp(-1.0, 1.0)


def shuffled_scores(scores: torch.Tensor) -> torch.Tensor:
    if len(scores) <= 1:
        return scores
    return scores[torch.arange(len(scores), device=scores.device).roll(1)]
