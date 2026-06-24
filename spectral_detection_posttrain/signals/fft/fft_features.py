from __future__ import annotations

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.signals.fft.radial_profile import radial_profile


def _gray_with_optional_hann(roi: torch.Tensor, use_hann: bool = True) -> torch.Tensor:
    if roi.ndim != 3:
        raise ValueError("roi must have shape [C, H, W].")
    gray = roi.mean(dim=0)
    if use_hann:
        h, w = gray.shape
        window = torch.outer(torch.hann_window(h, device=roi.device), torch.hann_window(w, device=roi.device))
        gray = gray * window
    return gray


def compute_fft_amplitude(roi: torch.Tensor, use_hann: bool = True) -> torch.Tensor:
    gray = _gray_with_optional_hann(roi, use_hann=use_hann)
    fft = torch.fft.fft2(gray, dim=(-2, -1))
    amp = torch.fft.fftshift(torch.abs(fft))
    amp = torch.log1p(amp)
    return (amp - amp.min()) / (amp.max() - amp.min()).clamp(min=1e-6)


def compute_amplitude_profile(roi: torch.Tensor, num_bins: int = 32, use_hann: bool = True) -> torch.Tensor:
    return radial_profile(compute_fft_amplitude(roi, use_hann=use_hann), num_bins=num_bins)


def compute_lowfreq_phase_stats(roi: torch.Tensor, radius_ratio: float = 0.25, use_hann: bool = True) -> torch.Tensor:
    gray = _gray_with_optional_hann(roi, use_hann=use_hann)
    phase = torch.fft.fftshift(torch.angle(torch.fft.fft2(gray, dim=(-2, -1))))
    height, width = phase.shape
    y, x = torch.meshgrid(torch.arange(height, device=roi.device), torch.arange(width, device=roi.device), indexing="ij")
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    mask = radius <= radius.max().clamp(min=1e-6) * radius_ratio
    selected = phase[mask]
    if selected.numel() == 0:
        return torch.zeros((4,), dtype=roi.dtype, device=roi.device)
    cos_phase = selected.cos()
    sin_phase = selected.sin()
    return torch.stack(
        [
            cos_phase.mean(),
            cos_phase.std(unbiased=False),
            sin_phase.mean(),
            sin_phase.std(unbiased=False),
        ]
    )


def compute_sobel_structure_features(roi: torch.Tensor) -> torch.Tensor:
    gray = roi.mean(dim=0, keepdim=True).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=roi.dtype,
        device=roi.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=roi.dtype,
        device=roi.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(gray, sobel_y, padding=1).squeeze()
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    threshold = magnitude.mean() + magnitude.std(unbiased=False)
    phase_stats = compute_lowfreq_phase_stats(roi)
    return torch.stack(
        [
            magnitude.mean(),
            magnitude.std(unbiased=False),
            magnitude.max(),
            (magnitude > threshold).float().mean(),
            grad_x.abs().mean(),
            grad_y.abs().mean(),
            phase_stats[0],
            phase_stats[2],
        ]
    )


def _normalized_sobel_magnitude(roi: torch.Tensor) -> torch.Tensor:
    gray = roi.mean(dim=0, keepdim=True).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=roi.dtype, device=roi.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=roi.dtype, device=roi.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(gray, sobel_y, padding=1).squeeze()
    mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    return (mag - mag.min()) / (mag.max() - mag.min()).clamp(min=1e-6)


def phase_correlation_score(roi_pred: torch.Tensor, roi_gt: torch.Tensor, use_hann: bool = True) -> torch.Tensor:
    pred_gray = _gray_with_optional_hann(roi_pred, use_hann=use_hann)
    gt_gray = _gray_with_optional_hann(roi_gt, use_hann=use_hann)
    pred_fft = torch.fft.fft2(pred_gray, dim=(-2, -1))
    gt_fft = torch.fft.fft2(gt_gray, dim=(-2, -1))
    cross_power = pred_fft * torch.conj(gt_fft)
    cross_power = cross_power / cross_power.abs().clamp(min=1e-6)
    corr = torch.fft.ifft2(cross_power, dim=(-2, -1)).abs()
    return corr.max().clamp(0.0, 1.0)


def edge_similarity_score(roi_pred: torch.Tensor, roi_gt: torch.Tensor) -> torch.Tensor:
    pred_edge = _normalized_sobel_magnitude(roi_pred).flatten()
    gt_edge = _normalized_sobel_magnitude(roi_gt).flatten()
    cosine = F.cosine_similarity(pred_edge, gt_edge, dim=0).clamp(-1.0, 1.0)
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def lowfreq_phase_similarity(
    roi_pred: torch.Tensor, roi_gt: torch.Tensor,
    radius_ratio: float = 0.25, use_hann: bool = True,
) -> torch.Tensor:
    pred_stats = compute_lowfreq_phase_stats(roi_pred, radius_ratio=radius_ratio, use_hann=use_hann)
    gt_stats = compute_lowfreq_phase_stats(roi_gt, radius_ratio=radius_ratio, use_hann=use_hann)
    mse = torch.mean((pred_stats - gt_stats).square())
    return torch.exp(-mse).clamp(0.0, 1.0)


def compute_structure_similarity(
    roi_pred: torch.Tensor, roi_gt: torch.Tensor,
    phase_weight: float = 0.45, edge_weight: float = 0.35, lowfreq_weight: float = 0.20,
) -> torch.Tensor:
    phase = phase_correlation_score(roi_pred, roi_gt)
    edge = edge_similarity_score(roi_pred, roi_gt)
    lowfreq = lowfreq_phase_similarity(roi_pred, roi_gt)
    total_weight = max(phase_weight + edge_weight + lowfreq_weight, 1e-6)
    score = (phase_weight * phase + edge_weight * edge + lowfreq_weight * lowfreq) / total_weight
    return score.clamp(0.0, 1.0)
