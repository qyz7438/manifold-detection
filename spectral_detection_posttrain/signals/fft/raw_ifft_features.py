from __future__ import annotations

import torch
import torch.nn.functional as F


def crop_and_resize_boxes(image: torch.Tensor, boxes: torch.Tensor, *, crop_size: int = 64) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape (C, H, W)")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    crops = []
    _, height, width = image.shape
    for box in boxes.detach().float().cpu():
        x1 = max(0, min(width - 1, int(torch.floor(box[0]).item())))
        y1 = max(0, min(height - 1, int(torch.floor(box[1]).item())))
        x2 = max(x1 + 1, min(width, int(torch.ceil(box[2]).item())))
        y2 = max(y1 + 1, min(height, int(torch.ceil(box[3]).item())))
        crop = image[:, y1:y2, x1:x2]
        crops.append(F.interpolate(crop.unsqueeze(0).float(), size=(crop_size, crop_size), mode="bilinear", align_corners=False).squeeze(0))
    if not crops:
        return image.new_empty((0, image.shape[0], int(crop_size), int(crop_size)))
    return torch.stack(crops, dim=0).to(image.device)


def sobel_edge_strength(crops: torch.Tensor) -> torch.Tensor:
    gray = crops.mean(dim=1, keepdim=True)
    sobel_x = crops.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = crops.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8).flatten(1).mean(dim=1)


def _frequency_radius(crops: torch.Tensor) -> torch.Tensor:
    freq_h = torch.fft.fftfreq(crops.shape[-2], device=crops.device)
    freq_w = torch.fft.rfftfreq(crops.shape[-1], device=crops.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x.pow(2) + grid_y.pow(2))
    return radius / radius.max().clamp_min(1e-6)


def _irfft_with_mask(crops: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")
    return torch.fft.irfft2(spectrum * mask, s=crops.shape[-2:], dim=(-2, -1), norm="ortho").real


def _phase_only_reconstruction(crops: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")
    phase_only = torch.exp(1j * torch.angle(spectrum))
    return torch.fft.irfft2(phase_only, s=crops.shape[-2:], dim=(-2, -1), norm="ortho").real


def raw_ifft_feature_summary(crops: torch.Tensor) -> torch.Tensor:
    if crops.ndim != 4:
        raise ValueError("crops must have shape (N, C, H, W)")
    if crops.numel() == 0:
        return crops.new_empty((crops.shape[0], 12))

    radius = _frequency_radius(crops)
    low_mask = (radius <= 0.3).to(crops.dtype).view(1, 1, *radius.shape)
    mid_mask = ((radius > 0.3) & (radius <= 0.7)).to(crops.dtype).view(1, 1, *radius.shape)
    high_mask = (radius > 0.7).to(crops.dtype).view(1, 1, *radius.shape)

    low_recon = _irfft_with_mask(crops, low_mask)
    high_recon = _irfft_with_mask(crops, high_mask)
    phase_recon = _phase_only_reconstruction(crops)

    spectrum = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")
    amp = torch.abs(spectrum)
    low_energy = (amp * low_mask).flatten(1).sum(dim=1)
    mid_energy = (amp * mid_mask).flatten(1).sum(dim=1)
    high_energy = (amp * high_mask).flatten(1).sum(dim=1)
    total_energy = (low_energy + mid_energy + high_energy).clamp_min(1e-8)

    raw_edge = sobel_edge_strength(crops)
    low_edge = sobel_edge_strength(low_recon)
    high_edge = sobel_edge_strength(high_recon)
    phase_edge = sobel_edge_strength(phase_recon)
    high_minus_low = high_edge - low_edge
    phase_diff = (phase_recon - crops).abs().flatten(1).mean(dim=1)
    high_diff = (high_recon - crops).abs().flatten(1).mean(dim=1)
    low_diff = (low_recon - crops).abs().flatten(1).mean(dim=1)

    return torch.stack(
        [
            raw_edge,
            low_edge,
            high_edge,
            phase_edge,
            high_minus_low,
            phase_diff,
            high_diff,
            low_diff,
            low_energy / total_energy,
            mid_energy / total_energy,
            high_energy / total_energy,
            high_energy / low_energy.clamp_min(1e-8),
        ],
        dim=1,
    )


def penn_fudan_legacy_ifft_metric_bank(crops: torch.Tensor) -> torch.Tensor:
    if crops.ndim != 4:
        raise ValueError("crops must have shape (N, C, H, W)")
    if crops.numel() == 0:
        return crops.new_empty((crops.shape[0], 23))

    radius = _frequency_radius(crops)
    low_mask = (radius <= 0.3).to(crops.dtype).view(1, 1, *radius.shape)
    mid_mask = ((radius > 0.3) & (radius <= 0.7)).to(crops.dtype).view(1, 1, *radius.shape)
    high_mask = (radius > 0.7).to(crops.dtype).view(1, 1, *radius.shape)
    hp015_mask = (radius > 0.15).to(crops.dtype).view(1, 1, *radius.shape)

    spectrum = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")
    amp = torch.abs(spectrum)
    phase = torch.angle(spectrum)
    low_energy = (amp * low_mask).flatten(2).sum(dim=2)
    mid_energy = (amp * mid_mask).flatten(2).sum(dim=2)
    high_energy = (amp * high_mask).flatten(2).sum(dim=2)
    total_energy = (low_energy + mid_energy + high_energy).clamp_min(1e-8)

    def mean_channel(values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=1)

    def phase_abs(mask: torch.Tensor) -> torch.Tensor:
        numerator = (phase.abs() * amp * mask).flatten(2).sum(dim=2)
        denominator = (amp * mask).flatten(2).sum(dim=2).clamp_min(1e-8)
        return mean_channel(numerator / denominator)

    gray = crops.mean(dim=1, keepdim=True)
    raw_edge = sobel_edge_strength(crops)
    phase_recon = _phase_only_reconstruction(crops)
    phase_edge = sobel_edge_strength(phase_recon)
    hp015_recon = _irfft_with_mask(crops, hp015_mask)
    hp015_edge = sobel_edge_strength(hp015_recon)
    low_recon = _irfft_with_mask(crops, low_mask)
    high_recon = _irfft_with_mask(crops, high_mask)
    low_edge = sobel_edge_strength(low_recon)
    high_edge = sobel_edge_strength(high_recon)

    grad_x = gray[..., :, 1:] - gray[..., :, :-1]
    grad_y = gray[..., 1:, :] - gray[..., :-1, :]
    edge = torch.sqrt(grad_x[..., :-1, :].pow(2) + grad_y[..., :, :-1].pow(2) + 1e-8).squeeze(1)
    total_edge = edge.flatten(1).sum(dim=1).clamp_min(1e-8)
    boundary_width = max(1, min(3, edge.shape[-1] // 4, edge.shape[-2] // 4))
    boundary_mask = torch.zeros(edge.shape[-2:], dtype=torch.bool, device=edge.device)
    boundary_mask[:boundary_width, :] = True
    boundary_mask[-boundary_width:, :] = True
    boundary_mask[:, :boundary_width] = True
    boundary_mask[:, -boundary_width:] = True
    boundary_edge = (edge * boundary_mask.float()).flatten(1).sum(dim=1)
    fft_edge_truncation = (1.0 - boundary_edge / total_edge).clamp(0.0, 1.0)

    flat = crops.flatten(2).clamp_min(1e-8)
    entropy = -(flat * flat.log()).sum(dim=(1, 2)) / max(crops.shape[1], 1)
    center = crops[..., crops.shape[-2] // 3 : 2 * crops.shape[-2] // 3, crops.shape[-1] // 3 : 2 * crops.shape[-1] // 3]
    center_surround = center.flatten(2).mean(dim=2).mean(dim=1) - crops.flatten(2).mean(dim=2).mean(dim=1)
    lap_kernel = crops.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, lap_kernel, padding=1).flatten(2).var(dim=2).squeeze(1)
    power = torch.abs(torch.fft.rfft2(gray, dim=(-2, -1), norm="ortho")).pow(2)
    autocorr = torch.fft.irfft2(power, dim=(-2, -1), norm="ortho", s=gray.shape[-2:]).squeeze(1)
    autocorr_peak = autocorr[:, autocorr.shape[-2] // 2, autocorr.shape[-1] // 2] / autocorr.flatten(1).mean(dim=1).clamp_min(1e-8)

    phase_std = phase.flatten(2).std(dim=2).mean(dim=1)
    low_ratio = mean_channel(low_energy / total_energy)
    mid_ratio = mean_channel(mid_energy / total_energy)
    high_ratio = mean_channel(high_energy / total_energy)

    return torch.stack(
        [
            raw_edge,
            phase_edge,
            hp015_edge,
            fft_edge_truncation,
            low_edge,
            high_edge,
            high_edge - low_edge,
            low_ratio,
            mid_ratio,
            high_ratio,
            high_ratio / low_ratio.clamp_min(1e-8),
            phase_abs(low_mask),
            phase_abs(mid_mask),
            phase_abs(high_mask),
            -(phase_abs(low_mask)),
            -(phase_abs(high_mask)),
            low_ratio * (-(phase_abs(high_mask))),
            entropy,
            center_surround,
            laplacian,
            autocorr_peak,
            phase_std,
            (phase_recon - crops).abs().flatten(1).mean(dim=1),
        ],
        dim=1,
    )
