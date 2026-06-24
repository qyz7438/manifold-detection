from __future__ import annotations

import torch

from .fourier import high_freq_perturb, low_pass_filter


def apply_fourier_training_aug(images: torch.Tensor, config: dict) -> torch.Tensor:
    perturb_cfg = config["perturb"]
    choice = torch.randint(0, 3, (1,), device=images.device).item()
    if choice == 0:
        return images
    if choice == 1:
        return low_pass_filter(images, ratio=float(perturb_cfg.get("low_ratio", 0.25)))
    return high_freq_perturb(
        images,
        strength=float(perturb_cfg.get("high_strength", 0.10)),
        ratio=float(perturb_cfg.get("high_ratio", 0.25)),
    )
