from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of parameters (including non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def count_flops(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 800, 1333),
    device: torch.device | str = "cpu",
) -> int:
    """Estimate FLOPs for one forward pass using PyTorch FlopCounterMode.

    The model is put in train mode and called with a dummy input plus dummy
    targets so that detection losses are computed, matching the training graph.
    """
    model.eval()
    device = torch.device(device)
    dummy_images = torch.randn(input_size, device=device)
    # Detection models expect a list of images and a list of targets during training.
    image_list = [img for img in dummy_images]
    num_images = len(image_list)
    # Minimal dummy targets: one box per image.
    dummy_targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=device),
            "labels": torch.tensor([1], dtype=torch.long, device=device),
        }
        for _ in range(num_images)
    ]

    try:
        with FlopCounterMode(model, display=False) as fcm:
            _ = model(image_list, dummy_targets)
        total_flops = fcm.get_total_flops()
        return int(total_flops) if total_flops is not None else -1
    except Exception:
        return -1
