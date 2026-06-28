from __future__ import annotations

import torch
import torch.nn.functional as F


def crop_and_resize_roi(image: torch.Tensor, box: torch.Tensor, size: int = 128) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W].")
    _, height, width = image.shape
    x1, y1, x2, y2 = box.detach().float().tolist()
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    crop = image[:, top:bottom, left:right]
    return F.interpolate(crop.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
