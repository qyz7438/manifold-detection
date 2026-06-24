from __future__ import annotations

import torch


def _make_patch(channels: int, size: int, patch_type: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if patch_type == "random":
        return torch.rand((channels, size, size), device=device, dtype=dtype)
    if patch_type == "checkerboard":
        yy, xx = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
        return ((yy + xx) % 2).to(dtype=dtype).view(1, size, size).repeat(channels, 1, 1)
    if patch_type in {"qr", "qr_like", "qr-like"}:
        block = max(2, size // 8)
        grid = max(1, size // block)
        coarse = torch.randint(0, 2, (1, grid, grid), device=device, dtype=dtype)
        patch = coarse.repeat_interleave(block, 1).repeat_interleave(block, 2)
        return patch[:, :size, :size].repeat(channels, 1, 1)
    raise ValueError(f"Unknown patch_type: {patch_type}")


_PLACEMENT_ALIASES = {
    "object_inside": "object",
    "object_edge": "edge",
}


def _location_for_box(box: torch.Tensor, height: int, width: int, size: int, placement: str) -> tuple[int, int]:
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    if placement in ("object", "object_inside"):
        cx = max(0, min(width - size, (x1 + x2 - size) // 2))
        cy = max(0, min(height - size, (y1 + y2 - size) // 2))
        return cy, cx
    if placement in ("edge", "object_edge"):
        cx = max(0, min(width - size, x2 - size // 2))
        cy = max(0, min(height - size, y2 - size // 2))
        return cy, cx
    if placement == "near_object":
        cx = max(0, min(width - size, x2 + size // 2))
        cy = max(0, min(height - size, y1 - size))
        return cy, cx
    raise ValueError(f"Unsupported box placement: {placement}")


def add_detection_patch(
    image: torch.Tensor,
    target: dict,
    placement: str = "random",
    patch_type: str = "random",
    patch_size: int = 48,
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W].")
    channels, height, width = image.shape
    size = min(patch_size, height - 1, width - 1)
    if size <= 0:
        return image.clone()
    out = image.clone()
    resolved = _PLACEMENT_ALIASES.get(placement, placement)
    box_placements = {"object", "edge", "object_inside", "object_edge", "near_object"}
    if resolved in box_placements and len(target.get("boxes", [])) > 0:
        top, left = _location_for_box(target["boxes"][0], height, width, size, resolved)
    else:
        top = int(torch.randint(0, height - size + 1, (1,)).item())
        left = int(torch.randint(0, width - size + 1, (1,)).item())
    patch = _make_patch(channels, size, patch_type, image.device, image.dtype)
    out[:, top : top + size, left : left + size] = patch
    return out
