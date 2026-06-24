r"""Simplified digital adversarial patch attack for object detection.

This module implements a lightweight RP2-style patch attack that optimises a
small patch in the digital domain.  It is intentionally simplified: physical
constraints (printability, camera transformations) are left for future work.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdversarialPatchAttack:
    r"""Optimise a local patch to degrade detector predictions.

    The attack iteratively updates the patch pixels with projected gradient
    descent.  By default the attack is untargeted: it minimises the score
    returned by ``model_or_loss``.

    Args:
        model_or_loss: callable that receives an image tensor of shape
            ``(C, H, W)`` and returns a scalar loss.  Lower loss should
            correspond to worse detection performance for an untargeted
            attack.
        patch_size: tuple ``(height, width)`` of the patch.
        location: optional top-left corner ``(top, left)``.  If ``None``, a
            random valid location is chosen for each ``attack`` call.
        max_iter: number of PGD iterations.
        step_size: patch update step size in :math:`[0, 1]` pixel units.
        clamp_range: valid pixel range ``(min, max)``.
        targeted: if ``True``, maximise ``model_or_loss`` instead of
            minimising it.
        smooth_sigma: if ``> 0``, apply a small Gaussian blur to the patch
            after each update to suppress high-frequency noise.
    """

    def __init__(
        self,
        model_or_loss: callable,
        patch_size: tuple[int, int],
        location: tuple[int, int] | None = None,
        max_iter: int = 100,
        step_size: float = 0.05,
        clamp_range: tuple[float, float] = (0.0, 1.0),
        targeted: bool = False,
        smooth_sigma: float = 0.0,
        momentum: float = 0.0,
        random_init: bool = False,
    ):
        if not callable(model_or_loss):
            raise TypeError("model_or_loss must be callable")
        if patch_size[0] <= 0 or patch_size[1] <= 0:
            raise ValueError("patch_size must be positive")
        self.model_or_loss = model_or_loss
        self.patch_size = tuple(patch_size)
        self.location = location
        self.max_iter = max_iter
        self.step_size = step_size
        self.clamp_min, self.clamp_max = clamp_range
        self.targeted = targeted
        self.smooth_sigma = smooth_sigma
        self.momentum = momentum
        self.random_init = random_init

        if smooth_sigma > 0.0:
            self._blur_kernel = self._build_blur_kernel(smooth_sigma)
        else:
            self._blur_kernel = None

    @staticmethod
    def _build_blur_kernel(sigma: float) -> torch.Tensor:
        """Build a small separable Gaussian kernel for patch smoothing."""
        size = max(3, int(4 * sigma + 1) | 1)  # odd integer >= 3
        half = size // 2
        x = torch.arange(size, dtype=torch.float32) - half
        kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel

    def _apply_patch(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        top: int,
        left: int,
    ) -> torch.Tensor:
        """Paste ``patch`` into a copy of ``image`` at (top, left)."""
        patched = image.clone()
        ph, pw = patch.shape[-2:]
        patched[..., top : top + ph, left : left + pw] = patch
        return patched

    def _smooth_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Apply a light Gaussian blur to the patch pixels."""
        if self._blur_kernel is None:
            return patch
        kernel = self._blur_kernel.to(patch.device)
        c = patch.shape[-3]
        # Expand kernel to (out_ch, in_ch, k) for depthwise separable blur.
        kernel_h = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)
        kernel_w = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
        pad = kernel.shape[0] // 2
        blurred = nn.functional.conv2d(
            nn.functional.pad(patch, (0, 0, pad, pad), mode="reflect"),
            kernel_h,
            groups=c,
        )
        blurred = nn.functional.conv2d(
            nn.functional.pad(blurred, (pad, pad, 0, 0), mode="reflect"),
            kernel_w,
            groups=c,
        )
        return blurred

    def attack(
        self,
        image: torch.Tensor,
        target_boxes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Generate an adversarially patched image.

        Args:
            image: input image of shape ``(C, H, W)``.
            target_boxes: unused in the simplified version; kept for API
                compatibility with future targeted attacks.

        Returns:
            ``(patched_image, patch)`` where ``patched_image`` has the same
            shape as ``image`` and ``patch`` has shape ``(C, ph, pw)``.
        """
        if image.dim() != 3:
            raise ValueError("attack expects a single image of shape (C, H, W)")
        c, h, w = image.shape
        ph, pw = self.patch_size
        if ph > h or pw > w:
            raise ValueError(f"patch size {self.patch_size} exceeds image size ({h}, {w})")

        if self.location is not None:
            top, left = self.location
        elif target_boxes is not None and len(target_boxes) > 0:
            # Place patch at the centre of the largest GT box (DPatch/RP2 style).
            boxes = target_boxes.float()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            biggest = boxes[areas.argmax()]
            cx = (biggest[0] + biggest[2]) / 2.0
            cy = (biggest[1] + biggest[3]) / 2.0
            top = int(cy - ph / 2.0)
            left = int(cx - pw / 2.0)
            top = max(0, min(top, h - ph))
            left = max(0, min(left, w - pw))
        else:
            top = torch.randint(0, h - ph + 1, (1,)).item()
            left = torch.randint(0, w - pw + 1, (1,)).item()

        if top < 0 or left < 0 or top + ph > h or left + pw > w:
            raise ValueError(f"patch location ({top}, {left}) is out of bounds")

        # Initialise patch from the underlying image region for realism, or
        # from uniform random noise when random_init is requested.
        if self.random_init:
            patch = torch.rand(
                (c, ph, pw), dtype=image.dtype, device=image.device
            ).detach().requires_grad_(True)
        else:
            patch = image[:, top : top + ph, left : left + pw].clone().detach().requires_grad_(True)

        velocity = torch.zeros_like(patch.data)
        sign_scale = 1.0 if self.targeted else -1.0

        for _ in range(self.max_iter):
            if patch.grad is not None:
                patch.grad.zero_()

            patched = self._apply_patch(image, patch, top, left)
            loss = self.model_or_loss(patched)
            loss.backward()

            if patch.grad is None:
                break

            with torch.no_grad():
                # Momentum-accelerated PGD update.
                grad = patch.grad
                if self.momentum > 0.0:
                    velocity.mul_(self.momentum).add_((1.0 - self.momentum) * grad)
                    update = velocity
                else:
                    update = grad
                patch.add_(sign_scale * self.step_size * update.sign())
                patch.clamp_(self.clamp_min, self.clamp_max)
                if self._blur_kernel is not None:
                    patch.copy_(self._smooth_patch(patch.unsqueeze(0)).squeeze(0))
                    patch.clamp_(self.clamp_min, self.clamp_max)
            patch = patch.detach().requires_grad_(True)

        final_patch = patch.detach()
        patched_image = self._apply_patch(image, final_patch, top, left)
        return patched_image, final_patch
