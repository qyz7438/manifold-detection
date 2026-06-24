r"""DPatch / RP2-style adversarial patch attack for object detectors.

Reference implementations and papers:
- DPatch: Liu et al., "DPATCH: An Adversarial Patch Attack on Object Detectors", 2018.
- RP2: Eykholt et al., "Robust Physical-World Attacks on Deep Learning Models", 2018.
- EOT: Athalye et al., "Synthesizing Robust Adversarial Examples", 2018.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.transforms.functional as tv_f


class ObjectDetectorPatchAttack:
    r"""Optimise a local patch to suppress detections of a target class.

    This attack follows the DPatch / RP2 disappearance attack recipe:
    place a patch on/around the target object and minimise the objectness
    and classification scores assigned to that object.  A total-variation
    penalty is added to keep the patch smooth.

    Args:
        model: a PyTorch object detector in eval mode (e.g. Faster R-CNN).
        device: torch device used for inference.
        target_label: class label to suppress.
        patch_size: tuple ``(height, width)``.
        max_iter: number of PGD steps.
        step_size: patch update step size in :math:`[0, 1]` pixel units.
        momentum: momentum factor for the PGD velocity term.
        tv_weight: weight of the total-variation smoothness penalty.
        eot_transforms: number of Expectation-over-Transformations samples
            per optimisation step.  ``1`` disables EOT.
        clamp_range: valid pixel range.
        score_threshold: predictions below this threshold are ignored by the
            attack loss (they are already suppressed).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        target_label: int = 1,
        patch_size: tuple[int, int] = (80, 80),
        max_iter: int = 300,
        step_size: float = 0.5,
        momentum: float = 0.9,
        tv_weight: float = 0.01,
        eot_transforms: int = 1,
        clamp_range: tuple[float, float] = (0.0, 1.0),
        score_threshold: float = 0.05,
    ):
        self.model = model
        self.device = device
        self.target_label = target_label
        self.patch_size = tuple(patch_size)
        self.max_iter = max_iter
        self.step_size = step_size
        self.momentum = momentum
        self.tv_weight = tv_weight
        self.eot_transforms = max(1, eot_transforms)
        self.clamp_min, self.clamp_max = clamp_range
        self.score_threshold = score_threshold
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Patch placement helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_patch_location(
        image_h: int,
        image_w: int,
        patch_h: int,
        patch_w: int,
        target_boxes: torch.Tensor | None = None,
    ) -> tuple[int, int]:
        """Return (top, left) placing patch at GT bbox centre if available."""
        if target_boxes is not None and len(target_boxes) > 0:
            # Use the largest box by area.
            boxes = target_boxes.float()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            biggest = boxes[areas.argmax()]
            cx = (biggest[0] + biggest[2]) / 2.0
            cy = (biggest[1] + biggest[3]) / 2.0
            top = int(cy - patch_h / 2.0)
            left = int(cx - patch_w / 2.0)
        else:
            top = (image_h - patch_h) // 2
            left = (image_w - patch_w) // 2

        # Clamp to valid region.
        top = max(0, min(top, image_h - patch_h))
        left = max(0, min(left, image_w - patch_w))
        return top, left

    # ------------------------------------------------------------------ #
    # Loss helpers
    # ------------------------------------------------------------------ #

    def detection_loss(self, image: torch.Tensor) -> torch.Tensor:
        """Loss that is lower when target-class detections are suppressed."""
        with torch.set_grad_enabled(True):
            output = self.model([image])[0]
        scores = output.get("scores", torch.empty(0, device=self.device))
        labels = output.get("labels", torch.empty(0, device=self.device, dtype=torch.long))

        mask = labels == self.target_label
        person_scores = scores[mask]

        if person_scores.numel() == 0:
            # Already suppressed; encourage patch to remain small/natural.
            return image.sum() * 1e-4

        # Suppress the highest-confidence detection strongly.
        max_score_loss = person_scores.max()
        # Also push all detections below the evaluation threshold.
        threshold_penalty = (person_scores - self.score_threshold).clamp(min=0.0).sum()
        return max_score_loss + 2.0 * threshold_penalty

    @staticmethod
    def total_variation_loss(patch: torch.Tensor) -> torch.Tensor:
        """Isotropic total variation on the patch."""
        diff_h = (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()
        diff_w = (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean()
        return diff_h + diff_w

    # ------------------------------------------------------------------ #
    # EOT transform
    # ------------------------------------------------------------------ #

    def _eot_augment(self, patch: torch.Tensor) -> torch.Tensor:
        """Apply a random scale/rotation to the patch for EOT robustness."""
        c, ph, pw = patch.shape
        # Random scale in [0.8, 1.2].
        scale = 0.8 + 0.4 * torch.rand(1).item()
        new_h = max(3, int(ph * scale))
        new_w = max(3, int(pw * scale))
        scaled = tv_f.resize(patch, (new_h, new_w), antialias=True)
        # Random rotation in [-15, 15] degrees.
        angle = 30.0 * torch.rand(1).item() - 15.0
        rotated = tv_f.rotate(scaled, angle, interpolation=tv_f.InterpolationMode.BILINEAR)
        return rotated

    # ------------------------------------------------------------------ #
    # Attack
    # ------------------------------------------------------------------ #

    def attack(
        self,
        image: torch.Tensor,
        target_boxes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Generate an adversarially patched image.

        Args:
            image: input image of shape ``(C, H, W)``.
            target_boxes: ground-truth boxes of shape ``(N, 4)`` in
                ``(x1, y1, x2, y2)`` format.  Used to centre the patch.

        Returns:
            ``(patched_image, patch)``.
        """
        if image.dim() != 3:
            raise ValueError("attack expects a single image of shape (C, H, W)")
        c, h, w = image.shape
        ph, pw = self.patch_size
        if ph > h or pw > w:
            raise ValueError(f"patch size {self.patch_size} exceeds image size ({h}, {w})")

        top, left = self.compute_patch_location(h, w, ph, pw, target_boxes)

        # Initialise patch from the underlying region for realism.
        patch = (
            image[:, top : top + ph, left : left + pw]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        velocity = torch.zeros_like(patch.data)

        for _ in range(self.max_iter):
            if patch.grad is not None:
                patch.grad.zero_()

            # Expectation over transformations.
            det_losses = []
            for _eot in range(self.eot_transforms):
                if self.eot_transforms > 1:
                    aug_patch = self._eot_augment(patch)
                    # Centre the augmented patch on the original location.
                    aph, apw = aug_patch.shape[-2:]
                    etop = max(0, min(top + (ph - aph) // 2, h - aph))
                    eleft = max(0, min(left + (pw - apw) // 2, w - apw))
                else:
                    aug_patch = patch
                    etop, eleft = top, left

                patched = image.clone()
                patched[:, etop : etop + aug_patch.shape[-2], eleft : eleft + aug_patch.shape[-1]] = aug_patch
                det_losses.append(self.detection_loss(patched.to(self.device)))

            loss = torch.stack(det_losses).mean()
            if self.tv_weight > 0.0:
                loss = loss + self.tv_weight * self.total_variation_loss(patch)

            loss.backward()

            if patch.grad is None:
                break

            with torch.no_grad():
                grad = patch.grad
                if self.momentum > 0.0:
                    velocity.mul_(self.momentum).add_((1.0 - self.momentum) * grad)
                    update = velocity
                else:
                    update = grad
                patch.add_(-self.step_size * update.sign())
                patch.clamp_(self.clamp_min, self.clamp_max)
            patch = patch.detach().requires_grad_(True)

        final_patch = patch.detach()
        patched_image = image.clone()
        patched_image[:, top : top + ph, left : left + pw] = final_patch
        return patched_image.clamp(self.clamp_min, self.clamp_max), final_patch
