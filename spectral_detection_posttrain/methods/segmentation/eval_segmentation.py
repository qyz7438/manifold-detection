r"""Semantic-segmentation evaluation utilities.

Provides mean IoU, per-class IoU, pixel accuracy, and boundary IoU implemented
in pure PyTorch so that the module has no extra dependencies beyond the
project's existing ones.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _filter_ignore_index(
    pred: torch.Tensor, target: torch.Tensor, ignore_index: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask out pixels that match ``ignore_index``.

    Args:
        pred: predictions, either ``(B, H, W)`` long or ``(B, C, H, W)`` logits.
        target: ground-truth labels ``(B, H, W)``.
        ignore_index: class index to ignore, or ``None``.

    Returns:
        ``(pred, target)`` with ignored pixels set to ``-1`` in both tensors.
    """
    if ignore_index is None:
        return pred, target
    mask = target != ignore_index
    if pred.ndim == target.ndim:
        pred = pred.where(mask, torch.tensor(-1, device=pred.device))
    target = target.where(mask, torch.tensor(-1, device=target.device))
    return pred, target


def _pred_to_label(pred: torch.Tensor) -> torch.Tensor:
    """Convert logits or probabilities to hard labels.

    Args:
        pred: ``(B, C, H, W)`` logits/probabilities or ``(B, H, W)`` labels.

    Returns:
        ``(B, H, W)`` long labels.
    """
    if pred.ndim == 3:
        return pred.long()
    return pred.argmax(dim=1).long()


def _confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute a batched confusion matrix.

    Args:
        pred: ``(B, H, W)`` long labels.
        target: ``(B, H, W)`` long labels.
        num_classes: number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)``.
    """
    valid = (pred >= 0) & (pred < num_classes) & (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]
    indices = pred * num_classes + target
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(torch.float32)


def per_class_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    r"""Per-class intersection-over-union.

    Args:
        pred: predictions of shape ``(B, H, W)`` or ``(B, C, H, W)``.
        target: ground-truth labels of shape ``(B, H, W)``.
        num_classes: number of classes.
        ignore_index: optional class index to ignore.

    Returns:
        Tensor of shape ``(num_classes,)`` containing IoU per class.  Classes
        that never appear in the target are set to NaN so that they do not
        contribute to the mean.
    """
    pred = _pred_to_label(pred)
    pred, target = _filter_ignore_index(pred, target, ignore_index)
    cm = _confusion_matrix(pred, target, num_classes)
    intersection = torch.diag(cm)
    union = cm.sum(dim=0) + cm.sum(dim=1) - intersection + 1e-12
    iou = intersection / union

    # Mark classes absent from the target as NaN.
    target_present = (cm.sum(dim=0) + cm.sum(dim=1)) > 0
    iou = iou.where(target_present, torch.tensor(float("nan"), device=iou.device))
    return iou


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    r"""Mean intersection-over-union over classes present in the target.

    Args:
        pred: predictions of shape ``(B, H, W)`` or ``(B, C, H, W)``.
        target: ground-truth labels of shape ``(B, H, W)``.
        num_classes: number of classes.
        ignore_index: optional class index to ignore.

    Returns:
        Scalar mean IoU.
    """
    iou = per_class_iou(pred, target, num_classes, ignore_index)
    return iou.nanmean()


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = None,
) -> torch.Tensor:
    r"""Pixel-level classification accuracy.

    Args:
        pred: predictions of shape ``(B, H, W)`` or ``(B, C, H, W)``.
        target: ground-truth labels of shape ``(B, H, W)``.
        ignore_index: optional class index to ignore.

    Returns:
        Scalar accuracy in ``[0, 1]``.
    """
    pred = _pred_to_label(pred)
    pred, target = _filter_ignore_index(pred, target, ignore_index)
    valid = target >= 0
    if not valid.any():
        return torch.tensor(0.0, device=pred.device)
    correct = (pred == target)[valid].float()
    return correct.mean()


def _dilate_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Morphological dilation of a binary mask.

    Args:
        mask: binary tensor of shape ``(N, 1, H, W)``.
        width: dilation radius.

    Returns:
        Dilated binary mask of the same shape.
    """
    kernel_size = 2 * width + 1
    return F.max_pool2d(
        mask.float(), kernel_size=kernel_size, stride=1, padding=width
    ).to(mask.dtype)


def _erode_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Morphological erosion of a binary mask.

    Args:
        mask: binary tensor of shape ``(N, 1, H, W)``.
        width: erosion radius.

    Returns:
        Eroded binary mask of the same dtype as input.
    """
    return (~_dilate_mask(~mask, width)).to(mask.dtype)


def _boundary_mask(
    mask: torch.Tensor, width: int
) -> torch.Tensor:
    """Extract the boundary of a binary mask.

    Args:
        mask: binary tensor of shape ``(N, 1, H, W)``.
        width: boundary width.

    Returns:
        Boundary mask of the same dtype as input.
    """
    dilated = _dilate_mask(mask, width)
    eroded = _erode_mask(mask, width)
    return dilated & eroded


def boundary_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    boundary_width: int = 2,
    ignore_index: int | None = None,
) -> torch.Tensor:
    r"""Mean IoU computed only on boundary pixels.

    Boundaries are extracted from the ground-truth mask using morphological
    dilation/erosion.  For each class, intersection and union are accumulated
    over pixels within the class boundary, and the mean is taken over classes
    present in the target.

    Args:
        pred: predictions of shape ``(B, H, W)`` or ``(B, C, H, W)``.
        target: ground-truth labels of shape ``(B, H, W)``.
        num_classes: number of classes.
        boundary_width: boundary radius in pixels.
        ignore_index: optional class index to ignore.

    Returns:
        Scalar boundary IoU.
    """
    pred = _pred_to_label(pred)
    pred, target = _filter_ignore_index(pred, target, ignore_index)
    B, H, W = target.shape
    device = target.device

    ious = []
    for c in range(num_classes):
        target_mask = (target == c).reshape(B, 1, H, W)
        if not target_mask.any():
            continue
        boundary = _boundary_mask(target_mask, boundary_width).reshape(B, H, W)

        pred_mask = (pred == c)
        intersection = (pred_mask & target_mask.reshape(B, H, W) & boundary).float().sum()
        union = ((pred_mask | target_mask.reshape(B, H, W)) & boundary).float().sum()
        if union == 0:
            continue
        ious.append(intersection / (union + 1e-12))

    if not ious:
        return torch.tensor(0.0, device=device)
    return torch.stack(ious).mean()


def eval_segmentation(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    boundary_width: int = 2,
    ignore_index: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute a standard set of segmentation metrics.

    Args:
        pred: predictions of shape ``(B, H, W)`` or ``(B, C, H, W)``.
        target: ground-truth labels of shape ``(B, H, W)``.
        num_classes: number of classes.
        boundary_width: boundary radius for ``boundary_iou``.
        ignore_index: optional class index to ignore.

    Returns:
        Dictionary with keys ``mIoU``, ``boundary_iou``, ``pixel_accuracy``,
        and ``per_class_iou``.
    """
    return {
        "mIoU": mean_iou(pred, target, num_classes, ignore_index),
        "boundary_iou": boundary_iou(
            pred, target, num_classes, boundary_width, ignore_index
        ),
        "pixel_accuracy": pixel_accuracy(pred, target, ignore_index),
        "per_class_iou": per_class_iou(pred, target, num_classes, ignore_index),
    }
