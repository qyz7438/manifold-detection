"""Evaluation utilities for image classification."""

from __future__ import annotations

import torch
import torch.nn as nn


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)
) -> list[float]:
    """Compute top-k accuracy for a single batch of predictions.

    Args:
        output: classifier logits of shape ``(B, num_classes)``.
        target: ground-truth class indices of shape ``(B,)``.
        topk: tuple of ``k`` values to evaluate.

    Returns:
        List of accuracy percentages, one entry per ``k`` in ``topk``.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader,
    device: torch.device | str,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Evaluate top-k accuracy of a classifier over a dataset.

    Args:
        model: classifier model.
        dataloader: iterable yielding ``(inputs, targets)`` batches.
        device: device on which to run evaluation.
        topk: tuple of ``k`` values to evaluate.

    Returns:
        Dictionary mapping ``f"top{k}_acc"`` to the corresponding accuracy.
    """
    model.eval()
    model.to(device)
    total = 0
    corrects = {k: 0 for k in topk}
    maxk = max(topk)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        total += targets.size(0)
        for k in topk:
            corrects[k] += correct[:k].reshape(-1).float().sum().item()

    return {f"top{k}_acc": 100.0 * corrects[k] / total for k in topk}
