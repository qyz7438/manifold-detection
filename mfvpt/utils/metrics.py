from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def prediction_consistency(logits_clean: torch.Tensor, logits_aug: torch.Tensor) -> float:
    return (logits_clean.argmax(dim=1) == logits_aug.argmax(dim=1)).float().mean().item()


def high_confidence_error_rate(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.9) -> float:
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    return ((pred != labels) & (conf > threshold)).float().mean().item()


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    correct = (pred == labels).float()
    ece = torch.zeros((), device=logits.device)
    boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    for i in range(n_bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if i == 0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf > lower) & (conf <= upper)
        if mask.any():
            bucket_acc = correct[mask].mean()
            bucket_conf = conf[mask].mean()
            ece = ece + mask.float().mean() * (bucket_acc - bucket_conf).abs()
    return ece.item()
