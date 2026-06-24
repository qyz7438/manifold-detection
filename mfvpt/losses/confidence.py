from __future__ import annotations

import torch


def high_confidence_error_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    wrong = (pred != labels).float()
    return (wrong * conf).mean()
