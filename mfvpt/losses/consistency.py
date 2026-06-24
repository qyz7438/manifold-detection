from __future__ import annotations

import torch

from .view_consistency import kl_view_consistency_loss


def kl_consistency_loss(logits_teacher: torch.Tensor, logits_student: torch.Tensor) -> torch.Tensor:
    return kl_view_consistency_loss(logits_teacher, logits_student)
