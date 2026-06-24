from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_view_consistency_loss(logits_teacher: torch.Tensor, logits_student: torch.Tensor) -> torch.Tensor:
    teacher_prob = torch.softmax(logits_teacher.detach(), dim=1)
    student_log_prob = torch.log_softmax(logits_student, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
