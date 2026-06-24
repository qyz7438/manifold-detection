import torch

from mfvpt.losses.confidence import high_confidence_error_penalty
from mfvpt.losses.view_consistency import kl_view_consistency_loss


def test_kl_view_consistency_is_non_negative_and_detaches_teacher():
    teacher = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    student = torch.tensor([[1.5, 0.5], [0.5, 1.5]], requires_grad=True)
    loss = kl_view_consistency_loss(teacher, student)
    assert loss.item() >= 0.0
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_high_confidence_error_penalty_zero_when_correct():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    loss = high_confidence_error_penalty(logits, labels)
    assert loss.item() == 0.0


def test_high_confidence_error_penalty_larger_for_confident_wrong():
    high_wrong = torch.tensor([[10.0, 0.0]])
    low_wrong = torch.tensor([[0.1, 0.0]])
    labels = torch.tensor([1])
    assert high_confidence_error_penalty(high_wrong, labels) > high_confidence_error_penalty(low_wrong, labels)
