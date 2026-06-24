import torch

from spectral_detection_posttrain.rlvr.roi_policy_loss import (
    baseline_kl_loss,
    signed_roi_policy_loss,
)


def test_signed_policy_loss_rewards_high_advantage_action_probability():
    low_action_logit = torch.tensor([[0.0, 0.0]], requires_grad=True)
    high_action_logit = torch.tensor([[0.0, 2.0]], requires_grad=True)
    action_labels = torch.tensor([1])
    advantages = torch.tensor([1.0])

    low_loss = signed_roi_policy_loss(low_action_logit, action_labels, advantages)
    high_loss = signed_roi_policy_loss(high_action_logit, action_labels, advantages)

    assert high_loss.item() < low_loss.item()


def test_signed_policy_loss_penalizes_low_reward_action_probability():
    low_action_logit = torch.tensor([[0.0, 0.0]], requires_grad=True)
    high_action_logit = torch.tensor([[0.0, 2.0]], requires_grad=True)
    action_labels = torch.tensor([1])
    advantages = torch.tensor([-1.0])

    low_loss = signed_roi_policy_loss(low_action_logit, action_labels, advantages)
    high_loss = signed_roi_policy_loss(high_action_logit, action_labels, advantages)

    assert high_loss.item() > low_loss.item()


def test_baseline_kl_loss_is_zero_for_identical_logits():
    logits = torch.tensor([[1.0, 0.5], [0.2, 2.0]], requires_grad=True)
    loss = baseline_kl_loss(logits, logits.detach())

    assert loss.item() < 1e-7


def test_baseline_kl_loss_positive_for_different_logits():
    current = torch.tensor([[2.0, 0.0]], requires_grad=True)
    baseline = torch.tensor([[0.0, 2.0]])

    loss = baseline_kl_loss(current, baseline)

    assert loss.item() > 0.1
