import pytest
import torch

from spectral_detection_posttrain.rlvr.action_verifier import (
    ActionVerifierConfig,
    build_action_batch,
    build_dpo_pairs,
    build_rlvr_rewards,
    compute_fft_action_quality,
    compute_manifold_action_quality,
    decode_box_actions,
)


def test_decode_box_actions_preserves_shape_and_changes_box():
    proposals = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    deltas = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])

    boxes = decode_box_actions(proposals, deltas, image_size=(100, 100))

    assert boxes.shape == (1, 2, 4)
    assert torch.allclose(boxes[0, 0], proposals[0])
    assert boxes[0, 1, 0] > proposals[0, 0]


def test_decode_box_actions_supports_gradient_to_deltas():
    proposals = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    deltas = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], requires_grad=True)

    boxes = decode_box_actions(proposals, deltas, image_size=(100, 100))
    loss = boxes[..., 0].sum()
    loss.backward()

    assert deltas.grad is not None
    assert float(deltas.grad.abs().sum()) > 0.0


def test_build_action_batch_returns_log_probs_and_decoded_boxes():
    proposals = torch.tensor([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 10.0, 10.0]])
    mu = torch.zeros((2, 4))
    cfg = ActionVerifierConfig(num_samples=3, sigma=0.1, seed=123)

    batch = build_action_batch(proposals, mu, image_size=(100, 100), cfg=cfg)

    assert batch.proposals.shape == (2, 4)
    assert batch.deltas.shape == (2, 3, 4)
    assert batch.decoded_boxes.shape == (2, 3, 4)
    assert batch.log_probs.shape == (2, 3)


def test_build_action_batch_can_include_identity_proposal_action():
    proposals = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    mu = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    cfg = ActionVerifierConfig(num_samples=3, sigma=0.1, seed=123, include_identity_action=True)

    batch = build_action_batch(proposals, mu, image_size=(100, 100), cfg=cfg)

    assert torch.allclose(batch.deltas[0, 0], torch.zeros(4))
    assert torch.allclose(batch.decoded_boxes[0, 0], proposals[0])
    assert not torch.allclose(batch.decoded_boxes[0, 1], proposals[0])


def test_action_batch_log_probs_keep_gradient_to_mu():
    proposals = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    mu = torch.zeros((1, 4), requires_grad=True)
    cfg = ActionVerifierConfig(num_samples=2, sigma=0.1, seed=123)

    batch = build_action_batch(proposals, mu, image_size=(100, 100), cfg=cfg)
    loss = -batch.log_probs[:, 1].mean()
    loss.backward()

    assert mu.grad is not None
    assert float(mu.grad.abs().sum().item()) > 0.0


def test_fft_action_quality_is_action_dependent():
    image = torch.zeros((3, 64, 64))
    image[:, 20:44, 20:44] = 1.0
    boxes = torch.tensor([[[20.0, 20.0, 44.0, 44.0], [0.0, 0.0, 12.0, 12.0]]])

    quality = compute_fft_action_quality(image, boxes, crop_size=32)

    assert quality.shape == (1, 2)
    assert quality[0, 0] > quality[0, 1]


def test_manifold_action_quality_prefers_reference_like_features():
    features = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    reference = torch.tensor([[0.0, 0.0], [0.2, 0.1]])

    quality = compute_manifold_action_quality(features, reference)

    assert quality.shape == (1, 2)
    assert quality[0, 0] > quality[0, 1]


def test_rlvr_rewards_gate_positive_reward_to_matched_actions():
    iou = torch.tensor([[0.9, 0.2]])
    verifier = torch.tensor([[0.5, 1.0]])
    matched = iou >= 0.5

    rewards = build_rlvr_rewards(iou, verifier, matched, verifier_weight=0.5)

    assert rewards[0, 0] > 0.9
    assert rewards[0, 1] <= 0.0


def test_dpo_pairs_skip_ties_by_margin():
    quality = torch.tensor([[0.8, 0.7], [0.5, 0.49]])
    pairs = build_dpo_pairs(quality, margin=0.05)

    assert pairs.valid.tolist() == [True, False]
    assert pairs.chosen_indices.tolist() == [0, 0]
    assert pairs.rejected_indices.tolist() == [1, 1]


def test_identity_action_lets_dpo_prefer_high_iou_proposal_over_decoded_action():
    proposals = torch.tensor([[10.0, 10.0, 30.0, 50.0]])
    gt = torch.tensor([[10.0, 10.0, 30.0, 50.0]])
    mu = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    cfg = ActionVerifierConfig(num_samples=2, sigma=0.1, seed=123, include_identity_action=True)

    batch = build_action_batch(proposals, mu, image_size=(100, 100), cfg=cfg)
    iou = torch.stack(
        [
            torch.tensor(
                [
                    1.0,
                    0.0 if torch.allclose(batch.decoded_boxes[0, 1], gt[0]) else 0.5,
                ]
            )
        ]
    )
    pairs = build_dpo_pairs(iou, margin=0.0)

    assert pairs.valid.tolist() == [True]
    assert pairs.chosen_indices.tolist() == [0]
    assert pairs.rejected_indices.tolist() == [1]


def test_dpo_pairs_reject_more_than_two_actions_for_now():
    quality = torch.tensor([[0.8, 0.7, 0.6]])

    with pytest.raises(ValueError, match="exactly two"):
        build_dpo_pairs(quality, margin=0.05)
