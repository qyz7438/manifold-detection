import torch

from spectral_detection_posttrain.rlvr.round211_spatial_verifier import center_size_reward, iou_reward


def test_iou_reward_perfect_box_is_one():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert torch.allclose(iou_reward(box, box), torch.tensor([1.0]))


def test_center_size_reward_is_bounded():
    pred = torch.tensor([[1.0, 1.0, 9.0, 9.0]])
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    value = center_size_reward(pred, gt)
    assert float(value.min()) >= 0.0
    assert float(value.max()) <= 1.0
