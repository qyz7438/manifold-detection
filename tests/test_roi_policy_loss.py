import torch

from spectral_detection_posttrain.rlvr.roi_policy_loss import (
    resize_boxes_to_image,
    weighted_fastrcnn_policy_loss,
)


def test_resize_boxes_to_image_scales_xyxy_coordinates():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    scaled = resize_boxes_to_image(boxes, original_size=(100, 200), new_size=(200, 400))
    assert torch.allclose(scaled, torch.tensor([[20.0, 40.0, 60.0, 80.0]]))


def test_weighted_fastrcnn_policy_loss_handles_no_candidates():
    class_logits = torch.empty((0, 2), requires_grad=True)
    box_regression = torch.empty((0, 8), requires_grad=True)
    labels = torch.empty((0,), dtype=torch.long)
    regression_targets = torch.empty((0, 4))
    weights = torch.empty((0,))

    loss = weighted_fastrcnn_policy_loss(class_logits, box_regression, labels, regression_targets, weights)

    assert loss["loss_roi_policy_cls"].item() == 0.0
    assert loss["loss_roi_policy_box"].item() == 0.0


def test_weighted_fastrcnn_policy_loss_backpropagates():
    class_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    box_regression = torch.zeros((2, 8), requires_grad=True)
    labels = torch.tensor([1, 0])
    regression_targets = torch.zeros((2, 4))
    weights = torch.tensor([2.0, 1.0])

    loss = weighted_fastrcnn_policy_loss(class_logits, box_regression, labels, regression_targets, weights)
    total = loss["loss_roi_policy_cls"] + loss["loss_roi_policy_box"]
    total.backward()

    assert class_logits.grad is not None
    assert box_regression.grad is not None
