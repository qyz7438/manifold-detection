from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import spectral_detection_posttrain.trainers.detection.action_local_transport as action_transport

from spectral_detection_posttrain.methods.energy_transport import (
    ROIActionState,
    ROITransportActions,
    apply_box_delta,
)
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    ProposalActionBatch,
    SupervisedActionLossConfig,
    action_batch_to_predictions,
    effective_actions,
    encode_box_delta,
    rematch_roi_action_state,
    supervised_action_transport_loss,
)


def _make_batch() -> ProposalActionBatch:
    features = torch.randn(4, 8)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 40.0, 40.0],
            [50.0, 50.0, 80.0, 80.0],
            [5.0, 5.0, 15.0, 15.0],
        ],
        dtype=torch.float32,
    )
    target_boxes = torch.tensor(
        [
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 40.0, 40.0],
            [52.0, 52.0, 82.0, 82.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    state = ROIActionState(
        features=features,
        boxes=boxes,
        scores=torch.tensor([0.9, 0.8, 0.7, 0.1]),
        labels=torch.tensor([1, 1, 2, 0]),
        image_indices=torch.tensor([0, 0, 0, 0]),
        proposal_indices=torch.arange(4),
        matched_gt_indices=torch.tensor([0, 1, 2, -1]),
        ious=torch.tensor([0.70, 0.90, 0.74, 0.0]),
    )
    return ProposalActionBatch(
        state=state,
        matched_gt_boxes=target_boxes,
        matched_gt_labels=torch.tensor([1, 1, 2, 0]),
        image_sizes=[(100, 100)],
    )


def test_encode_box_delta_round_trip_with_apply_box_delta() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 30.0, 20.0]])
    targets = torch.tensor([[1.0, 2.0, 12.0, 14.0], [12.0, 9.0, 34.0, 23.0]])
    deltas = encode_box_delta(boxes, targets)
    decoded = apply_box_delta(boxes, deltas)
    assert torch.allclose(decoded, targets, atol=1e-5)


def test_effective_actions_uses_keep_logit_as_action_strength() -> None:
    batch = _make_batch()
    actions = ROITransportActions(
        feature_delta=torch.ones_like(batch.state.features),
        score_delta=torch.ones(batch.state.batch_size),
        box_delta=torch.ones(batch.state.batch_size, 4),
        keep_logit=torch.tensor([-20.0, 0.0, 20.0, 20.0]),
    )
    gated = effective_actions(actions, gate_actions=True)
    assert gated.box_delta[0].abs().max().item() < 1e-6
    assert gated.box_delta[1].mean().item() == 0.5
    assert gated.box_delta[2].mean().item() > 0.999


def test_rematch_roi_action_state_supports_class_agnostic_matching() -> None:
    state = ROIActionState(
        features=torch.randn(1, 4),
        boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([2]),
        image_indices=torch.tensor([0]),
        proposal_indices=torch.tensor([0]),
    )
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([1]),
        }
    ]
    aware = rematch_roi_action_state(state, targets, match_mode="class_aware")
    agnostic = rematch_roi_action_state(state, targets, match_mode="class_agnostic")
    assert aware.matched_gt_indices.tolist() == [-1]
    assert aware.ious.tolist() == [0.0]
    assert agnostic.matched_gt_indices.tolist() == [0]
    assert agnostic.ious.tolist() == [1.0]


def test_supervised_action_transport_loss_has_gradients() -> None:
    batch = _make_batch()
    actions = ROITransportActions(
        feature_delta=torch.zeros_like(batch.state.features, requires_grad=True),
        score_delta=torch.zeros(batch.state.batch_size, requires_grad=True),
        box_delta=torch.zeros(batch.state.batch_size, 4, requires_grad=True),
        keep_logit=torch.zeros(batch.state.batch_size, requires_grad=True),
    )
    loss_dict = supervised_action_transport_loss(
        batch,
        actions,
        config=SupervisedActionLossConfig(
            energy_weight=0.0,
            gate_loss_weight=0.1,
            high_iou_preserve_weight=0.1,
        ),
    )
    assert loss_dict["loss_total"].item() > 0.0
    loss_dict["loss_total"].backward()
    assert actions.box_delta.grad is not None
    assert actions.keep_logit.grad is not None


def test_action_batch_to_predictions_outputs_detection_format() -> None:
    batch = _make_batch()
    actions = ROITransportActions(
        feature_delta=torch.zeros_like(batch.state.features),
        score_delta=torch.zeros(batch.state.batch_size),
        box_delta=torch.zeros(batch.state.batch_size, 4),
        keep_logit=torch.zeros(batch.state.batch_size),
    )
    predictions = action_batch_to_predictions(
        batch,
        actions,
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=2,
    )
    assert len(predictions) == 1
    pred = predictions[0]
    assert set(pred) == {"boxes", "scores", "labels"}
    assert pred["boxes"].shape[1] == 4
    assert pred["scores"].numel() <= 2


def test_zero_actions_match_torchvision_native_postprocess_exactly() -> None:
    assert hasattr(action_transport, "postprocess_action_detections")

    class NativePostprocess:
        box_coder = det_utils.BoxCoder((10.0, 10.0, 5.0, 5.0))
        score_thresh = 0.05
        nms_thresh = 0.5
        detections_per_img = 100

        postprocess_detections = RoIHeads.postprocess_detections

    native = NativePostprocess()
    proposals = [
        torch.tensor(
            [[0.0, 0.0, 20.0, 20.0], [2.0, 2.0, 22.0, 22.0], [40.0, 40.0, 60.0, 60.0]]
        )
    ]
    logits = torch.tensor(
        [[-2.0, 3.0, 2.0], [-2.0, 2.5, 2.2], [-1.0, 0.5, 2.5]],
        dtype=torch.float32,
    )
    box_regression = torch.zeros((3, 12), dtype=torch.float32)
    labels = F.softmax(logits, dim=-1)[:, 1:].argmax(dim=1) + 1
    actions = ROITransportActions(
        feature_delta=torch.zeros((3, 8)),
        score_delta=torch.zeros(3),
        box_delta=torch.zeros((3, 4)),
        keep_logit=torch.zeros(3),
    )

    expected = native.postprocess_detections(
        logits,
        box_regression,
        proposals,
        [(64, 64)],
    )
    actual = action_transport.postprocess_action_detections(
        box_coder=native.box_coder,
        class_logits=logits,
        box_regression=box_regression,
        proposals=proposals,
        image_shapes=[(64, 64)],
        action_labels=labels,
        actions=actions,
        score_threshold=native.score_thresh,
        nms_threshold=native.nms_thresh,
        detections_per_img=native.detections_per_img,
    )

    for expected_group, actual_group in zip(expected, actual):
        for expected_tensor, actual_tensor in zip(expected_group, actual_group):
            assert torch.equal(expected_tensor, actual_tensor)


def test_native_action_batch_restores_original_coordinates() -> None:
    class NativeRoIHeads:
        box_coder = det_utils.BoxCoder((10.0, 10.0, 5.0, 5.0))
        score_thresh = 0.05
        nms_thresh = 0.5
        detections_per_img = 100

    class NativeModel:
        roi_heads = NativeRoIHeads()
        transform = GeneralizedRCNNTransform(
            min_size=32,
            max_size=32,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
        ).eval()

    proposals = [torch.tensor([[0.0, 0.0, 20.0, 20.0], [8.0, 8.0, 28.0, 28.0]])]
    logits = torch.tensor([[-2.0, 3.0, 2.0], [-1.0, 0.5, 2.5]])
    box_regression = torch.zeros((2, 12))
    scores = F.softmax(logits, dim=-1)
    labels = scores[:, 1:].argmax(dim=1) + 1
    state = ROIActionState(
        features=torch.zeros((2, 8)),
        boxes=proposals[0],
        scores=scores[torch.arange(2), labels],
        labels=labels,
        image_indices=torch.zeros(2, dtype=torch.long),
        proposal_indices=torch.arange(2),
    )
    batch = ProposalActionBatch(
        state=state,
        matched_gt_boxes=torch.zeros((2, 4)),
        matched_gt_labels=torch.zeros(2, dtype=torch.long),
        image_sizes=[(32, 32)],
        original_image_sizes=[(32, 48)],
        proposals=proposals,
        class_logits=logits,
        box_regression=box_regression,
    )
    actions = ROITransportActions(
        feature_delta=torch.zeros((2, 8)),
        score_delta=torch.zeros(2),
        box_delta=torch.zeros((2, 4)),
        keep_logit=torch.zeros(2),
    )

    boxes, native_scores, native_labels = RoIHeads.postprocess_detections(
        NativeModel.roi_heads,
        logits,
        box_regression,
        proposals,
        [(32, 32)],
    )
    expected = NativeModel.transform.postprocess(
        [{"boxes": boxes[0], "scores": native_scores[0], "labels": native_labels[0]}],
        [(32, 32)],
        [(32, 48)],
    )
    actual = action_batch_to_predictions(
        batch,
        actions,
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=100,
        native_model=NativeModel(),
    )

    assert torch.equal(actual[0]["labels"], expected[0]["labels"])
    assert torch.equal(actual[0]["scores"], expected[0]["scores"])
    assert torch.equal(actual[0]["boxes"], expected[0]["boxes"])
