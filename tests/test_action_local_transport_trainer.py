from __future__ import annotations

import torch

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
