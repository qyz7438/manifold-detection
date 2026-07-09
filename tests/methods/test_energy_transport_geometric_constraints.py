"""Unit tests for action-local geometric constraints in energy_transport."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport import (
    ActionLocalTransportHead,
    ROIActionState,
    ROITransportActions,
)
from spectral_detection_posttrain.methods.energy_transport.geometric_constraints import (
    GeometricConstraintConfig,
    bbox_aware_action_loss,
    bbox_aware_direct_loss,
    bbox_aware_linearization_loss,
    bg_proto_action_loss,
    classify_error_modes,
    cls_err_action_loss,
    compute_action_local_prototypes,
    feat_preserve_action_loss,
    fg_bg_sep_action_loss,
    geometric_transport_loss,
    intra_tp_action_loss,
    loc_err_action_loss,
)
from spectral_detection_posttrain.methods.energy_transport.high_water_mark import (
    HighWaterMarkLossConfig,
    ap75_boundary_weights,
    capture_high_water_mark_module,
    high_water_mark_action_loss,
    load_high_water_mark_module,
    should_update_high_water_mark,
    stop_high_water_mark_loss,
)


def _make_state(
    *,
    batch: int = 8,
    dim: int = 16,
    num_classes: int = 4,
    labels: torch.Tensor | None = None,
    ious: torch.Tensor | None = None,
    seed: int = 42,
) -> ROIActionState:
    """Build a deterministic ``ROIActionState`` for testing."""
    torch.manual_seed(seed)
    features = torch.randn(batch, dim)
    boxes = torch.rand(batch, 4)
    boxes[:, 2:] += boxes[:, :2]  # x2 > x1, y2 > y1
    scores = torch.rand(batch)
    if labels is None:
        labels = torch.randint(0, num_classes, (batch,))
    if ious is None:
        ious = torch.rand(batch)
    image_indices = torch.zeros(batch, dtype=torch.long)
    proposal_indices = torch.arange(batch, dtype=torch.long)
    logits = torch.randn(batch, num_classes)
    return ROIActionState(
        features=features,
        boxes=boxes,
        scores=scores,
        labels=labels,
        image_indices=image_indices,
        proposal_indices=proposal_indices,
        logits=logits,
        matched_gt_indices=torch.randint(-1, 5, (batch,)),
        ious=ious,
    )


def _make_actions(state: ROIActionState) -> ROITransportActions:
    torch.manual_seed(43)
    return ROITransportActions(
        feature_delta=torch.randn_like(state.features) * 0.05,
        score_delta=torch.randn(state.batch_size) * 0.05,
        box_delta=torch.randn(state.batch_size, 4) * 0.05,
        keep_logit=torch.randn(state.batch_size),
    )


def test_compute_action_local_prototypes_shape():
    state = _make_state(batch=16, dim=8, num_classes=4)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    assert prototypes.shape == (4, 8)


def test_compute_action_local_prototypes_empty_class():
    # All samples belong to class 1; class 0,2,3 should be zero.
    labels = torch.ones(16, dtype=torch.long)
    state = _make_state(batch=16, dim=8, num_classes=4, labels=labels)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    assert prototypes[0].abs().max().item() == 0.0
    assert prototypes[2].abs().max().item() == 0.0
    assert prototypes[3].abs().max().item() == 0.0
    assert prototypes[1].abs().max().item() > 0.0


def test_compute_action_local_prototypes_iou_floor():
    labels = torch.ones(16, dtype=torch.long)
    ious = torch.linspace(0.0, 1.0, 16)
    state = _make_state(batch=16, dim=8, num_classes=2, labels=labels, ious=ious)
    prototypes_low = compute_action_local_prototypes(state, 2, iou_floor=0.0)
    prototypes_high = compute_action_local_prototypes(state, 2, iou_floor=0.8)
    # High floor should use fewer samples -> different prototype.
    assert not torch.allclose(prototypes_low[1], prototypes_high[1])


def test_classify_error_modes():
    # 0: TP (correct, high IoU)
    # 1: CLS_ERR (predicted class 2, GT class 1, high IoU)
    # 2: LOC_ERR (background prediction, high IoU with GT class 1)
    # 3: pure_BG (background, low IoU)
    # 4: ambig_BG (background, moderate IoU)
    labels = torch.tensor([1, 2, 0, 0, 0])
    ious = torch.tensor([0.8, 0.8, 0.6, 0.1, 0.4])
    state = _make_state(batch=5, dim=8, num_classes=3, labels=labels, ious=ious)
    gt_labels = torch.tensor([1, 1, 1, 0, 0])
    masks = classify_error_modes(
        state,
        gt_labels,
        iou_threshold=0.5,
        high_iou_threshold=0.7,
        ambiguous_iou_upper=0.3,
    )
    assert masks["tp"][0].item()
    assert masks["cls_err"][1].item()
    assert masks["loc_err"][2].item()
    assert masks["pure_bg"][3].item()
    assert masks["ambig_bg"][4].item()
    # Mutually exclusive.
    overlap = (
        masks["tp"]
        | masks["cls_err"]
        | masks["loc_err"]
        | masks["pure_bg"]
        | masks["ambig_bg"]
    )
    assert overlap.sum().item() == 5


def test_classify_error_modes_requires_ious():
    state = _make_state()
    state = ROIActionState(
        features=state.features,
        boxes=state.boxes,
        scores=state.scores,
        labels=state.labels,
        image_indices=state.image_indices,
        proposal_indices=state.proposal_indices,
        ious=None,
    )
    with pytest.raises(ValueError, match="ious"):
        classify_error_modes(state, torch.zeros(state.batch_size))


def test_intra_tp_action_loss_direct():
    state = _make_state(batch=16, dim=8, num_classes=3)
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=3)
    gt_labels = state.labels.clone()
    masks = classify_error_modes(state, gt_labels)
    loss = intra_tp_action_loss(
        state,
        actions,
        prototypes,
        masks,
        lambda_intra=1.0,
        normalize=True,
        mode="direct",
    )
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_intra_tp_action_loss_conservative_punishes_moving_away():
    state = _make_state(batch=16, dim=8, num_classes=3)
    prototypes = compute_action_local_prototypes(state, num_classes=3)
    gt_labels = state.labels.clone()
    masks = classify_error_modes(state, gt_labels)

    # Action that moves features away from their prototype.
    z = F.normalize(state.features, dim=-1)
    target = F.normalize(prototypes, dim=-1)[state.labels]
    away = target - z  # point toward prototype
    away = -away  # point away from prototype
    actions = ROITransportActions(
        feature_delta=away * 0.5,
        score_delta=torch.zeros(state.batch_size),
        box_delta=torch.zeros(state.batch_size, 4),
        keep_logit=torch.zeros(state.batch_size),
    )

    loss_direct = intra_tp_action_loss(
        state, actions, prototypes, masks, lambda_intra=1.0, mode="direct"
    )
    loss_cons = intra_tp_action_loss(
        state, actions, prototypes, masks, lambda_intra=1.0, mode="conservative"
    )
    # Conservative loss should be positive when moving away; direct loss may be
    # larger because it penalizes absolute distance, not just the increase.
    assert loss_cons.item() > 0.0
    assert loss_direct.item() > 0.0


def test_intra_tp_action_loss_zero_when_disabled():
    state = _make_state()
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    masks = classify_error_modes(state, state.labels)
    loss = intra_tp_action_loss(
        state, actions, prototypes, masks, lambda_intra=0.0
    )
    assert loss.item() == 0.0


def test_fg_bg_sep_action_loss():
    state = _make_state(batch=16, dim=8, num_classes=4)
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    masks = classify_error_modes(state, state.labels)
    loss = fg_bg_sep_action_loss(
        state,
        actions,
        prototypes,
        masks,
        lambda_sep=1.0,
        margin=0.5,
        normalize=True,
    )
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_cls_err_action_loss():
    # Two samples: one CLS_ERR and one TP.
    labels = torch.tensor([1, 2])
    gt_labels = torch.tensor([2, 2])
    ious = torch.tensor([0.8, 0.8])
    state = _make_state(batch=2, dim=8, num_classes=3, labels=labels, ious=ious)
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=3)
    masks = classify_error_modes(state, gt_labels)
    assert masks["cls_err"][0].item()
    assert masks["tp"][1].item()
    loss = cls_err_action_loss(
        state, actions, prototypes, masks, gt_labels, lambda_cls=1.0
    )
    assert loss.item() >= 0.0


def test_loc_err_action_loss():
    state = _make_state(batch=16, dim=8, num_classes=4)
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    masks = classify_error_modes(state, state.labels)
    bg_proto = prototypes[0]
    loss = loc_err_action_loss(
        state, actions, prototypes, masks, bg_proto, lambda_loc=1.0
    )
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_bg_proto_action_loss():
    state = _make_state(batch=16, dim=8, num_classes=4)
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    masks = classify_error_modes(state, state.labels)
    loss = bg_proto_action_loss(
        state, actions, masks, prototypes[0], lambda_bg=1.0
    )
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_feat_preserve_action_loss():
    state = _make_state(batch=16, dim=8, num_classes=4)
    actions = _make_actions(state)
    teacher = state.features.clone()
    loss = feat_preserve_action_loss(
        state, actions, teacher, lambda_preserve=1.0
    )
    assert loss.item() >= 0.0


def test_bbox_aware_direct_loss_protects_high_iou():
    # Two foreground proposals: one IoU=0.9, one IoU=0.4.
    labels = torch.tensor([1, 1])
    ious = torch.tensor([0.9, 0.4])
    state = _make_state(batch=2, dim=8, num_classes=2, labels=labels, ious=ious)
    actions = ROITransportActions(
        feature_delta=torch.zeros(2, 8),
        score_delta=torch.zeros(2),
        box_delta=torch.ones(2, 4) * 0.1,
        keep_logit=torch.zeros(2),
    )
    loss = bbox_aware_direct_loss(
        state, actions, lambda_bbox=1.0, iou_threshold=0.5
    )
    # Only the high-IoU sample is above threshold, so loss should reflect its
    # (weighted) contribution.
    assert loss.item() > 0.0


def test_bbox_aware_direct_loss_zero_when_disabled():
    state = _make_state()
    actions = _make_actions(state)
    loss = bbox_aware_direct_loss(state, actions, lambda_bbox=0.0)
    assert loss.item() == 0.0


def test_bbox_aware_linearization_loss():
    state = _make_state(batch=16, dim=8, num_classes=4)
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    masks = classify_error_modes(state, state.labels)
    # bbox_pred_weight shape: (num_classes * 4, D)
    bbox_pred_weight = torch.randn(4 * 4, 8)
    loss = bbox_aware_linearization_loss(
        state,
        actions,
        prototypes,
        masks,
        bbox_pred_weight,
        num_classes=4,
        lambda_bbox=1.0,
    )
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_bbox_aware_action_loss_routing():
    state = _make_state(batch=16, dim=8, num_classes=4)
    actions = _make_actions(state)
    prototypes = compute_action_local_prototypes(state, num_classes=4)
    masks = classify_error_modes(state, state.labels)

    config_direct = GeometricConstraintConfig(
        lambda_bbox_aware=1.0, bbox_aware_mode="direct"
    )
    loss_direct = bbox_aware_action_loss(
        state, actions, config_direct, prototypes, masks
    )

    config_lin = GeometricConstraintConfig(
        lambda_bbox_aware=1.0, bbox_aware_mode="linearization"
    )
    bbox_pred_weight = torch.randn(4 * 4, 8)
    loss_lin = bbox_aware_action_loss(
        state,
        actions,
        config_lin,
        prototypes,
        masks,
        bbox_pred_weight=bbox_pred_weight,
        num_classes=4,
    )
    assert loss_direct.item() >= 0.0
    assert loss_lin.item() >= 0.0


def test_geometric_transport_loss_aggregator():
    state = _make_state(batch=16, dim=8, num_classes=4)
    actions = _make_actions(state)
    config = GeometricConstraintConfig(
        lambda_intra_tp=1.0,
        lambda_fg_bg_sep=1.0,
        lambda_cls_err=1.0,
        lambda_loc_err=1.0,
        lambda_bbox_aware=1.0,
        lambda_geo_total=0.5,
    )
    loss_dict = geometric_transport_loss(
        state, actions, num_classes=4, config=config, gt_labels=state.labels
    )
    expected_keys = {
        "loss_geo_total",
        "loss_geo_intra_tp",
        "loss_geo_fg_bg_sep",
        "loss_geo_cls_err",
        "loss_geo_loc_err",
        "loss_geo_bg_proto",
        "loss_geo_feat_preserve",
        "loss_geo_bbox_aware",
    }
    assert set(loss_dict.keys()) == expected_keys
    assert loss_dict["loss_geo_total"].item() >= 0.0


def test_geometric_transport_loss_grad_flow():
    # Explicit labels/IoUs so that we are guaranteed TP and high-IoU FG samples.
    # Classes: 0=background, 1,2=foreground.  IoU threshold=0.5, high=0.7.
    labels = torch.tensor([1, 1, 2, 1, 0, 0, 0, 1])
    gt_labels = torch.tensor([1, 1, 2, 0, 1, 0, 0, 1])
    ious = torch.tensor([0.8, 0.9, 0.85, 0.6, 0.6, 0.1, 0.4, 0.55])
    state = _make_state(
        batch=8, dim=8, num_classes=3, labels=labels, ious=ious
    )
    actions = _make_actions(state)
    # In a trainer actions come from a neural net; simulate that by requiring
    # grads on the action tensors.
    actions.feature_delta.requires_grad_(True)
    actions.score_delta.requires_grad_(True)
    actions.box_delta.requires_grad_(True)
    config = GeometricConstraintConfig(
        lambda_intra_tp=1.0,
        lambda_bbox_aware=1.0,
    )
    loss_dict = geometric_transport_loss(
        state, actions, num_classes=3, config=config, gt_labels=gt_labels
    )
    loss = loss_dict["loss_geo_total"]
    assert loss.requires_grad
    loss.backward()
    assert actions.feature_delta.grad is not None
    assert actions.box_delta.grad is not None


def test_ap75_boundary_weights_peak_at_boundary():
    ious = torch.tensor([0.55, 0.70, 0.75, 0.80, 0.95])
    weights = ap75_boundary_weights(ious, center=0.75, band=0.10)
    assert weights[2].item() == pytest.approx(1.0)
    assert weights[1].item() == pytest.approx(weights[3].item())
    assert weights[0].item() == pytest.approx(0.0)
    assert weights[4].item() == pytest.approx(0.0)


def test_high_water_mark_action_loss_zero_inside_margin():
    state = _make_state(batch=4, dim=8, num_classes=3)
    teacher = _make_actions(state)
    actions = ROITransportActions(
        feature_delta=teacher.feature_delta.clone(),
        score_delta=teacher.score_delta.clone(),
        box_delta=teacher.box_delta.clone() + 0.001,
        keep_logit=teacher.keep_logit.clone(),
    )
    loss_dict = high_water_mark_action_loss(
        state,
        actions,
        teacher,
        HighWaterMarkLossConfig(epsilon=0.05, box_weight=1.0, keep_weight=0.0),
    )
    assert loss_dict["loss_hwm_total"].item() == pytest.approx(0.0)


def test_high_water_mark_action_loss_has_grad_near_ap75():
    labels = torch.ones(4, dtype=torch.long)
    ious = torch.tensor([0.55, 0.74, 0.76, 0.95])
    state = _make_state(batch=4, dim=8, num_classes=2, labels=labels, ious=ious)
    teacher = ROITransportActions(
        feature_delta=torch.zeros_like(state.features),
        score_delta=torch.zeros(state.batch_size),
        box_delta=torch.zeros(state.batch_size, 4),
        keep_logit=torch.zeros(state.batch_size),
    )
    actions = ROITransportActions(
        feature_delta=torch.zeros_like(state.features),
        score_delta=torch.zeros(state.batch_size),
        box_delta=torch.full((state.batch_size, 4), 0.2, requires_grad=True),
        keep_logit=torch.zeros(state.batch_size),
    )
    loss_dict = high_water_mark_action_loss(
        state,
        actions,
        teacher,
        HighWaterMarkLossConfig(epsilon=0.01, box_weight=1.0, keep_weight=0.0),
    )
    loss = loss_dict["loss_hwm_total"]
    assert loss.item() > 0.0
    loss.backward()
    assert actions.box_delta.grad is not None
    grad_norms = actions.box_delta.grad.flatten(1).norm(dim=1)
    assert grad_norms[1].item() > 0.0
    assert grad_norms[2].item() > 0.0
    assert grad_norms[0].item() == pytest.approx(0.0)
    assert grad_norms[3].item() == pytest.approx(0.0)


def test_stop_high_water_mark_loss_only_uses_keep_logit():
    state = _make_state(batch=4, dim=8, num_classes=3)
    teacher = ROITransportActions(
        feature_delta=torch.zeros_like(state.features),
        score_delta=torch.zeros(state.batch_size),
        box_delta=torch.zeros(state.batch_size, 4),
        keep_logit=torch.zeros(state.batch_size),
    )
    actions = ROITransportActions(
        feature_delta=torch.ones_like(state.features) * 10.0,
        score_delta=torch.ones(state.batch_size) * 10.0,
        box_delta=torch.ones(state.batch_size, 4) * 10.0,
        keep_logit=torch.ones(state.batch_size, requires_grad=True),
    )
    loss_dict = stop_high_water_mark_loss(
        state, actions, teacher, lambda_stop=1.0, epsilon=0.0
    )
    loss = loss_dict["loss_hwm_total"]
    assert loss.item() > 0.0
    loss.backward()
    assert actions.keep_logit.grad is not None


def test_high_water_mark_module_snapshot_round_trip():
    torch.manual_seed(7)
    head = ActionLocalTransportHead(feature_dim=8, hidden_dim=12, residual_scale=0.01)
    features = torch.randn(3, 8)
    before = head(features).box_delta.detach().clone()
    snapshot = capture_high_water_mark_module(
        head, metric_value=0.4037, epoch=5, metadata={"metric": "ap75"}
    )
    assert snapshot.is_available
    assert snapshot.metric_value == pytest.approx(0.4037)
    assert snapshot.epoch == 5
    assert snapshot.metadata["metric"] == "ap75"

    with torch.no_grad():
        for param in head.parameters():
            param.add_(0.5)
    changed = head(features).box_delta.detach()
    assert not torch.allclose(before, changed)

    load_high_water_mark_module(head, snapshot)
    restored = head(features).box_delta.detach()
    assert torch.allclose(before, restored)


def test_should_update_high_water_mark():
    snapshot = capture_high_water_mark_module(
        ActionLocalTransportHead(feature_dim=4, hidden_dim=4),
        metric_value=0.40,
        epoch=1,
    )
    assert should_update_high_water_mark(0.405, snapshot, min_delta=0.001)
    assert not should_update_high_water_mark(0.4005, snapshot, min_delta=0.001)


def test_config_validation():
    with pytest.raises(ValueError):
        GeometricConstraintConfig(intra_tp_mode="invalid")
    with pytest.raises(ValueError):
        GeometricConstraintConfig(bbox_aware_mode="invalid")
    with pytest.raises(ValueError):
        GeometricConstraintConfig(iou_threshold=1.5)
