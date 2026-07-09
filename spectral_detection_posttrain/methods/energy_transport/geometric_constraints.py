"""Action-local geometric constraints for energy-guided ROI transport.

This module migrates the ROI-MQ / PG-AW / Plan A / Plan B geometric losses
from the old feature-level manifold path into the action-local
``energy_transport`` framework.

Key design changes from the old path:

1. **Action-level, not feature-level.**  We do not pull raw RoI features
   toward prototypes.  Instead we regularize the transport action so that
   ``feature_delta`` / ``box_delta`` / ``score_delta`` produce the desired
   geometric effect.
2. **BBox-aware is active protection.**  The old Plan B loss penalized the
   box change caused by prototype attraction, which was tiny and passive.
   Here we directly bound ``box_delta`` on high-IoU proposals and can also
   fall back to the old ``bbox_pred.weight`` linearization for ablations.
3. **Sample types use ``ROIActionState.ious``.**  The state already carries
   matched IoUs, so TP / CLS_ERR / LOC_ERR / BG classification is explicit.
4. **Integration with ``ConstraintConfig``.**  All geometric penalties share
   the same safety surface (thresholds, margins, budgets) used by score/box
   actions.

The module is meant to be used inside an action-local trainer:

.. code-block:: python

    state = extract_proposal_roi_action_state(model, images, targets)
    actions = action_head(state.features)
    config = GeometricConstraintConfig(lambda_intra_tp=1.0, lambda_bbox_aware=0.5)
    loss_dict = geometric_transport_loss(
        state, actions, num_classes, config, gt_labels=gt_labels
    )
    loss = loss_dict["loss_geo_total"]

References:

* Old feature-level losses:
  ``spectral_detection_posttrain/methods/manifold/detection_feature_manifold_loss.py``
* PG-AW / Plan A / Plan B reports:
  ``runs/planA_analysis.md``, ``runs/planB_analysis.md``,
  ``runs/final_pgaw_seven_cycle_report.md`` (remote checkout).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.actions import (
    ROITransportActions,
)
from spectral_detection_posttrain.methods.energy_transport.contracts import (
    ROIActionState,
)


SAMPLE_TP = 0
SAMPLE_CLS_ERR = 1
SAMPLE_LOC_ERR = 2
SAMPLE_PURE_BG = 3
SAMPLE_AMBIGUOUS_BG = 4


@dataclass(frozen=True)
class GeometricConstraintConfig:
    """Weights and thresholds for action-local geometric transport losses."""

    lambda_intra_tp: float = 0.0
    lambda_fg_bg_sep: float = 0.0
    lambda_cls_err: float = 0.0
    lambda_loc_err: float = 0.0
    lambda_bbox_aware: float = 0.0
    lambda_bg_proto: float = 0.0
    lambda_feat_preserve: float = 0.0
    lambda_geo_total: float = 1.0

    # Error-mode thresholds.
    iou_threshold: float = 0.5
    high_iou_threshold: float = 0.7
    ambiguous_iou_upper: float = 0.3

    # Margins for contrastive terms.
    margin_fg_bg: float = 0.5
    margin_ambig: float | None = None
    margin_cls: float = 0.3
    margin_loc: float = 0.2
    ambig_iou_tau: float = 0.3

    # Behavior flags.
    normalize: bool = True
    intra_tp_mode: str = "direct"  # "direct" or "conservative"
    bbox_aware_mode: str = "direct"  # "direct" or "linearization"
    min_prototype_samples: int = 1
    iou_floor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0.0 <= self.high_iou_threshold <= 1.0:
            raise ValueError("high_iou_threshold must be in [0, 1]")
        if not 0.0 <= self.ambiguous_iou_upper <= 1.0:
            raise ValueError("ambiguous_iou_upper must be in [0, 1]")
        if self.intra_tp_mode not in ("direct", "conservative"):
            raise ValueError("intra_tp_mode must be 'direct' or 'conservative'")
        if self.bbox_aware_mode not in ("direct", "linearization"):
            raise ValueError("bbox_aware_mode must be 'direct' or 'linearization'")


@torch.no_grad()
def compute_action_local_prototypes(
    state: ROIActionState,
    num_classes: int,
    *,
    min_samples: int = 1,
    iou_floor: float = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-class prototypes from ``ROIActionState``.

    Args:
        state: proposal-aligned ROI state.
        num_classes: total number of classes including background (0).
        min_samples: minimum number of samples to form a prototype.
        iou_floor: if > 0, only use proposals whose matched IoU is at least this.
        normalize: if True, L2-normalize non-empty prototypes.

    Returns:
        ``(num_classes, D)`` prototype tensor.  Empty classes have zero vectors.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    device = state.features.device
    dtype = state.features.dtype
    dim = state.feature_dim
    prototypes = state.features.new_zeros(num_classes, dim)
    counts = state.features.new_zeros(num_classes, dtype=torch.long)

    valid = state.labels >= 0
    if iou_floor > 0.0 and state.ious is not None:
        valid = valid & (state.ious >= iou_floor)

    features = state.features
    labels = state.labels
    for c in range(num_classes):
        mask = valid & (labels == c)
        count = int(mask.sum().item())
        if count >= min_samples:
            prototypes[c] = features[mask].mean(dim=0)
            counts[c] = count

    if normalize:
        nonzero = counts > 0
        if nonzero.any():
            prototypes[nonzero] = F.normalize(prototypes[nonzero], dim=-1)
    return prototypes


@torch.no_grad()
def classify_error_modes(
    state: ROIActionState,
    gt_labels: torch.Tensor,
    *,
    iou_threshold: float = 0.5,
    high_iou_threshold: float = 0.7,
    ambiguous_iou_upper: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Classify proposals into TP / CLS_ERR / LOC_ERR / pure_BG / ambiguous_BG.

    The classification mirrors the old ``detection_feature_manifold_loss``
    sample-type taxonomy but uses the matched IoUs already present in
    ``ROIActionState``.

    Args:
        state: proposal-aligned ROI state.  Must contain ``ious``.
        gt_labels: ``(B,)`` ground-truth class label of the nearest GT box.
            0 = background / no match.
        iou_threshold: IoU above which a foreground prediction is considered
            matched for TP/CLS_ERR/LOC_ERR purposes.
        high_iou_threshold: IoU above which a correct-class prediction is TP.
        ambiguous_iou_upper: upper IoU bound for "pure" background.

    Returns:
        Dict of boolean masks keyed by sample type.
    """
    if gt_labels.shape != state.labels.shape:
        raise ValueError(
            f"gt_labels shape {gt_labels.shape} does not match state.labels {state.labels.shape}"
        )
    if state.ious is None:
        raise ValueError("state.ious is required for error-mode classification")

    batch = state.batch_size
    device = state.features.device
    pred_labels = state.labels
    ious = state.ious

    pred_fg_mask = pred_labels >= 1
    pred_bg_mask = pred_labels == 0

    # TP: foreground prediction, correct class, high IoU.
    tp_mask = pred_fg_mask & (pred_labels == gt_labels) & (ious >= high_iou_threshold)

    # CLS_ERR: foreground prediction, wrong class, IoU above threshold.
    cls_err_mask = pred_fg_mask & (pred_labels != gt_labels) & (ious >= iou_threshold)

    # LOC_ERR: background prediction with high IoU to a foreground GT.
    loc_err_mask = pred_bg_mask & (gt_labels >= 1) & (ious >= iou_threshold)

    # Pure background: background prediction, low IoU.
    pure_bg_mask = pred_bg_mask & (ious < ambiguous_iou_upper)

    # Ambiguous background: background prediction, moderate IoU.
    ambig_bg_mask = (
        pred_bg_mask & (ious >= ambiguous_iou_upper) & (ious < iou_threshold)
    )

    # Sanity: masks are mutually exclusive.
    overlap = tp_mask | cls_err_mask | loc_err_mask | pure_bg_mask | ambig_bg_mask
    # Unclassified samples are typically foreground predictions below threshold.
    unclassified = ~overlap

    return {
        "tp": tp_mask,
        "cls_err": cls_err_mask,
        "loc_err": loc_err_mask,
        "pure_bg": pure_bg_mask,
        "ambig_bg": ambig_bg_mask,
        "unclassified": unclassified,
    }


def _normalize_features(
    features: torch.Tensor, normalize: bool
) -> torch.Tensor:
    return F.normalize(features, dim=-1) if normalize else features


def _maybe_zero(
    features: torch.Tensor, condition: bool
) -> torch.Tensor:
    if condition:
        return features.new_tensor(0.0)
    return features.sum() * 0.0


def _active_features(
    state: ROIActionState,
    actions: ROITransportActions,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized current features and post-action features."""
    z = _normalize_features(state.features, normalize)
    z_new = z + actions.feature_delta
    return z, z_new


def intra_tp_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    prototypes: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    lambda_intra: float = 1.0,
    normalize: bool = True,
    mode: str = "direct",
) -> torch.Tensor:
    """Action-local version of the old ``intra_tp`` loss.

    ``direct`` mode penalizes the squared distance from the post-action
    feature to its class prototype (same spirit as the old loss, but applied
    to ``z + feature_delta``).  ``conservative`` mode only penalizes actions
    that move the feature *farther* from the prototype.
    """
    if lambda_intra <= 0.0 or not masks["tp"].any():
        return _maybe_zero(state.features, True)

    tp_mask = masks["tp"]
    z, z_new = _active_features(state, actions, normalize)
    proto = _normalize_features(prototypes, normalize)

    y = state.labels[tp_mask].long()
    valid = (y >= 1) & (y < proto.shape[0])
    if not valid.any():
        return _maybe_zero(state.features, True)

    y = y[valid]
    z_tp = z[tp_mask][valid]
    z_new_tp = z_new[tp_mask][valid]
    mu = proto[y]

    if mode == "direct":
        loss = ((z_new_tp - mu) ** 2).sum(dim=-1).mean()
    else:  # conservative
        dist_before = (z_tp - mu).norm(dim=-1)
        dist_after = (z_new_tp - mu).norm(dim=-1)
        loss = F.relu(dist_after - dist_before).pow(2).mean()

    return float(lambda_intra) * loss


def fg_bg_sep_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    prototypes: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    lambda_sep: float = 1.0,
    margin: float = 0.5,
    margin_ambig: float | None = None,
    ambig_iou_tau: float = 0.3,
    normalize: bool = True,
) -> torch.Tensor:
    """Action-local foreground-background separation.

    Background samples should be pushed away from the nearest foreground
    prototype.  Ambiguous background (IoU near threshold) uses a softened
    margin.
    """
    bg_mask = masks["pure_bg"] | masks["ambig_bg"]
    if lambda_sep <= 0.0 or not bg_mask.any():
        return _maybe_zero(state.features, True)

    if state.ious is None:
        return _maybe_zero(state.features, True)

    z, z_new = _active_features(state, actions, normalize)
    proto = _normalize_features(prototypes, normalize)
    fg_centers = proto[1:]  # drop background

    z_bg = z_new[bg_mask]
    ious_bg = state.ious[bg_mask]
    ambig_local = masks["ambig_bg"][bg_mask]

    dist_to_fg = torch.cdist(z_bg, fg_centers, p=2)
    d_min = dist_to_fg.min(dim=1).values

    m_fg = float(margin)
    margins = torch.full_like(d_min, m_fg)
    if margin_ambig is not None and ambig_local.any():
        soft = torch.clamp(ious_bg[ambig_local] / float(ambig_iou_tau), 0.0, 1.0)
        margins[ambig_local] = m_fg - (m_fg - float(margin_ambig)) * soft

    loss = F.relu(margins - d_min).pow(2).mean()
    return float(lambda_sep) * loss


def cls_err_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    prototypes: torch.Tensor,
    masks: dict[str, torch.Tensor],
    gt_labels: torch.Tensor,
    *,
    lambda_cls: float = 1.0,
    margin: float = 0.3,
    normalize: bool = True,
) -> torch.Tensor:
    """Action-local classification-error correction.

    Pull the post-action feature toward the true class prototype and push it
    away from the predicted (wrong) class prototype.
    """
    if lambda_cls <= 0.0 or not masks["cls_err"].any():
        return _maybe_zero(state.features, True)

    mask = masks["cls_err"]
    z, z_new = _active_features(state, actions, normalize)
    proto = _normalize_features(prototypes, normalize)
    fg_centers = proto[1:]
    num_fg = fg_centers.shape[0]

    y_true = gt_labels[mask].long() - 1
    y_pred = state.labels[mask].long() - 1
    valid = ((y_true >= 0) & (y_true < num_fg)) & ((y_pred >= 0) & (y_pred < num_fg))
    if not valid.any():
        return _maybe_zero(state.features, True)

    z_err = z_new[mask][valid]
    mu_true = fg_centers[y_true[valid]]
    mu_pred = fg_centers[y_pred[valid]]

    pull = ((z_err - mu_true) ** 2).sum(dim=-1)
    d_wrong = (z_err - mu_pred).norm(dim=-1)
    push = F.relu(float(margin) - d_wrong).pow(2)
    loss = (pull + push).mean()
    return float(lambda_cls) * loss


def loc_err_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    prototypes: torch.Tensor,
    masks: dict[str, torch.Tensor],
    bg_prototype: torch.Tensor,
    *,
    lambda_loc: float = 1.0,
    margin: float = 0.2,
    normalize: bool = True,
) -> torch.Tensor:
    """Action-local localization-error regularizer (Plan A migrated).

    High-IoU background / weakly-localized proposals should be pushed toward
    the background prototype and away from the nearest foreground center.
    """
    if lambda_loc <= 0.0 or not masks["loc_err"].any():
        return _maybe_zero(state.features, True)

    z, z_new = _active_features(state, actions, normalize)
    proto = _normalize_features(prototypes, normalize)
    fg_centers = proto[1:]
    mu_bg = _normalize_features(bg_prototype, normalize).detach()

    z_loc = z_new[masks["loc_err"]]
    d_bg = (z_loc - mu_bg.unsqueeze(0)).norm(dim=-1)
    dist_to_fg = torch.cdist(z_loc, fg_centers, p=2)
    d_fg = dist_to_fg.min(dim=1).values

    loss = F.relu(d_bg - d_fg + float(margin)).pow(2).mean()
    return float(lambda_loc) * loss


def bg_proto_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    masks: dict[str, torch.Tensor],
    bg_prototype: torch.Tensor,
    *,
    lambda_bg: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Pull pure-background samples toward the background prototype."""
    if lambda_bg <= 0.0 or not masks["pure_bg"].any():
        return _maybe_zero(state.features, True)

    z, z_new = _active_features(state, actions, normalize)
    mu_bg = _normalize_features(bg_prototype, normalize).detach()
    z_bg = z_new[masks["pure_bg"]]
    loss = ((z_bg - mu_bg.unsqueeze(0)) ** 2).sum(dim=-1).mean()
    return float(lambda_bg) * loss


def feat_preserve_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    teacher_features: torch.Tensor,
    *,
    lambda_preserve: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Keep post-action features close to a frozen teacher."""
    if lambda_preserve <= 0.0:
        return _maybe_zero(state.features, True)

    z, z_new = _active_features(state, actions, normalize)
    z_teacher = _normalize_features(teacher_features.detach(), normalize)
    loss = F.l1_loss(z_new, z_teacher)
    return float(lambda_preserve) * loss


def bbox_aware_direct_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    *,
    lambda_bbox: float = 1.0,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Directly bound ``box_delta`` on high-IoU proposals.

    This is the migrated Plan B idea made active: good localizations should
    not be perturbed strongly.  The penalty is weighted by matched IoU so that
    higher-IoU boxes are protected more strongly.
    """
    if lambda_bbox <= 0.0 or state.ious is None:
        return _maybe_zero(state.features, True)

    fg_mask = state.labels >= 1
    high_iou = state.ious >= iou_threshold
    mask = fg_mask & high_iou
    if not mask.any():
        return _maybe_zero(state.features, True)

    weights = state.ious[mask]
    box_deltas = actions.box_delta[mask]
    loss = (weights[:, None] * (box_deltas ** 2)).sum() / weights.sum().clamp_min(1.0)
    return float(lambda_bbox) * loss


def bbox_aware_linearization_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    prototypes: torch.Tensor,
    masks: dict[str, torch.Tensor],
    bbox_pred_weight: torch.Tensor,
    num_classes: int,
    *,
    lambda_bbox: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Old Plan B style: penalize feature moves that change bbox predictions.

    Uses the detector's ``bbox_pred.weight`` matrix to linearly approximate
    the box-delta induced by ``feature_delta`` for TP / CLS_ERR samples.
    """
    if lambda_bbox <= 0.0 or bbox_pred_weight is None:
        return _maybe_zero(state.features, True)

    aware_mask = masks["tp"] | masks["cls_err"]
    if not aware_mask.any():
        return _maybe_zero(state.features, True)

    z = _normalize_features(state.features, normalize)
    proto = _normalize_features(prototypes, normalize)
    fg_centers = proto[1:]
    num_fg = fg_centers.shape[0]

    y_aware = state.labels[aware_mask].long() - 1
    valid = (y_aware >= 0) & (y_aware < num_fg)
    if not valid.any():
        return _maybe_zero(state.features, True)

    y_aware = y_aware[valid]
    z_aware = z[aware_mask][valid]
    mu_aware = fg_centers[y_aware]
    delta = z_aware - mu_aware  # (M, D)

    # Class index in bbox_pred includes background at 0.
    start = (y_aware + 1) * 4
    coord_idx = start[:, None] + torch.arange(4, device=z.device)[None, :]
    W_blocks = bbox_pred_weight[coord_idx].to(device=z.device, dtype=z.dtype)
    delta_bbox = torch.einsum("mcd,md->mc", W_blocks, delta)
    loss = (delta_bbox ** 2).mean()
    return float(lambda_bbox) * loss


def bbox_aware_action_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    config: GeometricConstraintConfig,
    prototypes: torch.Tensor | None = None,
    masks: dict[str, torch.Tensor] | None = None,
    bbox_pred_weight: torch.Tensor | None = None,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Route to the selected bbox-aware loss implementation."""
    if config.bbox_aware_mode == "direct":
        return bbox_aware_direct_loss(
            state,
            actions,
            lambda_bbox=config.lambda_bbox_aware,
            iou_threshold=config.iou_threshold,
        )
    if prototypes is None or masks is None:
        raise ValueError(
            "linearization bbox_aware_mode requires prototypes and masks"
        )
    return bbox_aware_linearization_loss(
        state,
        actions,
        prototypes,
        masks,
        bbox_pred_weight,
        num_classes,
        lambda_bbox=config.lambda_bbox_aware,
        normalize=config.normalize,
    )


def geometric_transport_loss(
    state: ROIActionState,
    actions: ROITransportActions,
    num_classes: int,
    config: GeometricConstraintConfig,
    *,
    gt_labels: torch.Tensor | None = None,
    prototypes: torch.Tensor | None = None,
    bbox_pred_weight: torch.Tensor | None = None,
    teacher_features: torch.Tensor | None = None,
    bg_prototype: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the full action-local geometric transport loss.

    Args:
        state: proposal-aligned ROI state.
        actions: predicted transport actions.
        num_classes: total classes including background.
        config: geometric constraint configuration.
        gt_labels: optional ground-truth labels.  If ``None``, ``state.labels``
            is used as a fallback (useful for ablations, but not recommended
            for training).
        prototypes: optional pre-computed prototypes.  If ``None``, prototypes
            are computed from ``state``.
        bbox_pred_weight: optional detector bbox regression weight matrix for
            the linearization bbox-aware loss.
        teacher_features: optional frozen teacher features for feature
            preservation.
        bg_prototype: optional background prototype vector.  If ``None``,
            prototype index 0 is used.

    Returns:
        Dict containing ``loss_geo_total`` and detached per-component losses.
    """
    if prototypes is None:
        prototypes = compute_action_local_prototypes(
            state,
            num_classes,
            min_samples=config.min_prototype_samples,
            iou_floor=config.iou_floor,
            normalize=config.normalize,
        )

    _gt_labels = gt_labels if gt_labels is not None else state.labels
    masks = classify_error_modes(
        state,
        _gt_labels,
        iou_threshold=config.iou_threshold,
        high_iou_threshold=config.high_iou_threshold,
        ambiguous_iou_upper=config.ambiguous_iou_upper,
    )

    _bg_prototype = (
        bg_prototype if bg_prototype is not None else prototypes[0]
    )

    margin_ambig = (
        config.margin_ambig if config.margin_ambig is not None else config.margin_fg_bg
    )

    loss_intra = intra_tp_action_loss(
        state,
        actions,
        prototypes,
        masks,
        lambda_intra=config.lambda_intra_tp,
        normalize=config.normalize,
        mode=config.intra_tp_mode,
    )
    loss_sep = fg_bg_sep_action_loss(
        state,
        actions,
        prototypes,
        masks,
        lambda_sep=config.lambda_fg_bg_sep,
        margin=config.margin_fg_bg,
        margin_ambig=margin_ambig,
        ambig_iou_tau=config.ambig_iou_tau,
        normalize=config.normalize,
    )
    loss_cls = cls_err_action_loss(
        state,
        actions,
        prototypes,
        masks,
        _gt_labels,
        lambda_cls=config.lambda_cls_err,
        margin=config.margin_cls,
        normalize=config.normalize,
    )
    loss_loc = loc_err_action_loss(
        state,
        actions,
        prototypes,
        masks,
        _bg_prototype,
        lambda_loc=config.lambda_loc_err,
        margin=config.margin_loc,
        normalize=config.normalize,
    )
    loss_bg = bg_proto_action_loss(
        state,
        actions,
        masks,
        _bg_prototype,
        lambda_bg=config.lambda_bg_proto,
        normalize=config.normalize,
    )
    loss_preserve = feat_preserve_action_loss(
        state,
        actions,
        teacher_features,
        lambda_preserve=config.lambda_feat_preserve,
        normalize=config.normalize,
    ) if teacher_features is not None else state.features.new_tensor(0.0)

    loss_bbox = bbox_aware_action_loss(
        state,
        actions,
        config,
        prototypes=prototypes,
        masks=masks,
        bbox_pred_weight=bbox_pred_weight,
        num_classes=num_classes,
    )

    total = (
        loss_intra
        + loss_sep
        + loss_cls
        + loss_loc
        + loss_bg
        + loss_preserve
        + loss_bbox
    )
    total = float(config.lambda_geo_total) * total

    return {
        "loss_geo_total": total,
        "loss_geo_intra_tp": loss_intra.detach(),
        "loss_geo_fg_bg_sep": loss_sep.detach(),
        "loss_geo_cls_err": loss_cls.detach(),
        "loss_geo_loc_err": loss_loc.detach(),
        "loss_geo_bg_proto": loss_bg.detach(),
        "loss_geo_feat_preserve": loss_preserve.detach(),
        "loss_geo_bbox_aware": loss_bbox.detach(),
    }
