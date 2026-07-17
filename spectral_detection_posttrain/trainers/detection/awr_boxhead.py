"""Runtime primitives for oracle-utility-weighted box-head fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import random
from typing import Any

import torch


BOX_LOSS_KEYS = ("loss_classifier", "loss_box_reg")
TRAINING_ARMS = ("U", "W", "F", "S")
EXPECTED_CACHE_FORMAT = "re_roi_counterfactual_v1"
EXPECTED_ACTION_FAMILY_HASH = "re_roi_action_family_v2_9_non_identity_002_step"
EXPECTED_TRANSFORM_SIZE = 480
EXPECTED_UTILITY_DEFINITION = "raw_locked_scalarization_v1"
EXPECTED_COORDINATE_SPACE = "detector_transform"
EXPECTED_TOP_K_CANDIDATES = 3
EXPECTED_ACTION_FAMILIES = frozenset(
    {
        "identity_permutation",
        "score_down",
        "score_up",
        "translate_left",
        "translate_right",
        "translate_up",
        "translate_down",
        "scale_down",
        "scale_up",
        "drop",
    }
)


def configure_box_head_only(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Freeze the detector except for its RoI box head and predictor."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    try:
        box_head = model.roi_heads.box_head
        box_predictor = model.roi_heads.box_predictor
    except AttributeError as error:
        raise ValueError("model must expose roi_heads.box_head and box_predictor") from error
    for parameter in box_head.parameters():
        parameter.requires_grad = True
    for parameter in box_predictor.parameters():
        parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("box-head-only configuration produced no trainable parameters")
    return trainable


def configure_native_postprocess(model: torch.nn.Module) -> None:
    """Set the exact native postprocessing values locked by the protocol."""
    try:
        roi_heads = model.roi_heads
    except AttributeError as error:
        raise ValueError("model must expose roi_heads") from error
    roi_heads.score_thresh = 0.05
    roi_heads.nms_thresh = 0.50
    roi_heads.detections_per_img = 100


def weighted_box_loss(
    model: torch.nn.Module,
    images: Sequence[torch.Tensor],
    targets: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[float],
) -> torch.Tensor:
    """Return ``mean_i(weight_i * box_loss_i)`` for one logical batch.

    Each detector forward contains exactly one image so TorchVision cannot
    aggregate proposal losses across images before the locked image weight is
    applied. RPN losses are intentionally excluded because only the box head
    and box predictor are trainable in this protocol.
    """
    if not images:
        raise ValueError("logical batch must contain at least one image")
    if len(images) != len(targets) or len(images) != len(weights):
        raise ValueError("expected one weight per image and target")

    weighted_losses: list[torch.Tensor] = []
    for image, target, value in zip(images, targets, weights, strict=True):
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weights must be finite and positive")
        loss_dict = model([image], [dict(target)])
        if not isinstance(loss_dict, Mapping):
            raise RuntimeError("training detector must return a loss mapping")
        missing = [key for key in BOX_LOSS_KEYS if key not in loss_dict]
        if missing:
            raise RuntimeError(f"detector loss mapping is missing box losses: {missing}")
        box_loss = sum(loss_dict[key] for key in BOX_LOSS_KEYS)
        if not torch.is_tensor(box_loss) or box_loss.ndim != 0:
            raise RuntimeError("per-image box loss must be a scalar tensor")
        if not bool(torch.isfinite(box_loss).item()):
            raise RuntimeError("per-image box loss is non-finite")
        weighted_losses.append(box_loss * box_loss.new_tensor(weight))

    return torch.stack(weighted_losses).sum() / len(weighted_losses)


def build_epoch_image_schedule(
    fit_image_ids: Sequence[int],
    *,
    arm: str,
    seed: int,
    epoch: int,
    positive_image_ids: Sequence[int] | None = None,
) -> list[int]:
    """Build the exact per-epoch image-exposure schedule for one arm."""
    image_ids = [int(image_id) for image_id in fit_image_ids]
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("fit_image_ids must be non-empty and unique")
    if arm not in TRAINING_ARMS:
        raise ValueError(f"unknown AWR arm: {arm}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("epoch must be a positive integer")

    generator = random.Random(int(seed) * 1_000_003 + int(epoch))
    if arm in {"U", "W", "S"}:
        schedule = sorted(image_ids)
        generator.shuffle(schedule)
        return schedule

    positives = [] if positive_image_ids is None else [int(value) for value in positive_image_ids]
    if not positives or len(positives) != len(set(positives)):
        raise ValueError("F arm requires unique positive_image_ids")
    if not set(positives).issubset(image_ids):
        raise ValueError("positive_image_ids must be a subset of fit_image_ids")
    positives.sort()
    return [generator.choice(positives) for _ in image_ids]


def frozen_state_hashes(model: torch.nn.Module) -> dict[str, str]:
    """Hash frozen backbone and RPN parameters plus buffers."""
    try:
        modules = {"backbone": model.backbone, "rpn": model.rpn}
    except AttributeError as error:
        raise ValueError("model must expose backbone and rpn") from error
    return {name: _module_state_hash(module) for name, module in modules.items()}


def validate_re_roi_cache(
    payload: Mapping[str, Any],
    *,
    fit_image_ids: Sequence[int],
    split_manifest_sha256: str,
    checkpoint_sha256: str,
    annotation_sha256: str,
) -> list[Mapping[str, Any]]:
    """Validate the exact cache contract consumed by weighted training."""
    if not isinstance(payload, Mapping):
        raise ValueError("re-ROI cache payload must be a mapping")
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, Mapping) or not isinstance(records, list):
        raise ValueError("re-ROI cache must contain metadata and records")
    expected_metadata = {
        "format": (EXPECTED_CACHE_FORMAT, "cache format"),
        "split_name": ("fit", "cache split"),
        "split_manifest_sha256": (split_manifest_sha256, "split manifest"),
        "checkpoint_sha256": (checkpoint_sha256, "checkpoint"),
        "annotation_sha256": (annotation_sha256, "annotation"),
        "action_family_hash": (EXPECTED_ACTION_FAMILY_HASH, "action family"),
        "utility_definition": (EXPECTED_UTILITY_DEFINITION, "utility definition"),
        "coordinate_space": (EXPECTED_COORDINATE_SPACE, "coordinate space"),
    }
    for key, (expected, label) in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"re-ROI cache {label} mismatch")
    if not isinstance(metadata.get("git_commit"), str) or not metadata["git_commit"]:
        raise ValueError("re-ROI cache Git commit is missing")

    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping) or any(
        int(model_config.get(key, -1)) != EXPECTED_TRANSFORM_SIZE
        for key in ("min_size", "max_size")
    ):
        raise ValueError("re-ROI cache transform size mismatch")
    cache_config = metadata.get("cache_config")
    locked_postprocess = {
        "score_threshold": 0.05,
        "nms_threshold": 0.5,
        "detections_per_img": 100,
        "top_k_candidates": EXPECTED_TOP_K_CANDIDATES,
    }
    if not isinstance(cache_config, Mapping) or any(
        float(cache_config.get(key, float("nan"))) != value
        for key, value in locked_postprocess.items()
    ):
        raise ValueError("re-ROI cache native postprocessing mismatch")

    record_ids: list[int] = []
    for record in records:
        if not isinstance(record, Mapping) or "image_id" not in record:
            raise ValueError("re-ROI cache record is missing image_id")
        record_ids.append(int(record["image_id"]))
    expected_ids = sorted(int(value) for value in fit_image_ids)
    if len(expected_ids) != len(set(expected_ids)) or sorted(record_ids) != expected_ids:
        raise ValueError("re-ROI cache fit image IDs mismatch")
    for record in records:
        _validate_complete_cache_record(record)
    return list(records)


def _validate_complete_cache_record(record: Mapping[str, Any]) -> None:
    baseline = record.get("baseline")
    actions = record.get("actions")
    if not isinstance(baseline, Mapping):
        raise ValueError("re-ROI cache record baseline is missing")
    if not isinstance(actions, list):
        raise ValueError("re-ROI cache record actions are missing")

    required_tensors = ("boxes", "scores", "labels", "proposal_boxes", "roi_features")
    if any(not torch.is_tensor(baseline.get(key)) for key in required_tensors):
        raise ValueError("re-ROI cache baseline tensors are incomplete")
    boxes = baseline["boxes"]
    scores = baseline["scores"]
    labels = baseline["labels"]
    roi_features = baseline["roi_features"]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("re-ROI cache baseline boxes must have shape [N, 4]")
    detection_count = int(boxes.shape[0])
    if (
        scores.ndim != 1
        or labels.ndim != 1
        or len(scores) != detection_count
        or len(labels) != detection_count
    ):
        raise ValueError("re-ROI cache baseline detection tensors disagree")
    if roi_features.ndim < 1 or int(roi_features.shape[0]) != detection_count:
        raise ValueError("re-ROI cache baseline ROI feature count mismatch")

    expected_candidates = set(range(min(EXPECTED_TOP_K_CANDIDATES, detection_count)))
    families_by_candidate: dict[int, set[str]] = {index: set() for index in expected_candidates}
    for row in actions:
        if not isinstance(row, Mapping):
            raise ValueError("re-ROI cache action row must be a mapping")
        try:
            candidate_index = int(row["candidate_index"])
            family = str(row["family"])
            q_teacher = float(row["q_teacher"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("re-ROI cache action row is incomplete") from error
        if candidate_index not in expected_candidates:
            raise ValueError("re-ROI cache action has an unexpected candidate index")
        if family in families_by_candidate[candidate_index]:
            raise ValueError("re-ROI cache candidate has a duplicate action family")
        if family not in EXPECTED_ACTION_FAMILIES or not math.isfinite(q_teacher):
            raise ValueError("re-ROI cache action family or utility is invalid")
        if family == "identity_permutation" and q_teacher != 0.0:
            raise ValueError("re-ROI cache identity utility must be exactly zero")
        families_by_candidate[candidate_index].add(family)

    if any(families != EXPECTED_ACTION_FAMILIES for families in families_by_candidate.values()):
        raise ValueError("re-ROI cache candidate is missing the complete action family")


def assert_frozen_state_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> None:
    if dict(before) != dict(after):
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise AssertionError(f"frozen detector state changed: {changed}")


def _module_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "BOX_LOSS_KEYS",
    "TRAINING_ARMS",
    "EXPECTED_ACTION_FAMILY_HASH",
    "EXPECTED_CACHE_FORMAT",
    "EXPECTED_ACTION_FAMILIES",
    "EXPECTED_TRANSFORM_SIZE",
    "assert_frozen_state_unchanged",
    "build_epoch_image_schedule",
    "configure_box_head_only",
    "configure_native_postprocess",
    "frozen_state_hashes",
    "validate_re_roi_cache",
    "weighted_box_loss",
]
