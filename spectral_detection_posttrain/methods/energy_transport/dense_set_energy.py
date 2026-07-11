"""Dense train-time teacher energy for native detector prediction sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as torch_f
from torchvision.ops import box_iou


@dataclass(frozen=True)
class DenseTeacherConfig:
    coverage_temperature: float = 0.10
    coverage_kappa: float = 10.0
    target_iou: float = 0.75
    background_temperature: float = 0.10
    class_temperature: float = 0.10
    calibration_temperature: float = 0.10
    duplicate_temperature: float = 0.05
    duplicate_nms_reference: float = 0.50
    duplicate_edge_min_iou: float = 0.10
    correctness_iou: float = 0.50
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        temperatures = (
            self.coverage_temperature,
            self.background_temperature,
            self.class_temperature,
            self.calibration_temperature,
            self.duplicate_temperature,
        )
        if any(float(value) <= 0 for value in temperatures):
            raise ValueError("dense teacher temperatures must be positive")
        if float(self.coverage_kappa) <= 0 or float(self.epsilon) <= 0:
            raise ValueError("coverage_kappa and epsilon must be positive")
        for value in (
            self.target_iou,
            self.duplicate_nms_reference,
            self.duplicate_edge_min_iou,
            self.correctness_iou,
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("IoU thresholds must be in [0, 1]")


@dataclass(frozen=True)
class DenseTeacherComponents:
    coverage: torch.Tensor
    background_risk: torch.Tensor
    class_risk: torch.Tensor
    duplicate_risk: torch.Tensor
    calibration_error: torch.Tensor
    prediction_count: int
    ground_truth_count: int
    duplicate_edge_count: int

    def detached_dict(self) -> dict[str, float | int]:
        values = asdict(self)
        return {
            key: float(value.detach().cpu().item()) if torch.is_tensor(value) else int(value)
            for key, value in values.items()
        }


def robust_scalar_summary(
    values: Sequence[float] | torch.Tensor, *, min_iqr: float
) -> dict[str, float | int | bool]:
    """Summarize a scalar image-level component for fit-only standardization."""
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.numel() == 0:
        raise ValueError("values must be non-empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("values must be finite")
    if float(min_iqr) < 0:
        raise ValueError("min_iqr must be non-negative")
    quantiles = torch.quantile(tensor, torch.tensor([0.25, 0.5, 0.75], dtype=tensor.dtype))
    q25, median, q75 = (float(value.item()) for value in quantiles)
    iqr = q75 - q25
    return {
        "count": int(tensor.numel()),
        "min": float(tensor.min().item()),
        "q25": q25,
        "median": median,
        "q75": q75,
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "iqr": iqr,
        "zero_fraction": float(tensor.eq(0).double().mean().item()),
        "unique_rounded_1e6": int(torch.unique(torch.round(tensor * 1e6)).numel()),
        "degenerate": bool(iqr < float(min_iqr)),
    }


def _validate_set(name: str, value: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if "boxes" not in value or "labels" not in value:
        raise ValueError(f"{name} must contain boxes and labels")
    boxes = torch.as_tensor(value["boxes"])
    labels = torch.as_tensor(value["labels"], device=boxes.device)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} boxes must have shape (N, 4)")
    if labels.shape != (boxes.shape[0],):
        raise ValueError(f"{name} labels must have shape (N,)")
    if not boxes.dtype.is_floating_point:
        boxes = boxes.float()
    return boxes, labels.long()


def _canonical_order(
    boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor | None = None
) -> torch.Tensor:
    order = torch.arange(boxes.shape[0], device=boxes.device)
    keys = [labels, boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]]
    if scores is not None:
        keys.append(scores)
    for key in reversed(keys):
        order = order[torch.argsort(key[order], stable=True)]
    return order


def dense_teacher_components(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    config: DenseTeacherConfig | None = None,
) -> DenseTeacherComponents:
    """Compute differentiable raw teacher components for one image.

    Predictions are the detector's native post-NMS set. Ground truth supplies
    train-only supervision and never filters the prediction set.
    """
    cfg = config or DenseTeacherConfig()
    pred_boxes, pred_labels = _validate_set("prediction", prediction)
    gt_boxes, gt_labels = _validate_set("target", target)
    if "scores" not in prediction:
        raise ValueError("prediction must contain scores")
    scores = torch.as_tensor(prediction["scores"], dtype=pred_boxes.dtype, device=pred_boxes.device)
    if scores.shape != (pred_boxes.shape[0],):
        raise ValueError("prediction scores must have shape (N,)")
    if gt_boxes.device != pred_boxes.device:
        gt_boxes = gt_boxes.to(pred_boxes.device)
        gt_labels = gt_labels.to(pred_boxes.device)
    if not torch.isfinite(pred_boxes).all() or not torch.isfinite(gt_boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("dense teacher inputs must be finite")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("prediction scores must be probabilities in [0, 1]")

    prediction_order = _canonical_order(pred_boxes, pred_labels, scores)
    pred_boxes = pred_boxes[prediction_order]
    pred_labels = pred_labels[prediction_order]
    scores = scores[prediction_order]
    target_order = _canonical_order(gt_boxes, gt_labels)
    gt_boxes = gt_boxes[target_order]
    gt_labels = gt_labels[target_order]

    prediction_count = int(pred_boxes.shape[0])
    ground_truth_count = int(gt_boxes.shape[0])
    zero = scores.new_zeros(())
    overlaps = (
        box_iou(pred_boxes, gt_boxes)
        if prediction_count and ground_truth_count
        else scores.new_zeros((prediction_count, ground_truth_count))
    )

    coverage = zero
    if prediction_count and ground_truth_count:
        per_ground_truth = []
        log_scores = scores.clamp_min(float(cfg.epsilon)).log()
        for ground_truth_index in range(ground_truth_count):
            same_class = pred_labels.eq(gt_labels[ground_truth_index])
            if not same_class.any():
                per_ground_truth.append(zero)
                continue
            evidence = log_scores[same_class] + float(cfg.coverage_kappa) * (
                overlaps[same_class, ground_truth_index] - float(cfg.target_iou)
            )
            smooth_max = float(cfg.coverage_temperature) * torch.logsumexp(
                evidence / float(cfg.coverage_temperature), dim=0
            )
            per_ground_truth.append(torch_f.softplus(smooth_max))
        coverage = torch.stack(per_ground_truth).sum()

    if ground_truth_count:
        maximum_any = overlaps.amax(dim=1) if prediction_count else scores.new_zeros((0,))
        maximum_same_rows = []
        for prediction_index in range(prediction_count):
            same_class = gt_labels.eq(pred_labels[prediction_index])
            maximum_same_rows.append(
                overlaps[prediction_index, same_class].amax() if same_class.any() else zero
            )
        maximum_same = torch.stack(maximum_same_rows) if maximum_same_rows else scores.new_zeros((0,))
    else:
        maximum_any = scores.new_zeros((prediction_count,))
        maximum_same = scores.new_zeros((prediction_count,))

    background_gate = torch.sigmoid(
        (float(cfg.correctness_iou) - maximum_any) / float(cfg.background_temperature)
    )
    foreground_overlap_gate = torch.sigmoid(
        (maximum_any - float(cfg.correctness_iou)) / float(cfg.background_temperature)
    )
    wrong_class_gate = torch.sigmoid(
        (float(cfg.correctness_iou) - maximum_same) / float(cfg.class_temperature)
    )
    background_risk = (scores * background_gate).sum()
    class_risk = (scores * foreground_overlap_gate * wrong_class_gate).sum()

    duplicate_risk = zero
    duplicate_edge_count = 0
    if prediction_count > 1:
        prediction_overlaps = box_iou(pred_boxes, pred_boxes)
        upper = torch.triu(torch.ones_like(prediction_overlaps, dtype=torch.bool), diagonal=1)
        same_class = pred_labels[:, None].eq(pred_labels[None, :])
        edges = upper & same_class & prediction_overlaps.ge(float(cfg.duplicate_edge_min_iou))
        duplicate_edge_count = int(edges.sum().item())
        if duplicate_edge_count:
            score_products = scores[:, None] * scores[None, :]
            duplicate_risk = (
                score_products[edges]
                * torch.sigmoid(
                    (prediction_overlaps[edges] - float(cfg.duplicate_nms_reference))
                    / float(cfg.duplicate_temperature)
                )
            ).sum()

    soft_correctness = torch.sigmoid(
        (maximum_same - float(cfg.correctness_iou)) / float(cfg.calibration_temperature)
    )
    calibration_error = (scores - soft_correctness).square().sum()
    return DenseTeacherComponents(
        coverage=coverage,
        background_risk=background_risk,
        class_risk=class_risk,
        duplicate_risk=duplicate_risk,
        calibration_error=calibration_error,
        prediction_count=prediction_count,
        ground_truth_count=ground_truth_count,
        duplicate_edge_count=duplicate_edge_count,
    )
