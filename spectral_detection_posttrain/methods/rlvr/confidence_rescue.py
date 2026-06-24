from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.core.matching.box_iou import box_iou


@dataclass(frozen=True)
class ConfidenceRescueConfig:
    low_conf_max: float = 0.5
    high_conf_min: float = 0.7
    high_iou_min: float = 0.75
    low_iou_max: float = 0.3
    positive_weight: float = 1.0
    negative_weight: float = 0.25
    include_low_conf_negatives: bool = False
    verifier_positive_min: float | None = None
    verifier_hard_negative_min: float | None = None
    verifier_weight_mode: str = "hard"
    verifier_weight_temperature: float = 1.0


@dataclass(frozen=True)
class ConfidenceRescueTargets:
    target_labels: torch.Tensor
    weights: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor


@dataclass(frozen=True)
class ManifoldGateConfig:
    mode: str = "density_ratio"
    k: int = 5
    fp_weight: float = 1.0
    hard_negative_weight: float = 0.0
    margin_weight: float = 0.0
    use_bucket_thresholds: bool = False


@dataclass(frozen=True)
class ManifoldGateReference:
    tp_by_class_bucket: dict[tuple[int, int], torch.Tensor]
    fp_by_class_bucket: dict[tuple[int, int], torch.Tensor]
    hard_negative_by_class_bucket: dict[tuple[int, int], torch.Tensor]
    tp_by_class: dict[int, torch.Tensor]
    thresholds: dict[tuple[int, int], float]
    global_tp: torch.Tensor
    global_fp: torch.Tensor
    num_classes: int
    feature_projection: str = "identity"


@dataclass(frozen=True)
class BestCheckpointConfig:
    selection_metric: str = "ap75"
    higher_is_better: bool = True
    min_delta: float = 0.0
    max_prediction_ratio: float | None = None
    max_prediction_delta: int | None = None
    max_fp_rate_delta: float | None = None
    max_high_conf_fp_rate_delta: float | None = None
    max_ece_delta: float | None = None


def project_manifold_features(features: torch.Tensor, feature_projection: str = "identity") -> torch.Tensor:
    features = features.float()
    if features.ndim != 2:
        return features
    mode = str(feature_projection)
    if mode == "identity":
        return features
    width = max(1, int(features.shape[1]))
    half = max(1, width // 2)
    if mode == "first_half":
        return features[:, :half]
    if mode == "second_half":
        return features[:, half:] if half < width else features[:, :half]
    if mode == "l2":
        return F.normalize(features, p=2, dim=1)
    raise ValueError(f"Unknown manifold feature projection: {feature_projection}")


def confidence_iou_region_masks(
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    low_conf_scores: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    scores = scores.float()
    best_iou = best_iou.to(scores.device).float()
    low_source = scores if low_conf_scores is None else low_conf_scores.to(scores.device).float()
    low_conf = low_source <= float(cfg.low_conf_max)
    high_conf = scores >= float(cfg.high_conf_min)
    high_iou = best_iou >= float(cfg.high_iou_min)
    low_iou = best_iou <= float(cfg.low_iou_max)
    return {
        "low_conf_high_iou": low_conf & high_iou,
        "high_conf_low_iou": high_conf & low_iou,
        "low_conf_low_iou": low_conf & low_iou,
        "high_conf_high_iou": high_conf & high_iou,
    }


def summarize_confidence_iou_regions(
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    low_conf_scores: torch.Tensor | None = None,
) -> dict[str, float]:
    masks = confidence_iou_region_masks(scores, best_iou, cfg, low_conf_scores=low_conf_scores)
    scores = scores.float()
    best_iou = best_iou.to(scores.device).float()
    summary: dict[str, float] = {
        "proposal_count": int(scores.numel()),
    }
    for name, mask in masks.items():
        count = int(mask.sum().item())
        summary[f"{name}_count"] = count
        summary[f"{name}_rate"] = count / max(1, int(scores.numel()))
        summary[f"{name}_mean_score"] = float(scores[mask].mean().item()) if count else 0.0
        summary[f"{name}_mean_iou"] = float(best_iou[mask].mean().item()) if count else 0.0
    return summary


def scale_bucket_for_boxes(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.long, device=boxes.device)
    height, width = image_size
    area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
    rel_area = area / max(float(height * width), 1.0)
    return torch.where(rel_area < 0.02, torch.zeros_like(rel_area, dtype=torch.long), torch.where(rel_area < 0.12, torch.ones_like(rel_area, dtype=torch.long), torch.full_like(rel_area, 2, dtype=torch.long)))


def _mean_knn_distance(query: torch.Tensor, reference: torch.Tensor, k: int) -> torch.Tensor:
    if query.numel() == 0:
        return query.new_zeros((query.shape[0],))
    if reference.numel() == 0:
        return query.new_full((query.shape[0],), 1e6)
    reference = reference.to(query.device).float()
    distances = torch.cdist(query.float(), reference)
    k_eff = min(max(1, int(k)), reference.shape[0])
    return distances.topk(k_eff, largest=False, dim=1).values.mean(dim=1)


def _stack_or_empty(parts: list[torch.Tensor], feature_dim: int, device: torch.device) -> torch.Tensor:
    if parts:
        return torch.cat(parts, dim=0).detach()
    return torch.empty((0, feature_dim), device=device)


def build_manifold_gate_reference(
    features: torch.Tensor,
    labels: torch.Tensor,
    is_positive: torch.Tensor,
    boxes: torch.Tensor,
    *,
    image_size: tuple[int, int],
    num_classes: int,
    feature_projection: str = "identity",
) -> ManifoldGateReference:
    features = project_manifold_features(features.float(), feature_projection)
    labels = labels.to(features.device).long()
    is_positive = is_positive.to(features.device).bool()
    boxes = boxes.to(features.device).float()
    feature_dim = features.shape[1] if features.ndim == 2 else 0
    buckets = scale_bucket_for_boxes(boxes, image_size)
    tp_by_class_bucket: dict[tuple[int, int], torch.Tensor] = {}
    fp_by_class_bucket: dict[tuple[int, int], torch.Tensor] = {}
    hard_negative_by_class_bucket: dict[tuple[int, int], torch.Tensor] = {}
    tp_by_class: dict[int, torch.Tensor] = {}
    thresholds: dict[tuple[int, int], float] = {}

    all_tp_parts = []
    all_fp_parts = []
    for class_id in range(1, int(num_classes)):
        class_mask = labels == class_id
        class_tp = features[class_mask & is_positive]
        if class_tp.numel() > 0:
            tp_by_class[class_id] = class_tp.detach()
            all_tp_parts.append(class_tp)
        for bucket_id in range(3):
            mask = class_mask & (buckets == bucket_id)
            tp = features[mask & is_positive]
            fp = features[mask & (~is_positive)]
            key = (class_id, bucket_id)
            if tp.numel() > 0:
                tp_by_class_bucket[key] = tp.detach()
            if fp.numel() > 0:
                fp_by_class_bucket[key] = fp.detach()
                hard_negative_by_class_bucket[key] = fp.detach()
                all_fp_parts.append(fp)
            if tp.numel() > 0 or fp.numel() > 0:
                tp_center = tp.mean(dim=0, keepdim=True) if tp.numel() > 0 else features[class_mask].mean(dim=0, keepdim=True)
                tp_scores = -_mean_knn_distance(tp, tp_center, k=1) if tp.numel() > 0 else features.new_empty((0,))
                fp_scores = -_mean_knn_distance(fp, tp_center, k=1) if fp.numel() > 0 else features.new_empty((0,))
                if tp_scores.numel() > 0 and fp_scores.numel() > 0:
                    thresholds[key] = float(((tp_scores.mean() + fp_scores.mean()) * 0.5).item())
                elif tp_scores.numel() > 0:
                    thresholds[key] = float(tp_scores.mean().item())
                else:
                    thresholds[key] = float(fp_scores.mean().item())

    global_tp = _stack_or_empty(all_tp_parts, feature_dim, features.device)
    global_fp = _stack_or_empty(all_fp_parts, feature_dim, features.device)
    return ManifoldGateReference(
        tp_by_class_bucket=tp_by_class_bucket,
        fp_by_class_bucket=fp_by_class_bucket,
        hard_negative_by_class_bucket=hard_negative_by_class_bucket,
        tp_by_class=tp_by_class,
        thresholds=thresholds,
        global_tp=global_tp,
        global_fp=global_fp,
        num_classes=int(num_classes),
        feature_projection=str(feature_projection),
    )


def _lookup_reference(
    mapping: dict[tuple[int, int], torch.Tensor],
    class_id: int,
    bucket_id: int,
    fallback: torch.Tensor,
) -> torch.Tensor:
    return mapping.get((int(class_id), int(bucket_id)), fallback)


def _class_margin_score(
    reference: ManifoldGateReference,
    query: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if query.numel() == 0:
        return query.new_zeros((query.shape[0],))
    scores = []
    for row, class_id in zip(query, labels):
        target_tp = reference.tp_by_class.get(int(class_id), reference.global_tp)
        target_distance = _mean_knn_distance(row.unsqueeze(0), target_tp, k=k)[0]
        other_distances = []
        for other_class, other_tp in reference.tp_by_class.items():
            if int(other_class) == int(class_id):
                continue
            other_distances.append(_mean_knn_distance(row.unsqueeze(0), other_tp, k=k)[0])
        if other_distances:
            second_distance = torch.stack(other_distances).min()
            scores.append(second_distance - target_distance)
        else:
            scores.append(-target_distance)
    return torch.stack(scores).to(query.device)


def score_manifold_gate(
    reference: ManifoldGateReference,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    query_boxes: torch.Tensor,
    *,
    image_size: tuple[int, int],
    cfg: ManifoldGateConfig,
) -> torch.Tensor:
    query_features = project_manifold_features(query_features.float(), reference.feature_projection)
    query_labels = query_labels.to(query_features.device).long()
    query_boxes = query_boxes.to(query_features.device).float()
    buckets = scale_bucket_for_boxes(query_boxes, image_size)
    if cfg.mode == "margin":
        return _class_margin_score(reference, query_features, query_labels, k=int(cfg.k))

    base_scores = query_features.new_empty((query_features.shape[0],))
    for class_int in query_labels.unique().tolist():
        for bucket_int in buckets.unique().tolist():
            mask = (query_labels == int(class_int)) & (buckets == int(bucket_int))
            if not mask.any():
                continue
            rows = query_features[mask]
            class_int = int(class_int)
            bucket_int = int(bucket_int)
            tp_ref = _lookup_reference(reference.tp_by_class_bucket, class_int, bucket_int, reference.global_tp)
            fp_ref = _lookup_reference(reference.fp_by_class_bucket, class_int, bucket_int, reference.global_fp)
            hard_ref = _lookup_reference(reference.hard_negative_by_class_bucket, class_int, bucket_int, reference.global_fp)
            tp_distance = _mean_knn_distance(rows, tp_ref, k=int(cfg.k))
            fp_distance = _mean_knn_distance(rows, fp_ref, k=int(cfg.k))
            hard_distance = _mean_knn_distance(rows, hard_ref, k=int(cfg.k))
            density_score = -tp_distance + float(cfg.fp_weight) * fp_distance - float(cfg.hard_negative_weight) * (1.0 / hard_distance.clamp_min(1e-6))
            if bool(cfg.use_bucket_thresholds):
                density_score = density_score - float(reference.thresholds.get((class_int, bucket_int), 0.0))
            base_scores[mask] = density_score
    if float(cfg.margin_weight) != 0.0:
        base_scores = base_scores + float(cfg.margin_weight) * _class_margin_score(reference, query_features, query_labels, k=int(cfg.k))
    return base_scores


def summarize_verifier_gate(
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    verifier_scores: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    threshold: float,
    low_conf_scores: torch.Tensor | None = None,
) -> dict[str, float]:
    masks = confidence_iou_region_masks(scores, best_iou, cfg, low_conf_scores=low_conf_scores)
    verifier_positive = verifier_scores.to(scores.device).float() >= float(threshold)
    low_source = scores.float() if low_conf_scores is None else low_conf_scores.to(scores.device).float()
    low_conf = low_source <= float(cfg.low_conf_max)
    gated_low_conf = verifier_positive & low_conf
    gated_lchi = verifier_positive & masks["low_conf_high_iou"]
    gated_lcli = verifier_positive & masks["low_conf_low_iou"]
    total_lchi = int(masks["low_conf_high_iou"].sum().item())
    total_lcli = int(masks["low_conf_low_iou"].sum().item())
    gated_lchi_count = int(gated_lchi.sum().item())
    gated_lcli_count = int(gated_lcli.sum().item())
    gated_low_conf_count = int(gated_low_conf.sum().item())
    return {
        "gate_low_conf_high_iou_count": gated_lchi_count,
        "gate_low_conf_low_iou_count": gated_lcli_count,
        "gate_low_conf_total_count": gated_low_conf_count,
        "gate_low_conf_high_iou_recall": gated_lchi_count / max(1, total_lchi),
        "gate_low_conf_low_iou_recall": gated_lcli_count / max(1, total_lcli),
        "gate_low_conf_precision": gated_lchi_count / max(1, gated_lchi_count + gated_lcli_count),
        "gate_low_conf_false_rescue_rate": gated_lcli_count / max(1, gated_low_conf_count),
    }


def summarize_confidence_rescue_effect(
    baseline_probs: torch.Tensor,
    current_probs: torch.Tensor,
    lchi_mask: torch.Tensor,
    *,
    verifier_positive_mask: torch.Tensor | None = None,
    score_threshold: float,
    low_conf_max: float,
) -> dict[str, float]:
    baseline_probs = baseline_probs.float()
    current_probs = current_probs.to(baseline_probs.device).float()
    lchi = lchi_mask.to(baseline_probs.device).bool()
    if verifier_positive_mask is None:
        verifier_positive = torch.zeros_like(lchi)
    else:
        verifier_positive = verifier_positive_mask.to(baseline_probs.device).bool() & lchi

    def summarize(prefix: str, mask: torch.Tensor) -> dict[str, float]:
        count = int(mask.sum().item())
        if count == 0:
            return {
                f"{prefix}_count": 0,
                f"{prefix}_baseline_prob_sum": 0.0,
                f"{prefix}_current_prob_sum": 0.0,
                f"{prefix}_delta_sum": 0.0,
                f"{prefix}_delta_mean": 0.0,
                f"{prefix}_cross_score_threshold_count": 0,
                f"{prefix}_cross_score_threshold_rate": 0.0,
                f"{prefix}_cross_low_conf_max_count": 0,
                f"{prefix}_cross_low_conf_max_rate": 0.0,
            }
        selected_baseline = baseline_probs[mask]
        selected_current = current_probs[mask]
        delta = selected_current - selected_baseline
        cross_score = (selected_baseline < float(score_threshold)) & (selected_current >= float(score_threshold))
        cross_low_conf = (selected_baseline <= float(low_conf_max)) & (selected_current > float(low_conf_max))
        cross_score_count = int(cross_score.sum().item())
        cross_low_conf_count = int(cross_low_conf.sum().item())
        return {
            f"{prefix}_count": count,
            f"{prefix}_baseline_prob_sum": float(selected_baseline.sum().item()),
            f"{prefix}_current_prob_sum": float(selected_current.sum().item()),
            f"{prefix}_delta_sum": float(delta.sum().item()),
            f"{prefix}_delta_mean": float(delta.mean().item()),
            f"{prefix}_cross_score_threshold_count": cross_score_count,
            f"{prefix}_cross_score_threshold_rate": cross_score_count / max(1, count),
            f"{prefix}_cross_low_conf_max_count": cross_low_conf_count,
            f"{prefix}_cross_low_conf_max_rate": cross_low_conf_count / max(1, count),
        }

    summary = summarize("lchi_conf", lchi)
    summary.update(summarize("verifier_positive_lchi_conf", verifier_positive))
    return summary


def calibrate_classwise_thresholds(
    verifier_scores: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    min_precision: float,
    fallback_threshold: float = 0.0,
    min_positives: int = 1,
    min_threshold: float | None = None,
    low_conf_scores: torch.Tensor | None = None,
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    verifier_scores = verifier_scores.float()
    labels = labels.to(verifier_scores.device).long()
    scores = scores.to(verifier_scores.device).float()
    best_iou = best_iou.to(verifier_scores.device).float()
    low_source = scores if low_conf_scores is None else low_conf_scores.to(verifier_scores.device).float()
    low_conf = low_source <= float(cfg.low_conf_max)
    positive = low_conf & (best_iou >= float(cfg.high_iou_min)) & (labels > 0)
    negative = low_conf & (best_iou <= float(cfg.low_iou_max)) & (labels > 0)
    thresholds: dict[int, float] = {}
    diagnostics: dict[int, dict[str, float]] = {}

    for class_id in labels[labels > 0].unique().tolist():
        class_int = int(class_id)
        class_mask = labels == class_int
        candidate_mask = class_mask & low_conf & (positive | negative)
        class_positive_count = int((class_mask & positive).sum().item())
        if class_positive_count < int(min_positives) or not candidate_mask.any():
            thresholds[class_int] = float(fallback_threshold)
            diagnostics[class_int] = {
                "threshold": float(fallback_threshold),
                "precision": 0.0,
                "recall": 0.0,
                "selected": 0,
                "positive_count": class_positive_count,
            }
            continue

        class_scores = verifier_scores[candidate_mask]
        class_positive = positive[candidate_mask]
        best = None
        for threshold in torch.sort(class_scores.unique(), descending=True).values.tolist():
            if min_threshold is not None and float(threshold) < float(min_threshold):
                continue
            selected = class_scores >= float(threshold)
            selected_count = int(selected.sum().item())
            tp_count = int((selected & class_positive).sum().item())
            precision = tp_count / max(1, selected_count)
            recall = tp_count / max(1, class_positive_count)
            if precision < float(min_precision):
                continue
            candidate = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "selected": selected_count,
                "positive_count": class_positive_count,
                "tp_count": tp_count,
            }
            if best is None or (candidate["recall"], candidate["selected"]) > (best["recall"], best["selected"]):
                best = candidate

        if best is None:
            thresholds[class_int] = float(fallback_threshold)
            diagnostics[class_int] = {
                "threshold": float(fallback_threshold),
                "precision": 0.0,
                "recall": 0.0,
                "selected": 0,
                "positive_count": class_positive_count,
            }
        else:
            thresholds[class_int] = float(best["threshold"])
            diagnostics[class_int] = best
    return thresholds, diagnostics


def _auc_from_binary_scores(scores: torch.Tensor, positive: torch.Tensor) -> float:
    scores = scores.float()
    positive = positive.to(scores.device).bool()
    pos_scores = scores[positive]
    neg_scores = scores[~positive]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    comparisons = pos_scores[:, None] - neg_scores[None, :]
    wins = (comparisons > 0).float().sum()
    ties = (comparisons == 0).float().sum() * 0.5
    return float(((wins + ties) / comparisons.numel()).item())


def evaluate_verifier_offline(
    verifier_scores: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    threshold: float,
    precision_targets: tuple[float, ...] = (0.7, 0.8, 0.9),
    low_conf_scores: torch.Tensor | None = None,
) -> dict[str, float]:
    verifier_scores = verifier_scores.float()
    labels = labels.to(verifier_scores.device).long()
    scores = scores.to(verifier_scores.device).float()
    best_iou = best_iou.to(verifier_scores.device).float()
    low_source = scores if low_conf_scores is None else low_conf_scores.to(verifier_scores.device).float()
    low_conf = low_source <= float(cfg.low_conf_max)
    positive = low_conf & (best_iou >= float(cfg.high_iou_min)) & (labels > 0)
    negative = low_conf & (best_iou <= float(cfg.low_iou_max)) & (labels > 0)
    candidate = positive | negative
    candidate_scores = verifier_scores[candidate]
    candidate_positive = positive[candidate]
    selected = candidate_scores >= float(threshold)
    selected_count = int(selected.sum().item())
    selected_positive = int((selected & candidate_positive).sum().item())
    selected_negative = selected_count - selected_positive
    positive_count = int(candidate_positive.sum().item())
    negative_count = int((~candidate_positive).sum().item())

    report: dict[str, float] = {
        "candidate_count": int(candidate.sum().item()),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "auc": _auc_from_binary_scores(candidate_scores, candidate_positive) if candidate_scores.numel() else 0.0,
        "threshold": float(threshold),
        "selected_at_threshold": selected_count,
        "precision_at_threshold": selected_positive / max(1, selected_count),
        "recall_at_threshold": selected_positive / max(1, positive_count),
        "false_rescue_rate_at_threshold": selected_negative / max(1, selected_count),
    }

    if candidate_scores.numel() == 0:
        for target_precision in precision_targets:
            report[f"recall_at_precision_{target_precision:g}"] = 0.0
        return report

    thresholds = torch.sort(candidate_scores.unique(), descending=True).values.tolist()
    for target_precision in precision_targets:
        best_recall = 0.0
        best_threshold = 0.0
        for candidate_threshold in thresholds:
            chosen = candidate_scores >= float(candidate_threshold)
            chosen_count = int(chosen.sum().item())
            true_positive = int((chosen & candidate_positive).sum().item())
            precision = true_positive / max(1, chosen_count)
            recall = true_positive / max(1, positive_count)
            if precision >= float(target_precision) and recall >= best_recall:
                best_recall = recall
                best_threshold = float(candidate_threshold)
        report[f"recall_at_precision_{target_precision:g}"] = float(best_recall)
        report[f"threshold_at_precision_{target_precision:g}"] = float(best_threshold)
    return report


def match_boxes_to_targets(
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = boxes.float()
    device = boxes.device
    if boxes.numel() == 0:
        return boxes.new_zeros((0,)), torch.zeros((0,), dtype=torch.long, device=device)
    if gt_boxes.numel() == 0:
        return boxes.new_zeros((boxes.shape[0],)), torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)
    iou_matrix = box_iou(boxes, gt_boxes.to(device).float())
    best_iou, best_indices = iou_matrix.max(dim=1)
    labels = gt_labels.to(device).long()[best_indices]
    labels = torch.where(best_iou > 0.0, labels, torch.zeros_like(labels))
    return best_iou, labels


def match_boxes_to_target_boxes(
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = boxes.float()
    device = boxes.device
    if boxes.numel() == 0:
        return (
            boxes.new_zeros((0,)),
            torch.zeros((0,), dtype=torch.long, device=device),
            boxes.new_zeros((0, 4)),
        )
    if gt_boxes.numel() == 0:
        return (
            boxes.new_zeros((boxes.shape[0],)),
            torch.zeros((boxes.shape[0],), dtype=torch.long, device=device),
            boxes.new_zeros((boxes.shape[0], 4)),
        )
    iou_matrix = box_iou(boxes, gt_boxes.to(device).float())
    best_iou, best_indices = iou_matrix.max(dim=1)
    labels = gt_labels.to(device).long()[best_indices]
    labels = torch.where(best_iou > 0.0, labels, torch.zeros_like(labels))
    matched_boxes = gt_boxes.to(device).float()[best_indices]
    matched_boxes = torch.where((best_iou > 0.0).unsqueeze(1), matched_boxes, torch.zeros_like(matched_boxes))
    return best_iou, labels, matched_boxes


def combine_verifier_scores(
    fft_scores: torch.Tensor,
    manifold_scores: torch.Tensor,
    reference_stats: dict[str, float],
    *,
    fft_weight: float,
    manifold_weight: float,
) -> torch.Tensor:
    fft_scores = fft_scores.float()
    manifold_scores = manifold_scores.to(fft_scores.device).float()
    fft_mean = float(reference_stats.get("fft_mean", 0.0))
    fft_std = max(float(reference_stats.get("fft_std", 1.0)), 1e-6)
    manifold_mean = float(reference_stats.get("manifold_mean", 0.0))
    manifold_std = max(float(reference_stats.get("manifold_std", 1.0)), 1e-6)
    fft_z = (fft_scores - fft_mean) / fft_std
    manifold_z = (manifold_scores - manifold_mean) / manifold_std
    return float(fft_weight) * fft_z + float(manifold_weight) * manifold_z


def manifold_soft_rescue_weights(
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    verifier_scores: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    thresholds: torch.Tensor | None = None,
    temperature: float = 0.2,
    low_conf_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = scores.float()
    device = scores.device
    best_iou = best_iou.to(device).float()
    best_labels = best_labels.to(device).long()
    verifier_scores = verifier_scores.to(device).float()
    if thresholds is None:
        thresholds = verifier_scores.new_full(verifier_scores.shape, float(cfg.verifier_positive_min or 0.0))
    else:
        thresholds = thresholds.to(device).float()
    low_source = scores if low_conf_scores is None else low_conf_scores.to(device).float()
    lchi = (
        (low_source <= float(cfg.low_conf_max))
        & (best_iou >= float(cfg.high_iou_min))
        & (best_labels > 0)
    )
    temp = max(float(temperature), 1e-6)
    soft = torch.sigmoid((verifier_scores - thresholds) / temp).detach()
    weights = torch.zeros_like(scores)
    weights[lchi] = soft[lchi]
    return weights, lchi


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)


def _validate_aligned_boxes(boxes1: torch.Tensor, boxes2: torch.Tensor) -> None:
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError("boxes1 and boxes2 must both have shape (N, 4)")


def _aligned_iou_terms(boxes1: torch.Tensor, boxes2: torch.Tensor) -> dict[str, torch.Tensor]:
    _validate_aligned_boxes(boxes1, boxes2)
    if boxes1.numel() == 0:
        empty = boxes1.new_empty((boxes1.shape[0],))
        return {
            "iou": empty,
            "union": empty,
            "enc_area": empty,
            "enc_diag_sq": empty,
            "center_dist_sq": empty,
            "w1": empty,
            "h1": empty,
            "w2": empty,
            "h2": empty,
        }
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    union = (area1 + area2 - inter).clamp_min(1e-6)
    iou = inter / union

    enc_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enc_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp_min(0.0)
    enc_area = (enc_wh[:, 0] * enc_wh[:, 1]).clamp_min(1e-6)
    enc_diag_sq = (enc_wh[:, 0].pow(2) + enc_wh[:, 1].pow(2)).clamp_min(1e-6)
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5
    center_delta = center1 - center2
    center_dist_sq = center_delta[:, 0].pow(2) + center_delta[:, 1].pow(2)
    return {
        "iou": iou,
        "union": union,
        "enc_area": enc_area,
        "enc_diag_sq": enc_diag_sq,
        "center_dist_sq": center_dist_sq,
        "w1": (boxes1[:, 2] - boxes1[:, 0]).clamp_min(1e-6),
        "h1": (boxes1[:, 3] - boxes1[:, 1]).clamp_min(1e-6),
        "w2": (boxes2[:, 2] - boxes2[:, 0]).clamp_min(1e-6),
        "h2": (boxes2[:, 3] - boxes2[:, 1]).clamp_min(1e-6),
    }


def generalized_box_iou_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    terms = _aligned_iou_terms(boxes1, boxes2)
    return terms["iou"] - (terms["enc_area"] - terms["union"]) / terms["enc_area"]


def distance_box_iou_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    terms = _aligned_iou_terms(boxes1, boxes2)
    return terms["iou"] - terms["center_dist_sq"] / terms["enc_diag_sq"]


def complete_box_iou_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    terms = _aligned_iou_terms(boxes1, boxes2)
    diou = terms["iou"] - terms["center_dist_sq"] / terms["enc_diag_sq"]
    v = (4.0 / torch.pi**2) * (
        torch.atan(terms["w2"] / terms["h2"]) - torch.atan(terms["w1"] / terms["h1"])
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - terms["iou"] + v).clamp_min(1e-6)
    return diou - alpha * v


def aligned_box_iou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor, *, mode: str = "giou") -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "smooth_l1":
        _validate_aligned_boxes(boxes1, boxes2)
        return F.smooth_l1_loss(boxes1, boxes2, reduction="none").sum(dim=1)
    if mode == "giou":
        return 1.0 - generalized_box_iou_aligned(boxes1, boxes2)
    if mode == "diou":
        return 1.0 - distance_box_iou_aligned(boxes1, boxes2)
    if mode == "ciou":
        return 1.0 - complete_box_iou_aligned(boxes1, boxes2)
    raise ValueError(f"Unknown bbox localization loss mode: {mode}")


def bbox_localization_rescue_loss(
    decoded_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    rescue_weights: torch.Tensor | None = None,
    low_conf_scores: torch.Tensor | None = None,
    loss_mode: str = "giou",
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_mode = str(loss_mode).lower()
    if decoded_boxes.numel() == 0:
        zero = decoded_boxes.sum() * 0.0
        return zero, {
            "bbox_rescue_count": 0,
            "bbox_rescue_weight_sum": 0.0,
            "bbox_rescue_loss": 0.0,
            "bbox_rescue_loss_mode": loss_mode,
        }
    scores = scores.to(decoded_boxes.device).float()
    low_source = scores if low_conf_scores is None else low_conf_scores.to(decoded_boxes.device).float()
    best_iou = best_iou.to(decoded_boxes.device).float()
    best_labels = best_labels.to(decoded_boxes.device).long()
    target_boxes = target_boxes.to(decoded_boxes.device).float()
    lchi = (
        (low_source <= float(cfg.low_conf_max))
        & (best_iou >= float(cfg.high_iou_min))
        & (best_labels > 0)
    )
    if rescue_weights is None:
        weights = lchi.float()
    else:
        weights = rescue_weights.to(decoded_boxes.device).float() * lchi.float()
    selected = weights > 0
    if not selected.any():
        zero = decoded_boxes.sum() * 0.0
        return zero, {
            "bbox_rescue_count": 0,
            "bbox_rescue_weight_sum": 0.0,
            "bbox_rescue_loss": 0.0,
            "bbox_rescue_loss_mode": loss_mode,
        }
    raw_loss = aligned_box_iou_loss(decoded_boxes[selected], target_boxes[selected], mode=loss_mode)
    selected_weights = weights[selected].detach()
    loss = (raw_loss * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)
    return loss, {
        "bbox_rescue_count": int(selected.sum().item()),
        "bbox_rescue_weight_sum": float(selected_weights.sum().detach().cpu().item()),
        "bbox_rescue_loss": float(loss.detach().cpu().item()),
        "bbox_rescue_loss_mode": loss_mode,
    }


def build_pairwise_rescue_ranking_loss(
    class_logits: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    verifier_scores: torch.Tensor | None = None,
    margin: float = 0.1,
    negative_mode: str = "all_low_iou",
    low_conf_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {"pairwise_rescue_pair_count": 0}
    scores = scores.to(class_logits.device).float()
    best_iou = best_iou.to(class_logits.device).float()
    best_labels = best_labels.to(class_logits.device).long()
    masks = confidence_iou_region_masks(
        scores,
        best_iou,
        cfg,
        low_conf_scores=low_conf_scores.to(class_logits.device) if low_conf_scores is not None else None,
    )
    positive = masks["low_conf_high_iou"] & (best_labels > 0)
    if verifier_scores is not None and cfg.verifier_positive_min is not None:
        positive = positive & (verifier_scores.to(class_logits.device).float() >= float(cfg.verifier_positive_min))
    mode = str(negative_mode)
    if mode == "all_low_iou":
        negative = (masks["low_conf_low_iou"] | masks["high_conf_low_iou"]) & (best_labels > 0)
    elif mode == "dangerous":
        negative = masks["high_conf_low_iou"] & (best_labels > 0)
        if verifier_scores is not None and cfg.verifier_hard_negative_min is not None:
            verifier_positive = verifier_scores.to(class_logits.device).float() >= float(cfg.verifier_hard_negative_min)
            negative = negative | (masks["low_conf_low_iou"] & verifier_positive & (best_labels > 0))
    else:
        raise ValueError(f"Unknown pairwise negative mode: {negative_mode}")
    if verifier_scores is not None and cfg.verifier_hard_negative_min is not None:
        verifier_positive = verifier_scores.to(class_logits.device).float() >= float(cfg.verifier_hard_negative_min)
        negative = negative & (verifier_positive | masks["high_conf_low_iou"])
    positive_idx = torch.where(positive)[0]
    negative_idx = torch.where(negative)[0]
    losses = []
    for pos_idx in positive_idx.tolist():
        pos_label = int(best_labels[pos_idx].item())
        if pos_label <= 0 or pos_label >= class_logits.shape[1]:
            continue
        pos_logit = class_logits[pos_idx, pos_label]
        for neg_idx in negative_idx.tolist():
            if int(best_labels[neg_idx].item()) != pos_label:
                continue
            neg_logit = class_logits[neg_idx, pos_label]
            losses.append(F.softplus(float(margin) - (pos_logit - neg_logit)))
    if not losses:
        zero = class_logits.sum() * 0.0
        return zero, {"pairwise_rescue_pair_count": 0}
    return torch.stack(losses).mean(), {"pairwise_rescue_pair_count": len(losses)}


def build_verifier_guided_ranking_loss(
    class_logits: torch.Tensor,
    best_labels: torch.Tensor,
    best_iou: torch.Tensor,
    verifier_scores: torch.Tensor,
    *,
    positive_iou_min: float = 0.75,
    negative_iou_max: float = 0.3,
    positive_score_min: float = 0.5,
    negative_score_max: float = 0.0,
    margin: float = 0.1,
    max_pairs: int = 32,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "verifier_ranking_pair_count": 0,
            "verifier_ranking_active_count": 0,
            "verifier_ranking_positive_count": 0,
            "verifier_ranking_negative_count": 0,
            "verifier_ranking_loss": 0.0,
        }
    device = class_logits.device
    labels = best_labels.to(device).long()
    iou = best_iou.to(device).float()
    verifier = verifier_scores.to(device).float().detach()
    positive = (
        (labels > 0)
        & (labels < class_logits.shape[1])
        & (iou >= float(positive_iou_min))
        & (verifier >= float(positive_score_min))
    )
    negative = (
        (labels > 0)
        & (labels < class_logits.shape[1])
        & (iou <= float(negative_iou_max))
        & (verifier <= float(negative_score_max))
    )
    positive_idx = torch.where(positive)[0]
    negative_idx = torch.where(negative)[0]
    losses = []
    max_pair_count = max(0, int(max_pairs))
    for pos_idx in positive_idx.tolist():
        label = int(labels[pos_idx].item())
        pos_logit = class_logits[pos_idx, label]
        same_class_negatives = negative_idx[labels[negative_idx] == label]
        if same_class_negatives.numel() == 0:
            continue
        order = torch.argsort(verifier[same_class_negatives], descending=False)
        for neg_idx in same_class_negatives[order].tolist():
            neg_logit = class_logits[neg_idx, label]
            losses.append(F.softplus(float(margin) - (pos_logit - neg_logit)))
            if max_pair_count and len(losses) >= max_pair_count:
                break
        if max_pair_count and len(losses) >= max_pair_count:
            break
    if not losses:
        zero = class_logits.sum() * 0.0
        return zero, {
            "verifier_ranking_pair_count": 0,
            "verifier_ranking_active_count": 0,
            "verifier_ranking_positive_count": int(positive_idx.numel()),
            "verifier_ranking_negative_count": int(negative_idx.numel()),
            "verifier_ranking_loss": 0.0,
        }
    penalties = torch.stack(losses)
    loss = penalties.mean()
    return loss, {
        "verifier_ranking_pair_count": int(penalties.numel()),
        "verifier_ranking_active_count": int((penalties > 0).sum().item()),
        "verifier_ranking_positive_count": int(positive_idx.numel()),
        "verifier_ranking_negative_count": int(negative_idx.numel()),
        "verifier_ranking_loss": float(loss.detach().cpu().item()),
    }


def _verifier_weights(
    verifier_scores: torch.Tensor,
    gate: float,
    cfg: ConfidenceRescueConfig,
) -> torch.Tensor:
    mode = str(cfg.verifier_weight_mode)
    if mode == "hard":
        return (verifier_scores.float() >= float(gate)).float()
    if mode == "sigmoid":
        temperature = max(float(cfg.verifier_weight_temperature), 1e-6)
        return torch.sigmoid((verifier_scores.float() - float(gate)) / temperature)
    raise ValueError(f"Unknown verifier weight mode: {cfg.verifier_weight_mode}")


def build_confidence_rescue_targets(
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    verifier_scores: torch.Tensor | None = None,
    low_conf_scores: torch.Tensor | None = None,
    positive_candidate_mask: torch.Tensor | None = None,
) -> ConfidenceRescueTargets:
    scores = scores.float()
    device = scores.device
    best_iou = best_iou.to(device).float()
    best_labels = best_labels.to(device).long().clamp_min(0)
    masks = confidence_iou_region_masks(
        scores,
        best_iou,
        cfg,
        low_conf_scores=low_conf_scores.to(device) if low_conf_scores is not None else None,
    )

    positive_mask = masks["low_conf_high_iou"]
    if positive_candidate_mask is not None:
        positive_mask = positive_mask & positive_candidate_mask.to(device).bool()
    positive_weights = torch.ones_like(scores)
    if verifier_scores is not None and cfg.verifier_positive_min is not None:
        positive_weights = _verifier_weights(
            verifier_scores.to(device).float(),
            float(cfg.verifier_positive_min),
            cfg,
        )
        if str(cfg.verifier_weight_mode) == "hard":
            positive_mask = positive_mask & (positive_weights > 0)

    negative_mask = masks["high_conf_low_iou"]
    low_conf_negative_weights = torch.ones_like(scores)
    if bool(cfg.include_low_conf_negatives):
        negative_mask = negative_mask | masks["low_conf_low_iou"]
    if verifier_scores is not None and cfg.verifier_hard_negative_min is not None:
        low_conf_negative_weights = _verifier_weights(
            verifier_scores.to(device).float(),
            float(cfg.verifier_hard_negative_min),
            cfg,
        )
        negative_mask = negative_mask | (masks["low_conf_low_iou"] & (low_conf_negative_weights > 0))

    target_labels = torch.zeros_like(best_labels)
    target_labels[positive_mask] = best_labels[positive_mask]
    weights = torch.zeros_like(scores)
    weights[positive_mask] = float(cfg.positive_weight) * positive_weights[positive_mask]
    weights[negative_mask] = float(cfg.negative_weight)
    low_conf_negative_mask = negative_mask & masks["low_conf_low_iou"]
    weights[low_conf_negative_mask] = float(cfg.negative_weight) * low_conf_negative_weights[low_conf_negative_mask]
    return ConfidenceRescueTargets(
        target_labels=target_labels,
        weights=weights,
        positive_mask=positive_mask,
        negative_mask=negative_mask,
    )


def confidence_rescue_loss(
    class_logits: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    verifier_scores: torch.Tensor | None = None,
    low_conf_scores: torch.Tensor | None = None,
    positive_candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {"rescue_positive_count": 0, "rescue_negative_count": 0, "rescue_weight_sum": 0.0}

    targets = build_confidence_rescue_targets(
        scores.to(class_logits.device),
        best_iou.to(class_logits.device),
        best_labels.to(class_logits.device),
        cfg,
        verifier_scores=verifier_scores.to(class_logits.device) if verifier_scores is not None else None,
        low_conf_scores=low_conf_scores.to(class_logits.device) if low_conf_scores is not None else None,
        positive_candidate_mask=positive_candidate_mask.to(class_logits.device) if positive_candidate_mask is not None else None,
    )
    selected = targets.weights > 0
    if not selected.any():
        zero = class_logits.sum() * 0.0
        return zero, {
            "rescue_positive_count": int(targets.positive_mask.sum().item()),
            "rescue_negative_count": int(targets.negative_mask.sum().item()),
            "rescue_weight_sum": 0.0,
        }

    raw_loss = F.cross_entropy(class_logits[selected], targets.target_labels[selected], reduction="none")
    selected_weights = targets.weights[selected].to(class_logits.device)
    loss = (raw_loss * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)
    return loss, {
        "rescue_positive_count": int(targets.positive_mask.sum().item()),
        "rescue_negative_count": int(targets.negative_mask.sum().item()),
        "rescue_weight_sum": float(targets.weights.sum().detach().cpu().item()),
    }


def _gather_label_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    labels = labels.to(logits.device).long()
    valid = (labels > 0) & (labels < logits.shape[1])
    gathered = probs.new_zeros((labels.shape[0],))
    if valid.any():
        gathered[valid] = probs[valid].gather(1, labels[valid].unsqueeze(1)).squeeze(1)
    if (~valid).any() and logits.shape[1] > 1:
        gathered[~valid] = probs[~valid, 1:].max(dim=1).values
    return gathered


def confidence_rescue_increment_loss(
    class_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    verifier_scores: torch.Tensor | None = None,
    low_conf_scores: torch.Tensor | None = None,
    positive_candidate_mask: torch.Tensor | None = None,
    target_delta: float = 0.05,
    target_cap: float = 0.6,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "rescue_positive_count": 0,
            "rescue_negative_count": 0,
            "rescue_weight_sum": 0.0,
            "rescue_increment_target_mean": 0.0,
            "rescue_increment_current_mean": 0.0,
        }

    targets = build_confidence_rescue_targets(
        scores.to(class_logits.device),
        best_iou.to(class_logits.device),
        best_labels.to(class_logits.device),
        cfg,
        verifier_scores=verifier_scores.to(class_logits.device) if verifier_scores is not None else None,
        low_conf_scores=low_conf_scores.to(class_logits.device) if low_conf_scores is not None else None,
        positive_candidate_mask=positive_candidate_mask.to(class_logits.device) if positive_candidate_mask is not None else None,
    )
    selected = targets.weights > 0
    if not selected.any():
        zero = class_logits.sum() * 0.0
        return zero, {
            "rescue_positive_count": int(targets.positive_mask.sum().item()),
            "rescue_negative_count": int(targets.negative_mask.sum().item()),
            "rescue_weight_sum": 0.0,
            "rescue_increment_target_mean": 0.0,
            "rescue_increment_current_mean": 0.0,
        }

    weighted_losses = []
    weighted_weights = []
    positive_targets = class_logits.new_empty((0,))
    positive_current = class_logits.new_empty((0,))
    positive = targets.positive_mask & selected & (targets.target_labels > 0) & (targets.target_labels < class_logits.shape[1])
    if positive.any():
        labels = targets.target_labels[positive].to(class_logits.device).long()
        current_probs = F.softmax(class_logits[positive], dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            baseline_probs = F.softmax(baseline_logits.to(class_logits.device)[positive], dim=1).gather(
                1,
                labels.unsqueeze(1),
            ).squeeze(1)
            capped_target = torch.minimum(
                baseline_probs + float(target_delta),
                baseline_probs.new_full(baseline_probs.shape, float(target_cap)),
            )
            target_probs = torch.maximum(baseline_probs, capped_target)
        weights = targets.weights[positive].to(class_logits.device)
        weighted_losses.append((current_probs - target_probs).pow(2))
        weighted_weights.append(weights)
        positive_targets = target_probs.detach()
        positive_current = current_probs.detach()

    negative = targets.negative_mask & selected
    if negative.any():
        target_bg = torch.zeros((int(negative.sum().item()),), dtype=torch.long, device=class_logits.device)
        weighted_losses.append(F.cross_entropy(class_logits[negative], target_bg, reduction="none"))
        weighted_weights.append(targets.weights[negative].to(class_logits.device))

    if not weighted_losses:
        zero = class_logits.sum() * 0.0
        return zero, {
            "rescue_positive_count": int(targets.positive_mask.sum().item()),
            "rescue_negative_count": int(targets.negative_mask.sum().item()),
            "rescue_weight_sum": 0.0,
            "rescue_increment_target_mean": 0.0,
            "rescue_increment_current_mean": 0.0,
        }

    losses = torch.cat(weighted_losses)
    weights = torch.cat(weighted_weights)
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return loss, {
        "rescue_positive_count": int(targets.positive_mask.sum().item()),
        "rescue_negative_count": int(targets.negative_mask.sum().item()),
        "rescue_weight_sum": float(targets.weights.sum().detach().cpu().item()),
        "rescue_increment_target_mean": float(positive_targets.mean().item()) if positive_targets.numel() else 0.0,
        "rescue_increment_current_mean": float(positive_current.mean().item()) if positive_current.numel() else 0.0,
    }


def confidence_threshold_crossing_loss(
    class_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    verifier_scores: torch.Tensor | None = None,
    low_conf_scores: torch.Tensor | None = None,
    crossing_baseline_scores: torch.Tensor | None = None,
    positive_candidate_mask: torch.Tensor | None = None,
    score_threshold: float = 0.05,
    margin: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0 or class_logits.shape[1] <= 1:
        zero = class_logits.sum() * 0.0
        return zero, {
            "confidence_crossing_count": 0,
            "confidence_crossing_active_count": 0,
            "confidence_crossing_loss": 0.0,
        }

    device = class_logits.device
    scores = scores.to(device).float()
    best_iou = best_iou.to(device).float()
    best_labels = best_labels.to(device).long()
    baseline_logits = baseline_logits.to(device)
    low_source = scores if low_conf_scores is None else low_conf_scores.to(device).float()
    selected = (
        (low_source <= float(cfg.low_conf_max))
        & (best_iou >= float(cfg.high_iou_min))
        & (best_labels > 0)
        & (best_labels < class_logits.shape[1])
    )
    if positive_candidate_mask is not None:
        selected = selected & positive_candidate_mask.to(device).bool()
    if verifier_scores is not None and cfg.verifier_positive_min is not None:
        selected = selected & (verifier_scores.to(device).float() >= float(cfg.verifier_positive_min))
    if not selected.any():
        zero = class_logits.sum() * 0.0
        return zero, {
            "confidence_crossing_count": 0,
            "confidence_crossing_active_count": 0,
            "confidence_crossing_loss": 0.0,
        }

    labels = best_labels[selected]
    current_probs = F.softmax(class_logits[selected], dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if crossing_baseline_scores is None:
            baseline_probs = F.softmax(baseline_logits[selected], dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
        else:
            baseline_probs = crossing_baseline_scores.to(device).float()[selected]
        needs_crossing = baseline_probs < float(score_threshold)
    selected_count = int(needs_crossing.sum().item())
    if selected_count == 0:
        zero = class_logits.sum() * 0.0
        return zero, {
            "confidence_crossing_count": 0,
            "confidence_crossing_active_count": 0,
            "confidence_crossing_loss": 0.0,
        }

    target = float(score_threshold) + float(margin)
    penalties = F.relu(current_probs[needs_crossing].new_full((selected_count,), target) - current_probs[needs_crossing])
    loss = penalties.pow(2).mean()
    return loss, {
        "confidence_crossing_count": selected_count,
        "confidence_crossing_active_count": int((penalties > 0).sum().item()),
        "confidence_crossing_loss": float(loss.detach().cpu().item()),
    }


def score_shift_budget_loss(
    class_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    best_labels: torch.Tensor,
    cfg: ConfidenceRescueConfig,
    *,
    verifier_scores: torch.Tensor | None = None,
    low_conf_scores: torch.Tensor | None = None,
    delta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    if class_logits.numel() == 0 or class_logits.shape[1] <= 1:
        zero = class_logits.sum() * 0.0
        return zero, {"score_budget_count": 0, "score_budget_violation_count": 0, "score_budget_mean_shift": 0.0}

    targets = build_confidence_rescue_targets(
        scores.to(class_logits.device),
        best_iou.to(class_logits.device),
        best_labels.to(class_logits.device),
        cfg,
        verifier_scores=verifier_scores.to(class_logits.device) if verifier_scores is not None else None,
        low_conf_scores=low_conf_scores.to(class_logits.device) if low_conf_scores is not None else None,
    )
    budget_mask = ~targets.positive_mask
    if not budget_mask.any():
        zero = class_logits.sum() * 0.0
        return zero, {"score_budget_count": 0, "score_budget_violation_count": 0, "score_budget_mean_shift": 0.0}

    labels = best_labels.to(class_logits.device).long()
    current_probs = _gather_label_probabilities(class_logits, labels)
    with torch.no_grad():
        baseline_probs = _gather_label_probabilities(baseline_logits.to(class_logits.device), labels)
    shift = current_probs[budget_mask] - baseline_probs[budget_mask]
    penalty = F.relu(shift - float(delta))
    return penalty.mean(), {
        "score_budget_count": int(budget_mask.sum().item()),
        "score_budget_violation_count": int((penalty > 0).sum().item()),
        "score_budget_mean_shift": float(shift.mean().detach().cpu().item()) if shift.numel() else 0.0,
    }


def select_best_checkpoint_update(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
    cfg: BestCheckpointConfig,
) -> dict[str, object]:
    metric_name = str(cfg.selection_metric)
    current_metric = float(current_metrics.get(metric_name, 0.0))
    best_metric = None if best_metrics is None else float(best_metrics.get(metric_name, 0.0))
    if best_metric is None:
        metric_improved = True
    elif bool(cfg.higher_is_better):
        metric_improved = current_metric > best_metric + float(cfg.min_delta)
    else:
        metric_improved = current_metric < best_metric - float(cfg.min_delta)

    failed_guards: list[str] = []
    baseline_predictions = float(baseline_metrics.get("num_predictions", 0.0))
    current_predictions = float(current_metrics.get("num_predictions", 0.0))
    if cfg.max_prediction_ratio is not None and baseline_predictions > 0:
        prediction_ratio = current_predictions / max(1.0, baseline_predictions)
        if prediction_ratio > float(cfg.max_prediction_ratio):
            failed_guards.append("prediction_ratio")
    if cfg.max_prediction_delta is not None:
        if current_predictions - baseline_predictions > float(cfg.max_prediction_delta):
            failed_guards.append("prediction_delta")
    if cfg.max_fp_rate_delta is not None:
        fp_delta = float(current_metrics.get("false_positive_rate", 0.0)) - float(
            baseline_metrics.get("false_positive_rate", 0.0)
        )
        if fp_delta > float(cfg.max_fp_rate_delta):
            failed_guards.append("false_positive_rate")
    if cfg.max_high_conf_fp_rate_delta is not None:
        high_conf_fp_delta = float(current_metrics.get("high_conf_fp_rate", 0.0)) - float(
            baseline_metrics.get("high_conf_fp_rate", 0.0)
        )
        if high_conf_fp_delta > float(cfg.max_high_conf_fp_rate_delta):
            failed_guards.append("high_conf_fp_rate")
    if cfg.max_ece_delta is not None:
        ece_delta = float(current_metrics.get("ece", 0.0)) - float(baseline_metrics.get("ece", 0.0))
        if ece_delta > float(cfg.max_ece_delta):
            failed_guards.append("ece")

    safe_to_save = len(failed_guards) == 0
    return {
        "selection_metric": metric_name,
        "current_metric": current_metric,
        "best_metric": best_metric,
        "metric_improved": bool(metric_improved),
        "safe_to_save_best": bool(safe_to_save),
        "failed_guards": failed_guards,
        "should_update_best": bool(metric_improved and safe_to_save),
    }
