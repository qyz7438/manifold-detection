from __future__ import annotations

import torch

from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt


def _compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _compute_class_ap(
    predictions: list[dict],
    targets: list[dict],
    class_id: int,
    iou_threshold: float,
    score_threshold: float,
) -> float:
    """Compute AP for a single class (class-aware matching)."""
    scored: list[tuple[float, bool]] = []
    total_gt = 0

    for prediction, target in zip(predictions, targets):
        target_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}
        pred_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prediction.items()}

        gt_mask = target_cpu.get("labels", torch.empty((0,), dtype=torch.long)) == class_id
        total_gt += int(gt_mask.sum().item())

        pred_labels = pred_cpu.get("labels", torch.empty((0,), dtype=torch.long))
        pred_mask = pred_labels == class_id
        if not pred_mask.any():
            continue

        filtered_pred = {
            "boxes": pred_cpu["boxes"][pred_mask],
            "scores": pred_cpu["scores"][pred_mask],
            "labels": pred_cpu["labels"][pred_mask],
        }
        filtered_target = {
            "boxes": target_cpu["boxes"][gt_mask],
            "labels": target_cpu["labels"][gt_mask],
        }

        matched = match_predictions_to_gt(
            filtered_pred, filtered_target, iou_threshold=iou_threshold, score_threshold=score_threshold
        )
        matched_pred_indices = {m["pred_index"] for m in matched["matches"]}

        scores = filtered_pred["scores"]
        for pred_idx, score in enumerate(scores.tolist()):
            if score < score_threshold:
                continue
            scored.append((float(score), pred_idx in matched_pred_indices))

    if not scored or total_gt == 0:
        return 0.0

    scored.sort(key=lambda item: item[0], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for _, is_tp in scored:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / max(1, tp_cum + fp_cum))
        recalls.append(tp_cum / max(1, total_gt))

    return _compute_ap(recalls, precisions)


def precision_at_recall(scored: list[tuple[float, bool]], total_gt: int, target_recall: float = 0.85) -> float | None:
    if total_gt <= 0:
        return None
    tp_cum = 0
    fp_cum = 0
    best_precision = None
    for _, is_tp in sorted(scored, key=lambda item: item[0], reverse=True):
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        recall = tp_cum / max(1, total_gt)
        if recall >= target_recall:
            precision = tp_cum / max(1, tp_cum + fp_cum)
            best_precision = precision if best_precision is None else max(best_precision, precision)
    return best_precision


def detection_ece(scored: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not scored:
        return None
    scores = torch.tensor([item[0] for item in scored], dtype=torch.float32)
    labels = torch.tensor([1.0 if item[1] else 0.0 for item in scored], dtype=torch.float32)
    ece = 0.0
    for bin_idx in range(bins):
        left = bin_idx / bins
        right = (bin_idx + 1) / bins
        if bin_idx == bins - 1:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not mask.any():
            continue
        confidence = scores[mask].mean()
        accuracy = labels[mask].mean()
        ece += float(mask.float().mean().item()) * abs(float(confidence.item()) - float(accuracy.item()))
    return ece


def evaluate_detection_predictions(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    high_conf_threshold: float = 0.7,
    fixed_recall: float = 0.85,
    per_class: bool = False,
    num_classes: int | None = None,
) -> dict:
    scored = []
    total_gt = 0
    high_conf_fp = 0
    high_conf_total = 0
    unmatched_gt_total = 0

    for image_idx, (prediction, target) in enumerate(zip(predictions, targets)):
        target_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}
        pred_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prediction.items()}
        total_gt += len(target_cpu.get("boxes", []))
        matched = match_predictions_to_gt(pred_cpu, target_cpu, iou_threshold=iou_threshold, score_threshold=score_threshold)
        matched_pred_indices = {m["pred_index"] for m in matched["matches"]}
        unmatched_gt_total += len(matched["unmatched_gt"])

        scores = pred_cpu.get("scores", torch.empty((0,)))
        for pred_idx, score in enumerate(scores.tolist()):
            if score < score_threshold:
                continue
            is_tp = pred_idx in matched_pred_indices
            scored.append((float(score), is_tp))
            if score >= high_conf_threshold:
                high_conf_total += 1
                if not is_tp:
                    high_conf_fp += 1

    scored.sort(key=lambda item: item[0], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for _, is_tp in scored:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / max(1, tp_cum + fp_cum))
        recalls.append(tp_cum / max(1, total_gt))

    final_precision = tp_cum / max(1, tp_cum + fp_cum)
    final_recall = tp_cum / max(1, total_gt)

    ap75 = _compute_ap75(predictions, targets, score_threshold)

    result = {
        "ap50": _compute_ap(recalls, precisions),
        "ap75": ap75,
        "precision": final_precision,
        "recall": final_recall,
        "false_positive_rate": fp_cum / max(1, tp_cum + fp_cum),
        "high_conf_fp_rate": high_conf_fp / max(1, high_conf_total),
        "high_conf_fp_count": high_conf_fp,
        "high_conf_total": high_conf_total,
        "ece": detection_ece(scored),
        f"precision_at_recall_{str(fixed_recall).replace('.', '_')}": precision_at_recall(scored, total_gt, target_recall=fixed_recall),
        "miss_rate": unmatched_gt_total / max(1, total_gt),
        "num_predictions": len(scored),
        "num_gt": total_gt,
    }

    if per_class:
        if num_classes is None:
            all_labels = set()
            for target in targets:
                labels = target.get("labels", torch.empty((0,), dtype=torch.long))
                if torch.is_tensor(labels):
                    all_labels.update(labels.tolist())
            num_classes = max(all_labels, default=0) + 1

        per_class_ap50 = {}
        per_class_ap75 = {}
        for c in range(1, num_classes):
            per_class_ap50[str(c)] = _compute_class_ap(predictions, targets, c, iou_threshold=0.5, score_threshold=score_threshold)
            per_class_ap75[str(c)] = _compute_class_ap(predictions, targets, c, iou_threshold=0.75, score_threshold=score_threshold)
        result["per_class_ap50"] = per_class_ap50
        result["per_class_ap75"] = per_class_ap75

    return result


def _compute_ap75(predictions: list[dict], targets: list[dict], score_threshold: float = 0.05) -> float:
    iou_threshold = 0.75
    tp_fp_labels: list[tuple[float, bool]] = []
    total_gt = 0

    for prediction, target in zip(predictions, targets):
        target_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}
        pred_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prediction.items()}
        total_gt += len(target_cpu.get("boxes", []))
        matched = match_predictions_to_gt(pred_cpu, target_cpu, iou_threshold=iou_threshold, score_threshold=score_threshold)
        matched_indices = {m["pred_index"] for m in matched["matches"]}
        scores = pred_cpu.get("scores", torch.empty((0,)))
        for pred_idx, score in enumerate(scores.tolist()):
            if score < score_threshold:
                continue
            tp_fp_labels.append((float(score), pred_idx in matched_indices))

    if not tp_fp_labels:
        return 0.0

    tp_fp_labels.sort(key=lambda x: x[0], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for _, is_tp in tp_fp_labels:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / max(1, tp_cum + fp_cum))
        recalls.append(tp_cum / max(1, total_gt))

    return _compute_ap(recalls, precisions)


def summarize_iou_diagnostics(matched_ious: list[float], matched_scores: list[float]) -> dict[str, float]:
    if not matched_ious:
        return {"tp_iou_mean": 0.0, "tp_iou_median": 0.0, "tp_iou_ge_075_rate": 0.0, "score_iou_corr": 0.0}
    ious = torch.tensor(matched_ious, dtype=torch.float32)
    scores = torch.tensor(matched_scores, dtype=torch.float32)
    corr = 0.0
    if ious.numel() > 1 and float(scores.std(unbiased=False)) > 0 and float(ious.std(unbiased=False)) > 0:
        corr = float(torch.corrcoef(torch.stack([scores, ious]))[0, 1].item())
    return {
        "tp_iou_mean": float(ious.mean().item()),
        "tp_iou_median": float(ious.median().item()),
        "tp_iou_ge_075_rate": float((ious >= 0.75).float().mean().item()),
        "score_iou_corr": corr,
    }
