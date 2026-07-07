from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_energy_action_search import apply_action_search, best_iou_and_gt
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.methods.energy_transport import ActionSearchConfig
from spectral_detection_posttrain.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic proxy exploration for low-energy detection action endpoints."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", type=int, default=160)
    parser.add_argument("--min-gt", type=int, default=2)
    parser.add_argument("--max-gt", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--fp-per-image", type=int, default=4)
    parser.add_argument("--dense-fp-per-image", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--target-iou", type=float, default=0.75)
    parser.add_argument("--low-quality-iou", type=float, default=0.30)
    parser.add_argument("--threshold-margin", type=float, default=0.01)
    parser.add_argument("--max-score-delta", type=float, default=0.20)
    parser.add_argument("--verifier-mode", default="oracle", choices=("oracle", "noisy"))
    parser.add_argument("--verifier-noise-std", type=float, default=0.08)
    parser.add_argument("--verifier-fp-rate", type=float, default=0.05)
    parser.add_argument("--verifier-fn-rate", type=float, default=0.10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", default="output/energy_action_proxy.json")
    return parser.parse_args()


def random_box(rng: random.Random, image_size: int = 256) -> torch.Tensor:
    width = rng.uniform(16.0, 54.0)
    height = rng.uniform(16.0, 54.0)
    x1 = rng.uniform(0.0, image_size - width - 1.0)
    y1 = rng.uniform(0.0, image_size - height - 1.0)
    return torch.tensor([x1, y1, x1 + width, y1 + height], dtype=torch.float32)


def centered_iou_box(box: torch.Tensor, target_iou: float, *, image_size: int = 256) -> torch.Tensor:
    target_iou = max(min(float(target_iou), 1.0), 1e-3)
    center = torch.stack([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    wh = torch.stack([(box[2] - box[0]), (box[3] - box[1])])
    scale = target_iou ** -0.5
    new_wh = wh * scale
    x1y1 = center - new_wh * 0.5
    x2y2 = center + new_wh * 0.5
    out = torch.cat([x1y1, x2y2])
    out[0::2] = out[0::2].clamp(0.0, float(image_size - 1))
    out[1::2] = out[1::2].clamp(0.0, float(image_size - 1))
    return out


def jitter_box(box: torch.Tensor, rng: random.Random, *, frac: float = 0.15, image_size: int = 256) -> torch.Tensor:
    wh = torch.stack([(box[2] - box[0]), (box[3] - box[1])])
    shift = torch.tensor(
        [rng.uniform(-frac, frac) * wh[0], rng.uniform(-frac, frac) * wh[1]],
        dtype=torch.float32,
    )
    out = box.clone()
    out[:2] += shift
    out[2:] += shift
    out[0::2] = out[0::2].clamp(0.0, float(image_size - 1))
    out[1::2] = out[1::2].clamp(0.0, float(image_size - 1))
    return out


def append_prediction(prediction: dict, box: torch.Tensor, label: int, score: float) -> None:
    prediction["boxes"].append(box)
    prediction["labels"].append(int(label))
    prediction["scores"].append(float(score))


def finalize_prediction(prediction: dict) -> dict:
    if prediction["boxes"]:
        boxes = torch.stack(prediction["boxes"]).float()
        labels = torch.tensor(prediction["labels"], dtype=torch.long)
        scores = torch.tensor(prediction["scores"], dtype=torch.float32)
        order = torch.argsort(scores, descending=True)
        return {"boxes": boxes[order], "labels": labels[order], "scores": scores[order]}
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.long),
        "scores": torch.empty((0,), dtype=torch.float32),
    }


def make_synthetic_dataset(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], dict]:
    rng = random.Random(int(args.seed))
    targets: list[dict] = []
    default_predictions: list[dict] = []
    dense_predictions: list[dict] = []
    counts = {
        "gt": 0,
        "default_high_iou": 0,
        "default_mid_iou": 0,
        "default_missed": 0,
        "dense_rescue_candidates": 0,
    }

    for _ in range(int(args.images)):
        gt_count = rng.randint(int(args.min_gt), int(args.max_gt))
        gt_boxes = []
        gt_labels = []
        default = {"boxes": [], "labels": [], "scores": []}
        dense = {"boxes": [], "labels": [], "scores": []}

        for _gt_idx in range(gt_count):
            box = random_box(rng)
            label = rng.randint(1, int(args.num_classes) - 1)
            gt_boxes.append(box)
            gt_labels.append(label)
            counts["gt"] += 1

            mode = rng.random()
            if mode < 0.52:
                pred_box = jitter_box(centered_iou_box(box, rng.uniform(0.78, 0.93)), rng, frac=0.03)
                score = rng.uniform(0.35, 0.95)
                append_prediction(default, pred_box, label, score)
                append_prediction(dense, pred_box, label, score)
                counts["default_high_iou"] += 1
            elif mode < 0.78:
                pred_box = jitter_box(centered_iou_box(box, rng.uniform(0.52, 0.70)), rng, frac=0.03)
                score = rng.uniform(0.25, 0.88)
                append_prediction(default, pred_box, label, score)
                append_prediction(dense, pred_box, label, score)
                counts["default_mid_iou"] += 1
                rescue_box = jitter_box(centered_iou_box(box, rng.uniform(0.79, 0.94)), rng, frac=0.02)
                append_prediction(dense, rescue_box, label, rng.uniform(0.010, 0.045))
                counts["dense_rescue_candidates"] += 1
            elif mode < 0.92:
                rescue_box = jitter_box(centered_iou_box(box, rng.uniform(0.79, 0.94)), rng, frac=0.02)
                append_prediction(dense, rescue_box, label, rng.uniform(0.010, 0.045))
                counts["default_missed"] += 1
                counts["dense_rescue_candidates"] += 1
            else:
                wrong_label = rng.randint(1, int(args.num_classes) - 1)
                if wrong_label == label:
                    wrong_label = 1 + (wrong_label % (int(args.num_classes) - 1))
                pred_box = centered_iou_box(box, rng.uniform(0.78, 0.92))
                append_prediction(default, pred_box, wrong_label, rng.uniform(0.25, 0.85))
                append_prediction(dense, pred_box, wrong_label, rng.uniform(0.25, 0.85))
                rescue_box = jitter_box(centered_iou_box(box, rng.uniform(0.80, 0.94)), rng, frac=0.02)
                append_prediction(dense, rescue_box, label, rng.uniform(0.010, 0.045))
                counts["default_missed"] += 1
                counts["dense_rescue_candidates"] += 1

        for _ in range(int(args.fp_per_image)):
            append_prediction(
                default,
                random_box(rng),
                rng.randint(1, int(args.num_classes) - 1),
                rng.uniform(0.06, 0.80),
            )
        for item in range(len(default["boxes"])):
            if rng.random() < 0.15:
                continue
            append_prediction(
                dense,
                default["boxes"][item],
                default["labels"][item],
                default["scores"][item],
            )
        for _ in range(int(args.dense_fp_per_image)):
            append_prediction(
                dense,
                random_box(rng),
                rng.randint(1, int(args.num_classes) - 1),
                rng.uniform(0.001, 0.045),
            )

        targets.append(
            {
                "boxes": torch.stack(gt_boxes).float(),
                "labels": torch.tensor(gt_labels, dtype=torch.long),
            }
        )
        default_predictions.append(finalize_prediction(default))
        dense_predictions.append(finalize_prediction(dense))

    return default_predictions, dense_predictions, targets, counts


def evaluate_variant(predictions: list[dict], targets: list[dict], args: argparse.Namespace) -> dict:
    return evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=0.5,
        score_threshold=float(args.score_threshold),
        high_conf_threshold=0.7,
        per_class=False,
    )


def attach_noisy_verifier_quality(
    predictions: list[dict],
    targets: list[dict],
    args: argparse.Namespace,
) -> dict:
    rng = random.Random(int(args.seed) + 1009)
    stats = {
        "verifier_mode": str(args.verifier_mode),
        "num_items": 0,
        "true_high_iou": 0,
        "true_low_iou": 0,
        "false_negative_high_iou": 0,
        "false_positive_low_iou": 0,
        "selected_quality_ge_target": 0,
    }
    if args.verifier_mode == "oracle":
        return stats

    for prediction, target in zip(predictions, targets):
        ious, _, _ = best_iou_and_gt(prediction, target, class_aware=True)
        quality = torch.empty_like(ious)
        for idx, iou_value in enumerate(ious.tolist()):
            stats["num_items"] += 1
            is_high = iou_value >= float(args.target_iou)
            is_low = iou_value <= float(args.low_quality_iou)
            if is_high:
                stats["true_high_iou"] += 1
            if is_low:
                stats["true_low_iou"] += 1

            value = iou_value + rng.gauss(0.0, float(args.verifier_noise_std))
            if is_high and rng.random() < float(args.verifier_fn_rate):
                value = rng.uniform(0.0, max(0.0, float(args.target_iou) - 0.05))
                stats["false_negative_high_iou"] += 1
            if is_low and rng.random() < float(args.verifier_fp_rate):
                value = rng.uniform(float(args.target_iou), 1.0)
                stats["false_positive_low_iou"] += 1
            value = max(0.0, min(1.0, value))
            if value >= float(args.target_iou):
                stats["selected_quality_ge_target"] += 1
            quality[idx] = float(value)
        prediction["verifier_quality"] = quality
    return stats


def compact_metrics(metrics: dict) -> dict:
    keys = ["ap50", "ap75", "precision", "recall", "false_positive_rate", "ece", "num_predictions"]
    return {key: metrics.get(key) for key in keys}


def main() -> None:
    args = parse_args()
    default_predictions, dense_predictions, targets, counts = make_synthetic_dataset(args)
    verifier_stats = attach_noisy_verifier_quality(dense_predictions, targets, args)
    baseline = evaluate_variant(default_predictions, targets, args)
    dense_raw = evaluate_variant(dense_predictions, targets, args)

    results = {
        "config": vars(args),
        "counts": counts,
        "verifier_stats": verifier_stats,
        "baseline": compact_metrics(baseline),
        "dense_raw": compact_metrics(dense_raw),
        "variants": {},
    }
    for budget in args.budgets:
        config = ActionSearchConfig(
            score_threshold=float(args.score_threshold),
            target_iou=float(args.target_iou),
            low_quality_iou=float(args.low_quality_iou),
            max_score_delta=float(args.max_score_delta),
            threshold_margin=float(args.threshold_margin),
            max_rescues_per_image=int(budget),
        )
        for mode in ("rollout", "default_plus_selected", "default_replace_or_insert"):
            base_predictions = default_predictions if mode != "rollout" else None
            adjusted, summary = apply_action_search(
                dense_predictions,
                targets,
                config=config,
                oracle_relabel=False,
                only_unmatched_gt=True,
                base_predictions=base_predictions,
                base_mode=mode,
            )
            metrics = evaluate_variant(adjusted, targets, args)
            results["variants"][f"{mode}_b{budget}"] = {
                "metrics": compact_metrics(metrics),
                "summary": summary,
                "delta_vs_baseline": {
                    key: float(metrics[key]) - float(baseline[key])
                    for key in ("ap50", "ap75", "precision", "recall", "false_positive_rate", "num_predictions")
                },
            }

    save_json(results, args.output)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved proxy exploration to {args.output}")


if __name__ == "__main__":
    main()
