from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.core.models.spectral_quality_head import (
    SpectralQualityHead,
    build_quality_features,
    normalize_r_amp,
)
from spectral_detection_posttrain.signals.fft.roi_spectral_dataset import apply_nms_to_prediction, load_candidate_cache
from spectral_detection_posttrain.signals.fft.spectral_reward import auc_tp_vs_fp
from spectral_detection_posttrain.utils.config import load_config, save_config
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector candidate reranking with oracle R_amp or learned q_spec.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", default="learned", choices=["baseline", "oracle_ramp", "learned"])
    parser.add_argument("--quality-checkpoint", default=None)
    parser.add_argument("--normalization-cache", default=None)
    parser.add_argument("--combine", default="multiply", choices=["multiply", "blend"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    return parser.parse_args()


def _compute_r_amp_stats_from_cache(path: str | Path | None, payload: dict) -> dict:
    source = load_candidate_cache(path) if path else payload
    values = []
    for sample in source["samples"]:
        raw = sample["raw_r_amp"].float()
        if raw.numel() > 0:
            values.append(raw)
    if not values:
        return {"mode": "minmax", "min": 0.0, "max": 1.0}
    all_values = torch.cat(values)
    return {"mode": "minmax", "min": float(all_values.min().item()), "max": float(all_values.max().item())}


def _binary_ece(scores: list[float], labels: list[int], bins: int = 10) -> float | None:
    if not scores:
        return None
    score_tensor = torch.tensor(scores)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    ece = 0.0
    for bin_idx in range(bins):
        left = bin_idx / bins
        right = (bin_idx + 1) / bins
        mask = (score_tensor >= left) & (score_tensor < right if bin_idx < bins - 1 else score_tensor <= right)
        if not mask.any():
            continue
        conf = score_tensor[mask].mean()
        acc = label_tensor[mask].mean()
        ece += float(mask.float().mean().item()) * abs(float(conf.item()) - float(acc.item()))
    return ece


@torch.no_grad()
def _load_quality_head(path: str, device: torch.device) -> tuple[SpectralQualityHead, dict]:
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint["metadata"]
    head = SpectralQualityHead(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata.get("quality_head", {}).get("hidden_dim", 256)),
        dropout=float(metadata.get("quality_head", {}).get("dropout", 0.1)),
    ).to(device)
    head.load_state_dict(checkpoint["model"])
    head.eval()
    return head, metadata


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")

    payload = load_candidate_cache(args.candidates)
    device = resolve_device(config)
    quality_head = None
    feature_mode = None
    r_amp_stats = _compute_r_amp_stats_from_cache(args.normalization_cache, payload)
    if args.method == "learned":
        if not args.quality_checkpoint:
            raise ValueError("--quality-checkpoint is required for learned reranking.")
        quality_head, metadata = _load_quality_head(args.quality_checkpoint, device)
        feature_mode = str(metadata["feature_mode"])
        r_amp_stats = metadata.get("r_amp_stats", r_amp_stats)

    predictions = []
    targets = []
    q_tp: list[float] = []
    q_fp: list[float] = []
    q_scores: list[float] = []
    tp_labels: list[int] = []
    q_ious: list[float] = []

    for sample in payload["samples"]:
        boxes = sample["boxes"].float()
        labels = sample["labels"].long()
        scores = sample["scores"].float()
        if len(scores) == 0:
            predictions.append({"boxes": boxes, "labels": labels, "scores": scores})
            targets.append(sample["target"])
            continue

        if args.method == "baseline":
            q_spec = torch.ones_like(scores)
        elif args.method == "oracle_ramp":
            q_spec = normalize_r_amp(sample["raw_r_amp"].float(), r_amp_stats)
        else:
            assert quality_head is not None and feature_mode is not None
            features = build_quality_features(sample, feature_mode).to(device)
            q_spec = quality_head.predict_quality(features).detach().cpu()

        if args.combine == "multiply":
            final_scores = scores * q_spec
        else:
            final_scores = float(args.alpha) * scores + (1.0 - float(args.alpha)) * q_spec

        tp_mask = sample["is_tp"].bool()
        q_tp.extend(q_spec[tp_mask].tolist())
        q_fp.extend(q_spec[~tp_mask].tolist())
        q_scores.extend(q_spec.tolist())
        tp_labels.extend(tp_mask.long().tolist())
        q_ious.extend(sample["ious"].float().tolist())

        prediction = apply_nms_to_prediction(
            {"boxes": boxes, "labels": labels, "scores": final_scores},
            iou_threshold=float(args.nms_threshold),
            score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        )
        predictions.append(prediction)
        targets.append(sample["target"])

    metrics = evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
    )
    high_conf_metrics = evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
    )
    corr = None
    if len(q_scores) > 1:
        q_tensor = torch.tensor(q_scores)
        iou_tensor = torch.tensor(q_ious)
        if q_tensor.std(unbiased=False) > 1e-6 and iou_tensor.std(unbiased=False) > 1e-6:
            corr = float(torch.corrcoef(torch.stack([q_tensor, iou_tensor]))[0, 1].item())
    metrics.update(
        {
            "method": args.method,
            "combine": args.combine,
            "alpha": args.alpha,
            "candidate_cache": str(args.candidates),
            "patch_mode": payload["meta"].get("patch_mode", "unknown"),
            "patch_type": payload["meta"].get("patch_type", "unknown"),
            "q_spec_auc_tp_vs_fp": auc_tp_vs_fp(q_tp, q_fp),
            "q_spec_iou_corr": corr,
            "q_spec_ece": _binary_ece(q_scores, tp_labels),
            "mean_q_tp": sum(q_tp) / len(q_tp) if q_tp else None,
            "mean_q_fp": sum(q_fp) / len(q_fp) if q_fp else None,
            "high_conf_fn_rate": high_conf_metrics["miss_rate"],
        }
    )
    save_json(metrics, run_dir / "eval_rerank_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
