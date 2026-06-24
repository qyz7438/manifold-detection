"""Round 2.11 post-training runner — second-stage from baseline checkpoint."""
from __future__ import annotations

import argparse
import json
import yaml

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.rlvr.round211_spatial_verifier import center_size_reward, iou_reward
from spectral_detection_posttrain.spectral.round211_spectral_gate import shuffled_scores, spectral_gate_score
from spectral_detection_posttrain.utils.io import append_jsonl, ensure_run_dir, load_checkpoint, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


MODE_CHOICES = ["eval_only", "detection_only", "spatial", "spatial_spectral_loggate", "spatial_shuffled_spectral"]


def _to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def _freeze_except_box_head(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.roi_heads.box_head.parameters():
        param.requires_grad = True
    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = True


def _extract_roi_boxes(outputs: list[dict], targets: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    all_pred = []
    all_gt = []
    for out, tgt in zip(outputs, targets):
        boxes = out.get("boxes", None)
        if boxes is not None and boxes.numel() > 0:
            all_pred.append(boxes.cpu())
            all_gt.append(tgt["boxes"].cpu() if torch.is_tensor(tgt["boxes"]) else torch.tensor(tgt["boxes"]))
    if not all_pred:
        return torch.zeros((0, 4)), torch.zeros((0, 4))
    return torch.cat(all_pred), torch.cat(all_gt)


def _eval_and_save(model, val_loader, device, config, run_dir, mode, log_entry=None):
    model.eval()
    predictions, targets_list = [], []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])
    eval_cfg = config.get("eval", {})
    metrics = evaluate_detection_predictions(
        predictions, targets_list,
        iou_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        high_conf_threshold=float(eval_cfg.get("high_conf_threshold", 0.7)),
    )
    metrics["mode"] = mode
    save_json(metrics, run_dir / "eval_metrics.json")
    if log_entry:
        log_entry["AP50"] = metrics.get("AP50", None)
        log_entry["AP75"] = metrics.get("AP75", None)
        append_jsonl(log_entry, run_dir / "posttrain_log.jsonl")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", required=True, choices=MODE_CHOICES)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["seed"] = args.seed

    set_seed(args.seed)
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)

    batch_size = int(config["train"].get("batch_size", 2))
    train_loader, val_loader = build_voc_detection_loaders(
        config, limit_train=args.limit_train, limit_val=args.limit_val, batch_size=batch_size)

    model = build_detector(config).to(device)
    load_checkpoint(model, args.checkpoint, device)

    if args.mode == "eval_only":
        metrics = _eval_and_save(model, val_loader, device, config, run_dir, args.mode)
        print(f"eval_only AP50={metrics.get('AP50', 'N/A')}  AP75={metrics.get('AP75', 'N/A')}")
        return

    _freeze_except_box_head(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable params after freezing")

    pt_cfg = config.get("posttrain", {})
    optimizer = torch.optim.SGD(trainable_params, lr=float(pt_cfg.get("lr", 0.001)),
                                momentum=0.9, weight_decay=0.0005)
    spatial_weight = float(pt_cfg.get("spatial_weight", 0.1))

    use_spatial = args.mode in ("spatial", "spatial_spectral_loggate", "spatial_shuffled_spectral")
    use_spectral = args.mode == "spatial_spectral_loggate"
    use_shuffled = args.mode == "spatial_shuffled_spectral"

    log_entries = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_val = 0.0
        total_seen = 0
        spatial_rewards = []
        spectral_scores = []

        for images, targets in tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = _to_device(targets, device)
            loss_dict = model(images, targets)
            det_loss = sum(loss_dict.values())

            extra_loss = torch.tensor(0.0, device=device)
            if use_spatial:
                model.eval()
                with torch.no_grad():
                    outputs = model(images)
                model.train()
                pred_boxes, gt_boxes = _extract_roi_boxes(outputs, targets)
                if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
                    iou_r = iou_reward(pred_boxes, gt_boxes)
                    cs_r = center_size_reward(pred_boxes, gt_boxes)
                    reward = (iou_r + cs_r) / 2.0
                    extra_loss = spatial_weight * (1.0 - reward.mean())
                    spatial_rewards.append(float(reward.mean().item()))

            if use_spectral or use_shuffled:
                model.eval()
                with torch.no_grad():
                    outputs = model(images)
                model.train()
                for out, tgt in zip(outputs, targets):
                    pred_b = out.get("boxes", None)
                    gt_b = tgt["boxes"]
                    if pred_b is not None and pred_b.numel() > 0 and gt_b.numel() > 0:
                        scores = []
                        for pb in pred_b[:min(len(pred_b), 8)]:
                            best_iou = 0.0
                            best_gt = gt_b[0]
                            for gb in gt_b:
                                iou = _box_iou_single(pb, gb)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt = gb
                            score = spectral_gate_score(
                                _crop_roi(images[0], pb), _crop_roi(images[0], best_gt))
                            scores.append(score)
                        if scores:
                            stacked = torch.stack(scores)
                            if use_shuffled:
                                stacked = shuffled_scores(stacked)
                            spectral_scores.append(float(stacked.mean().item()))

            total_loss = det_loss + extra_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            total_loss_val += float(total_loss.item()) * len(images)
            total_seen += len(images)

        avg_loss = total_loss_val / max(1, total_seen)
        log_entry = {
            "epoch": epoch, "train_loss": avg_loss, "mode": args.mode,
            "spatial_reward_mean": sum(spatial_rewards) / max(1, len(spatial_rewards)) if spatial_rewards else None,
            "spectral_score_mean": sum(spectral_scores) / max(1, len(spectral_scores)) if spectral_scores else None,
        }
        log_entries.append(log_entry)
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch, "mode": args.mode})

    metrics = _eval_and_save(model, val_loader, device, config, run_dir, args.mode,
                             log_entries[-1] if log_entries else None)
    save_json({"entries": log_entries, "final": {k: v for k, v in metrics.items()}},
              run_dir / "posttrain_log.json")
    metric_ap50 = metrics.get("ap50", metrics.get("AP50", "N/A"))
    metric_ap75 = metrics.get("ap75", metrics.get("AP75", "N/A"))
    print(f"{args.mode} AP50={metric_ap50}  AP75={metric_ap75}")


def _box_iou_single(a: torch.Tensor, b: torch.Tensor) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _crop_roi(image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    _, h, w = image.shape
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return torch.zeros(3, 32, 32, device=image.device, dtype=image.dtype)
    crop = image[:, y1:y2, x1:x2]
    return torch.nn.functional.interpolate(crop.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False).squeeze(0)


if __name__ == "__main__":
    main()
