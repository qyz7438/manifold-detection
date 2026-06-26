from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectral_detection_posttrain.core.models.spectral_quality_head import (
    SpectralQualityHead,
    build_quality_features,
    make_quality_targets,
    pairwise_ranking_loss,
    quality_input_dim,
)
from spectral_detection_posttrain.signals.fft.roi_spectral_dataset import RoiSpectralCandidateDataset
from spectral_detection_posttrain.signals.fft.spectral_reward import auc_tp_vs_fp
from spectral_detection_posttrain.utils.config import load_config, save_config
from spectral_detection_posttrain.utils.io import append_jsonl, ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline Spectral Quality Head for detector candidate reranking.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--val-candidates", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--feature-mode",
        default=None,
        choices=["roi", "amp", "amp_structure", "roi_amp", "roi_amp_structure"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def _collate_samples(batch: list[dict]) -> list[dict]:
    return batch


def _compute_r_amp_stats(dataset: RoiSpectralCandidateDataset, mode: str = "minmax") -> dict:
    values = []
    for sample in dataset.samples:
        raw = sample["raw_r_amp"].float()
        if raw.numel() > 0:
            values.append(raw)
    if not values:
        return {"mode": mode, "min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0}
    all_values = torch.cat(values)
    return {
        "mode": mode,
        "min": float(all_values.min().item()),
        "max": float(all_values.max().item()),
        "mean": float(all_values.mean().item()),
        "std": float(all_values.std(unbiased=False).clamp(min=1e-6).item()),
    }


@torch.no_grad()
def evaluate_quality_head(
    head: SpectralQualityHead,
    dataset: RoiSpectralCandidateDataset,
    feature_mode: str,
    stats: dict,
    device: torch.device,
) -> dict:
    head.eval()
    losses = []
    q_tp = []
    q_fp = []
    ious = []
    qualities = []
    for sample in dataset.samples:
        if len(sample["scores"]) == 0:
            continue
        features = build_quality_features(sample, feature_mode).to(device)
        targets = make_quality_targets(sample, stats).to(device)
        logits = head(features)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        losses.append(float(loss.item()))
        q_spec = torch.sigmoid(logits).detach().cpu()
        tp_mask = sample["is_tp"].bool()
        q_tp.extend(q_spec[tp_mask].tolist())
        q_fp.extend(q_spec[~tp_mask].tolist())
        ious.extend(sample["ious"].float().tolist())
        qualities.extend(q_spec.tolist())
    corr = None
    if len(qualities) > 1:
        q_tensor = torch.tensor(qualities)
        iou_tensor = torch.tensor(ious)
        if q_tensor.std(unbiased=False) > 1e-6 and iou_tensor.std(unbiased=False) > 1e-6:
            corr = float(torch.corrcoef(torch.stack([q_tensor, iou_tensor]))[0, 1].item())
    return {
        "loss_quality": sum(losses) / max(1, len(losses)),
        "q_spec_auc_tp_vs_fp": auc_tp_vs_fp(q_tp, q_fp),
        "mean_q_tp": sum(q_tp) / len(q_tp) if q_tp else None,
        "mean_q_fp": sum(q_fp) / len(q_fp) if q_fp else None,
        "q_spec_iou_corr": corr,
        "num_tp": len(q_tp),
        "num_fp": len(q_fp),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    quality_cfg = config.setdefault("quality_head", {})
    feature_mode = args.feature_mode or str(quality_cfg.get("feature_mode", "roi_amp_structure"))
    if args.epochs is not None:
        quality_cfg["epochs"] = args.epochs
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")

    train_dataset = RoiSpectralCandidateDataset(args.train_candidates)
    val_dataset = RoiSpectralCandidateDataset(args.val_candidates)
    stats = _compute_r_amp_stats(train_dataset, mode=str(quality_cfg.get("r_amp_norm", "minmax")))
    input_dim = quality_input_dim(train_dataset.meta, feature_mode)
    if input_dim <= 0:
        raise ValueError(f"Invalid quality-head input_dim={input_dim} for mode={feature_mode}")

    device = resolve_device(config)
    head = SpectralQualityHead(
        input_dim=input_dim,
        hidden_dim=int(quality_cfg.get("hidden_dim", 256)),
        dropout=float(quality_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(quality_cfg.get("lr", 0.001)),
        weight_decay=float(quality_cfg.get("weight_decay", 0.0001)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(quality_cfg.get("batch_size", 8)),
        shuffle=True,
        collate_fn=_collate_samples,
    )
    epochs = int(quality_cfg.get("epochs", 10))
    rank_weight = float(quality_cfg.get("rank_weight", 0.5))
    margin = float(quality_cfg.get("rank_margin", 0.2))
    patience = args.early_stopping_patience
    if patience is None:
        patience = quality_cfg.get("early_stopping_patience")
    patience = int(patience) if patience is not None else None
    min_delta = float(args.early_stopping_min_delta)
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    def checkpoint_payload(epoch: int) -> dict:
        return {
            "model": head.state_dict(),
            "metadata": {
                "feature_mode": feature_mode,
                "input_dim": input_dim,
                "r_amp_stats": stats,
                "quality_head": dict(quality_cfg),
                "epoch": epoch,
            },
        }

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_quality = 0.0
        total_rank = 0.0
        seen = 0
        for batch in tqdm(loader, desc=f"{args.run_name} epoch {epoch}/{epochs}"):
            batch_loss = torch.tensor(0.0, device=device)
            batch_quality = torch.tensor(0.0, device=device)
            batch_rank = torch.tensor(0.0, device=device)
            usable = 0
            for sample in batch:
                if len(sample["scores"]) == 0:
                    continue
                features = build_quality_features(sample, feature_mode).to(device)
                targets = make_quality_targets(sample, stats).to(device)
                logits = head(features)
                loss_quality = F.binary_cross_entropy_with_logits(logits, targets)
                loss_rank = pairwise_ranking_loss(logits, sample["is_tp"].to(device), margin=margin)
                loss = loss_quality + rank_weight * loss_rank
                batch_loss = batch_loss + loss
                batch_quality = batch_quality + loss_quality.detach()
                batch_rank = batch_rank + loss_rank.detach()
                usable += 1
            if usable == 0:
                continue
            batch_loss = batch_loss / usable
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            total_loss += float(batch_loss.item()) * usable
            total_quality += float((batch_quality / usable).item()) * usable
            total_rank += float((batch_rank / usable).item()) * usable
            seen += usable

        train_row = {
            "epoch": epoch,
            "loss": total_loss / max(1, seen),
            "loss_quality": total_quality / max(1, seen),
            "loss_rank": total_rank / max(1, seen),
            "feature_mode": feature_mode,
        }
        val_row = evaluate_quality_head(head, val_dataset, feature_mode, stats, device)
        row = {**train_row, **{f"val_{k}": v for k, v in val_row.items()}}
        append_jsonl(row, run_dir / "metrics_train.jsonl")
        print(row)
        val_loss = float(val_row["loss_quality"])
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint_payload(epoch), run_dir / "quality_head_best.pth")
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                stop_row = {
                    "early_stopped": True,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss_quality": best_val_loss,
                    "patience": patience,
                }
                append_jsonl(stop_row, run_dir / "metrics_train.jsonl")
                print(stop_row)
                break

    checkpoint = checkpoint_payload(epoch)
    torch.save(checkpoint, run_dir / "quality_head_last.pth")
    if not (run_dir / "quality_head_best.pth").exists():
        torch.save(checkpoint, run_dir / "quality_head_best.pth")
    save_json(checkpoint["metadata"], run_dir / "quality_head_metadata.json")


if __name__ == "__main__":
    main()
