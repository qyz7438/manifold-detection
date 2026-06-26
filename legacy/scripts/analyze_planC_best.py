"""Analyze the best Plan C follow-up results.

Produces:
- Aggregate metric tables for baseline, e2e_sup_logit, sup_feature_scale10.
- Per-image improvement/degradation rankings.
- Visual comparisons for representative images.
- Epoch-wise metric curves for the two best configs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.trainers.segmentation.action_local_adapter import (
    load_baseline,
    install_seg_adapter,
)
from spectral_detection_posttrain.datasets.penn_fudan_seg import build_seg_loaders
from spectral_detection_posttrain.methods.segmentation.eval_segmentation import eval_segmentation
from spectral_detection_posttrain.utils.seed import set_seed


ROOT = Path("E:/CLIproject/RLimage")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
BASE_CKPT = ROOT / "runs/round41_baseline_s42/checkpoint_last.pth"
OUT_DIR = ROOT / "runs" / "planC_best_analysis"


CONFIGS = {
    "baseline": {
        "action": "feature",
        "scale": 1.0,
        "unfreeze": False,
        "adapter_ckpt": None,
    },
    "e2e_sup_logit": {
        "action": "logit",
        "scale": 1.0,
        "unfreeze": True,
        "adapter_ckpt": ROOT / "runs/planC_followup_e2e_sup_logit_3ep_ref/checkpoint_best.pth",
    },
    "sup_feature_scale10": {
        "action": "feature",
        "scale": 1.0,
        "unfreeze": False,
        "adapter_ckpt": ROOT / "runs/planC_followup_sup_feature_scale10_3ep/checkpoint_best.pth",
    },
}


def build_model(cfg: dict):
    base_model = load_baseline(str(BASE_CKPT), NUM_CLASSES, DEVICE)
    policy = install_seg_adapter(
        base_model,
        action_type=cfg["action"],
        num_classes=NUM_CLASSES,
        in_ch=2048,
        adapter_kwargs={"scale": cfg["scale"]},
    ).to(DEVICE)
    if cfg["unfreeze"]:
        for p in policy.base_model.parameters():
            p.requires_grad = True
    if cfg["adapter_ckpt"] is not None:
        ckpt = torch.load(cfg["adapter_ckpt"], map_location=DEVICE)
        policy.adapter.load_state_dict(ckpt)
    policy.eval()
    return policy


def evaluate_per_image(model, val_loader, name: str):
    records = []
    dataset = val_loader.dataset
    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            x = torch.stack([img.to(DEVICE) for img in images])
            target = torch.stack([m.to(DEVICE) for m in masks])
            out = model(x)["out"]
            target_r = F.interpolate(
                target.unsqueeze(1).float(),
                size=out.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
            metrics = eval_segmentation(out, target_r, NUM_CLASSES)
            per_class = metrics["per_class_iou"].cpu().numpy()
            records.append({
                "idx": idx,
                "filename": str(dataset.images[idx]),
                "mIoU": float(metrics["mIoU"]),
                "boundary_iou": float(metrics["boundary_iou"]),
                "pixel_accuracy": float(metrics["pixel_accuracy"]),
                "bg_iou": None if math.isnan(float(per_class[0])) else float(per_class[0]),
                "person_iou": None if math.isnan(float(per_class[1])) else float(per_class[1]),
            })
    return records


def aggregate(records):
    keys = ["mIoU", "boundary_iou", "pixel_accuracy", "bg_iou", "person_iou"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in records if r[k] is not None]
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return agg


def print_table(rows, title):
    print(f"\n{title}")
    print("-" * 80)
    header = f"{'name':<22} {'mIoU':>8} {'bIoU':>8} {'pAcc':>8} {'bgIoU':>8} {'personIoU':>10}"
    print(header)
    print("-" * 80)
    for name, agg in rows:
        print(
            f"{name:<22} "
            f"{agg['mIoU']['mean']:>8.4f} "
            f"{agg['boundary_iou']['mean']:>8.4f} "
            f"{agg['pixel_accuracy']['mean']:>8.4f} "
            f"{agg['bg_iou']['mean']:>8.4f} "
            f"{agg['person_iou']['mean']:>10.4f}"
        )


def rank_delta(base_records, method_records, key="mIoU"):
    deltas = []
    for b, m in zip(base_records, method_records):
        deltas.append({
            "idx": b["idx"],
            "filename": b["filename"],
            "base": b[key],
            "method": m[key],
            "delta": m[key] - b[key],
        })
    deltas.sort(key=lambda x: x["delta"], reverse=True)
    return deltas


def visualize_comparison(base_model, models, val_loader, indices, out_path):
    dataset = val_loader.dataset
    n = len(indices)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    cmap = plt.get_cmap("tab10")
    for row, idx in enumerate(indices):
        img_t, mask_t = dataset[idx]
        x = img_t.unsqueeze(0).to(DEVICE)
        target = mask_t.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            base_out = base_model(x)["out"]
            preds = {"baseline": base_out}
            for name, m in models.items():
                preds[name] = m(x)["out"]

        # resize target to match output size
        h, w = base_out.shape[-2:]
        target_r = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()

        ax = axes[row, 0]
        ax.imshow(img_t.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"image #{idx}")
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(target_r[0].cpu().numpy(), vmin=0, vmax=NUM_CLASSES - 1)
        ax.set_title("GT")
        ax.axis("off")

        for col, (name, pred) in enumerate(preds.items(), start=2):
            ax = axes[row, col]
            ax.imshow(pred.argmax(1)[0].cpu().numpy(), vmin=0, vmax=NUM_CLASSES - 1)
            ax.set_title(name)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure: {out_path}")


def plot_epoch_curves(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = ["e2e_sup_logit", "sup_feature_scale10"]
    run_map = {
        "e2e_sup_logit": "planC_followup_e2e_sup_logit_3ep_ref",
        "sup_feature_scale10": "planC_followup_sup_feature_scale10_3ep",
    }
    keys = ["mIoU", "boundary_iou", "pixel_accuracy"]
    titles = ["mIoU", "boundary IoU", "pixel accuracy"]

    for name in names:
        summary_path = ROOT / "runs" / run_map[name] / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.load(open(summary_path))
        train_log = summary.get("train_log", [])
        # final metrics per epoch are in eval_metrics_e*.json; use train_log epoch field if available
        epochs = list(range(1, len(train_log) + 1))
        metrics_by_epoch = {k: [] for k in keys}
        for epoch in epochs:
            eval_path = ROOT / "runs" / run_map[name] / f"eval_metrics_e{epoch}.json"
            d = json.load(open(eval_path))
            for k in keys:
                metrics_by_epoch[k].append(d[k])

        for ax, k, t in zip(axes, keys, titles):
            ax.plot(epochs, metrics_by_epoch[k], marker="o", label=name)
            ax.set_title(t)
            ax.set_xlabel("epoch")
            ax.set_ylabel(t)
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved epoch curve: {out_path}")


def main():
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, val_loader = build_seg_loaders("./data/PennFudanPed", max_size=320, batch_size=1, num_workers=0)

    models = {}
    all_records = {}
    rows = []
    for name, cfg in CONFIGS.items():
        print(f"\nBuilding model: {name}")
        model = build_model(cfg)
        models[name] = model
        print(f"Evaluating {name} on val set...")
        records = evaluate_per_image(model, val_loader, name)
        all_records[name] = records
        agg = aggregate(records)
        rows.append((name, agg))

    print_table(rows, "Aggregate validation metrics")

    # Save aggregate summary
    summary = {name: aggregate(recs) for name, recs in all_records.items()}
    (OUT_DIR / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2))

    # Rankings vs baseline
    base_records = all_records["baseline"]
    rankings = {}
    for name in ["e2e_sup_logit", "sup_feature_scale10"]:
        deltas = rank_delta(base_records, all_records[name], key="mIoU")
        rankings[name] = deltas
        print(f"\n{name}: top-5 mIoU improvements")
        for d in deltas[:5]:
            print(f"  idx={d['idx']:3d}  base={d['base']:.4f}  {name}={d['method']:.4f}  Δ={d['delta']:+.4f}  {Path(d['filename']).name}")
        print(f"{name}: top-5 mIoU degradations")
        for d in deltas[-5:]:
            print(f"  idx={d['idx']:3d}  base={d['base']:.4f}  {name}={d['method']:.4f}  Δ={d['delta']:+.4f}  {Path(d['filename']).name}")

    (OUT_DIR / "per_image_rankings.json").write_text(json.dumps(rankings, indent=2))

    # Visual comparisons: top improvers and worst degradations for each method
    for name in ["e2e_sup_logit", "sup_feature_scale10"]:
        deltas = rankings[name]
        top_improve = [d["idx"] for d in deltas[:3]]
        top_degrade = [d["idx"] for d in deltas[-3:]]
        visualize_comparison(
            models["baseline"],
            {name: models[name]},
            val_loader,
            top_improve,
            OUT_DIR / f"{name}_top_improvers.png",
        )
        visualize_comparison(
            models["baseline"],
            {name: models[name]},
            val_loader,
            top_degrade,
            OUT_DIR / f"{name}_top_degradations.png",
        )

    # Epoch curves
    plot_epoch_curves(OUT_DIR / "epoch_curves.png")

    print(f"\nAll analysis outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
