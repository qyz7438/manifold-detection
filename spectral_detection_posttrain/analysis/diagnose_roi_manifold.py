r"""Diagnose the ROI-feature manifold for a trained detector.

This script collects foreground ROI features and labels from the training set
(using the detector's own RPN proposals, exactly as the post-train trainer does)
and computes per-class geometric statistics:

* class counts and mean prototypes
* within-class variability (NC1) and effective rank
* pairwise prototype angles (NC2)
* cosine-based confusion / Voronoi cell sizes
* angular margins to the nearest wrong class

The outputs are saved under ``<run_dir>/roi_diagnostics/``:

* ``roi_features.pt`` – collected features, labels, and classifier logits
* ``diagnostics.json`` – scalar and per-class statistics
* ``figures/`` – PCA visualization, confusion heatmap, Voronoi areas, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# These imports are heavy, but they let us reuse the exact proposal-extraction
# path from the post-train trainer.
from spectral_detection_posttrain.core.models.adaptive_etf_predictor import (
    AdaptiveETFClassifier,
)
from spectral_detection_posttrain.core.models.etf_predictor import ETFClassifier
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.datasets.nwpu_vhr10 import NWPU_CLASS_TO_LABEL
from spectral_detection_posttrain.experiments.canonical_runner import (
    build_experiment_model,
    prepare_experiment_from_config,
)
from spectral_detection_posttrain.trainers.detection.train_manifold_posttrain import (
    _set_model_train_for_detection_loss,
    _to_device_targets,
    extract_proposal_box_features,
    warmup_class_centers,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


LABEL_TO_NWPU_CLASS = {v: k for k, v in NWPU_CLASS_TO_LABEL.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ROI feature manifold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-name", default="roi_manifold_diagnostic")
    parser.add_argument("--split", default="val", choices=("train", "val"),
                        help="Which split to use for the confusion matrix. Default is 'val' because "
                             "training-set confusion is usually identity for a well-trained baseline.")
    parser.add_argument("--limit", type=int, default=None, help="Max images to use")
    parser.add_argument("--max-features", type=int, default=50000,
                        help="Max foreground features to keep in memory")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_roi_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect foreground ROI features, labels, and logits.

    Returns ``features, labels, logits`` of shape ``(N, D)``, ``(N,)``, ``(N, C)``
    where ``N <= max_features``.
    """
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    _set_model_train_for_detection_loss(model)
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = _to_device_targets(targets, device)

            box_features, labels, _ = extract_proposal_box_features(
                model, images, targets, return_layers=False
            )
            fg_mask = labels >= 1
            if not fg_mask.any():
                continue
            feats = box_features[fg_mask]
            labels_fg = labels[fg_mask]

            # Classifier logits using the current head.
            logits = model.roi_heads.box_predictor(feats)
            if isinstance(logits, tuple):
                logits = logits[0]

            all_feats.append(feats.cpu())
            all_labels.append(labels_fg.cpu())
            all_logits.append(logits.cpu())

            if sum(t.shape[0] for t in all_feats) >= max_features:
                break

    features = torch.cat(all_feats, dim=0)[:max_features]
    labels = torch.cat(all_labels, dim=0)[:max_features]
    logits = torch.cat(all_logits, dim=0)[:max_features]
    return features, labels, logits


def compute_class_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor | None,
    num_classes: int,
) -> dict:
    """Compute per-class and global manifold statistics."""
    C = num_classes - 1  # foreground classes only
    device = features.device

    # Foreground only.
    fg_mask = labels >= 1
    feats = features[fg_mask]
    labs = labels[fg_mask]
    fg_logits = logits[fg_mask] if logits is not None else None
    if feats.shape[0] == 0:
        raise ValueError("No foreground features collected")

    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    means = torch.zeros(C, features.shape[-1], device=device)
    variances = torch.zeros(C, device=device)

    for c in range(1, num_classes):
        mask = labs == c
        counts[c] = int(mask.sum().item())
        if mask.any():
            means[c - 1] = feats[mask].mean(dim=0)
            variances[c - 1] = (feats[mask] - means[c - 1]).pow(2).sum(dim=1).mean()

    # Normalized prototypes.
    means_n = F.normalize(means, dim=1)
    gram = means_n @ means_n.T
    pairwise_angles = torch.acos(gram.clamp(-1 + 1e-6, 1 - 1e-6)) * 180.0 / np.pi

    # Effective rank of the class-mean matrix.
    s = torch.linalg.svdvals(means)
    effective_rank = float((s / s.sum()).pow(2).sum().pow(-1).item())

    # Confusion matrix from the original classifier logits if available; otherwise
    # fall back to nearest class-mean prototype assignment.
    if fg_logits is not None and fg_logits.shape[-1] == num_classes:
        # Use foreground logits (drop background column) and argmax.
        pred = fg_logits[:, 1:].argmax(dim=1) + 1
    else:
        feats_n = F.normalize(feats, dim=1)
        sims = feats_n @ means_n.T  # (N, C)
        pred = sims.argmax(dim=1) + 1  # foreground class ids
    confusion = torch.zeros(C, C, dtype=torch.long, device=device)
    for i in range(C):
        true_c = i + 1
        mask = labs == true_c
        if mask.any():
            for j in range(C):
                confusion[i, j] = int((pred[mask] == (j + 1)).sum().item())
    confusion_norm = confusion.float() / confusion.sum(dim=1, keepdim=True).clamp_min(1)

    # Voronoi cell sizes: fraction of samples whose nearest prototype is each class.
    feats_n = F.normalize(feats, dim=1)
    sims = feats_n @ means_n.T
    pred_voronoi = sims.argmax(dim=1) + 1
    voronoi_counts = torch.zeros(C, dtype=torch.long, device=device)
    for c in range(C):
        voronoi_counts[c] = int((pred_voronoi == (c + 1)).sum().item())
    voronoi_frac = voronoi_counts.float() / voronoi_counts.sum().clamp_min(1)

    # Per-class minimum angle to any wrong prototype.
    min_wrong_angle = torch.full((C,), 180.0, device=device)
    for i in range(C):
        angles_row = pairwise_angles[i]
        mask = torch.ones(C, dtype=torch.bool, device=device)
        mask[i] = False
        if mask.any():
            min_wrong_angle[i] = angles_row[mask].min()

    return {
        "counts": counts,
        "means": means,
        "means_normalized": means_n,
        "variances": variances,
        "gram": gram,
        "pairwise_angles": pairwise_angles,
        "effective_rank": effective_rank,
        "confusion": confusion,
        "confusion_normalized": confusion_norm,
        "voronoi_fractions": voronoi_frac,
        "min_wrong_angle": min_wrong_angle,
    }


def class_names(num_classes: int) -> list[str]:
    return ["background"] + [
        LABEL_TO_NWPU_CLASS.get(c, f"class_{c}") for c in range(1, num_classes)
    ]


def build_report(stats: dict, num_classes: int) -> dict:
    names = class_names(num_classes)
    C = num_classes - 1
    report: dict = {
        "effective_rank": round(float(stats["effective_rank"]), 4),
        "num_foreground": C,
        "classes": {},
        "global": {},
    }
    for c in range(1, num_classes):
        report["classes"][names[c]] = {
            "count": int(stats["counts"][c].item()),
            "nc1_variance": round(float(stats["variances"][c - 1].item()), 6),
            "min_wrong_angle_deg": round(float(stats["min_wrong_angle"][c - 1].item()), 2),
            "voronoi_fraction": round(float(stats["voronoi_fractions"][c - 1].item()), 4),
        }

    # Pairwise angle matrix.
    angle_mat = stats["pairwise_angles"].cpu().numpy()
    report["global"]["mean_pairwise_angle_deg"] = round(
        float(angle_mat[~np.eye(C, dtype=bool)].mean()), 2
    )
    report["global"]["std_pairwise_angle_deg"] = round(
        float(angle_mat[~np.eye(C, dtype=bool)].std()), 2
    )

    # Confusion off-diagonal sum.
    conf = stats["confusion_normalized"].cpu().numpy()
    report["global"]["mean_nearest_prototype_confusion"] = round(
        float(conf[~np.eye(C, dtype=bool)].mean()), 4
    )
    return report


def save_figures(stats: dict, features: torch.Tensor, labels: torch.Tensor, out_dir: Path, num_classes: int) -> None:
    """Save manifold visualizations using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figures: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    names = class_names(num_classes)
    C = num_classes - 1
    fg_mask = labels >= 1
    feats_fg = features[fg_mask].numpy()
    labs_fg = labels[fg_mask].numpy()

    # PCA 2D projection.
    if feats_fg.shape[0] > C and feats_fg.shape[1] >= 2:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(feats_fg)
        plt.figure(figsize=(8, 8))
        for c in range(1, num_classes):
            mask = labs_fg == c
            if mask.any():
                plt.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.6, label=names[c])
        plt.legend(markerscale=3)
        plt.title("ROI features PCA (foreground)")
        plt.tight_layout()
        plt.savefig(out_dir / "pca_foreground.png", dpi=150)
        plt.close()

    # Pairwise angle heatmap.
    angle_mat = stats["pairwise_angles"].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(angle_mat, vmin=0, vmax=180)
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(names[1:], rotation=45, ha="right")
    ax.set_yticklabels(names[1:])
    ax.set_title("Prototype pairwise angle (degrees)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "pairwise_angles.png", dpi=150)
    plt.close()

    # Confusion heatmap (normalized).
    conf = stats["confusion_normalized"].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf, vmin=0, vmax=1)
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(names[1:], rotation=45, ha="right")
    ax.set_yticklabels(names[1:])
    ax.set_title("Nearest-prototype confusion (row-normalized)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_nearest_prototype.png", dpi=150)
    plt.close()

    # Voronoi fractions + NC1 + margin.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = names[1:]
    axes[0].bar(xs, stats["voronoi_fractions"].cpu().numpy())
    axes[0].set_title("Voronoi cell fraction (nearest prototype)")
    axes[0].set_ylabel("fraction")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(xs, stats["variances"].cpu().numpy())
    axes[1].set_title("Within-class variance (NC1)")
    axes[1].set_ylabel("variance")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(xs, stats["min_wrong_angle"].cpu().numpy())
    axes[2].set_title("Min angle to wrong prototype")
    axes[2].set_ylabel("degrees")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(out_dir / "per_class_stats.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["posttrain"]["batch_size"] = args.batch_size
    set_seed(args.seed)

    context = prepare_experiment_from_config(
        config,
        args.config,
        args.run_name,
        phase="roi_diagnostic",
        checkpoint_path=args.baseline,
    )
    config = context.config
    run_dir = context.run_dir

    train_loader, val_loader = build_detection_loaders(
        config,
        limit_train=args.limit if args.split == "train" else None,
        limit_val=args.limit if args.split == "val" else None,
        batch_size=int(config["posttrain"].get("batch_size", 1)),
    )
    loader = train_loader if args.split == "train" else val_loader
    device = resolve_device(config)

    model = build_experiment_model(context, checkpoint_path=args.baseline, device=device, pretrained=False)
    num_classes = int(config["model"]["num_classes"])

    print(f"Collecting ROI features from {args.split} set (max={args.max_features})...")
    features, labels, logits = collect_roi_features(model, loader, device, args.max_features)
    print(f"Collected {features.shape[0]} foreground features")

    print("Computing class statistics...")
    stats = compute_class_statistics(features, labels, logits, num_classes)
    report = build_report(stats, num_classes)

    # Also collect class centers in the same way the trainer does (for downstream init).
    print("Computing warmup class centers...")
    centers, counts = warmup_class_centers(
        model, loader, device, num_classes, max_batches=len(loader),
        normalize=False, use_gt_boxes=False,
    )

    output_dir = run_dir / "roi_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": features,
            "labels": labels,
            "logits": logits,
            "centers": centers,
            "counts": counts,
            "stats": {k: v for k, v in stats.items() if isinstance(v, torch.Tensor)},
        },
        output_dir / "roi_features.pt",
    )

    # Save the confusion matrix as its own tensor for the adaptive-ETF trainer.
    torch.save(stats["confusion_normalized"].cpu(), output_dir / "target_confusion.pt")

    save_json(report, output_dir / "diagnostics.json")
    with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saving figures...")
    save_figures(stats, features, labels, output_dir / "figures", num_classes)

    print(f"Done. Outputs saved to {output_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
