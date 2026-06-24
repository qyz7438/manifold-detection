"""Recompute intrinsic dimension with both PCA and TwoNN from saved checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.core.models.box_heads import replace_box_head
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.experiments.schema import validate_experiment_config
from spectral_detection_posttrain.methods.manifold import IntrinsicDimEstimator
from spectral_detection_posttrain.trainers.detection.train_manifold_posttrain import extract_proposal_box_features
from spectral_detection_posttrain.utils.config import load_config


def load_model_for_run(run_dir: Path, baseline_path: Path, config_path: Path, device: torch.device):
    """Load a model with the same box head as used in run_dir."""
    config = load_config(config_path)
    config = validate_experiment_config(config)

    # Try manifold_result.json first (contains box_head_type), fall back to metadata.json.
    result_path = run_dir / "manifold_result.json"
    metadata_path = run_dir / "metadata.json"
    info = {}
    if result_path.exists():
        info.update(json.load(open(result_path)))
    if metadata_path.exists():
        info.update(json.load(open(metadata_path)))
    box_head_type = info.get("box_head_type", "original")
    box_head_rank = info.get("box_head_rank", 128)
    box_head_conv_channels = info.get("box_head_conv_channels", 128)
    box_head_bottleneck_dim = info.get("box_head_bottleneck_dim", 512)
    box_head_attention_channels = info.get("box_head_attention_channels", 64)

    model = build_detector(config)
    model.to(device)

    # Load baseline weights first.
    state = torch.load(baseline_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)

    # Replace head if needed.
    if box_head_type not in ("", "original"):
        replace_box_head(
            model,
            box_head_type,
            rank=box_head_rank,
            conv_channels=box_head_conv_channels,
            bottleneck_dim=box_head_bottleneck_dim,
            attention_channels=box_head_attention_channels,
            copy_compatible_weights=True,
        )

    return model, config, box_head_type


def collect_features(model, val_loader, device, config, use_gt_boxes: bool = False):
    """Collect box features and labels from validation set."""
    model.eval()
    all_features = []
    all_labels = []

    extractor = extract_proposal_box_features if not use_gt_boxes else None

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            try:
                box_features, labels, _ = extractor(model, images, targets)
            except Exception as e:
                print(f"Extraction error: {e}")
                continue

            if box_features.shape[0] == 0:
                continue

            # L2 normalize to match training setting.
            box_features = F.normalize(box_features, dim=-1)
            all_features.append(box_features.cpu())
            all_labels.append(labels.cpu())

    if not all_features:
        return None, None

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def twonn_id(features: torch.Tensor, k: int = 2, max_samples: int = 4096, seed: int = 42) -> float:
    """Memory-efficient TwoNN intrinsic dimension estimator.

    Uses torch.cdist to avoid materializing the (N, N, D) diff tensor.
    Subsamples to max_samples for speed if needed.
    """
    n = features.shape[0]
    if n <= k:
        return float("nan")

    if n > max_samples:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=generator)[:max_samples]
        features = features[idx]
        n = max_samples

    # Compute pairwise distances efficiently.
    dist = torch.cdist(features, features)  # (n, n)
    dist.fill_diagonal_(float("inf"))

    sorted_dist, _ = torch.topk(dist, k=min(k, n - 1), largest=False, dim=-1)
    r1 = sorted_dist[:, 0].clamp_min(1e-12)
    r2 = sorted_dist[:, 1].clamp_min(1e-12)

    mu = r2 / r1
    log_mu = torch.log(mu.clamp_min(1e-12))
    id_est = 1.0 / log_mu.mean()
    return float(id_est.clamp_min(1.0).item())


def compute_ids(features, labels, num_classes):
    """Compute PCA and TwoNN IDs for overall, foreground, and per-class."""
    results = {}
    fg_mask = labels >= 1

    pca_estimator = IntrinsicDimEstimator(method="pca", pca_variance_threshold=0.90)

    # PCA IDs.
    if features.shape[0] >= 2:
        results["pca_overall"] = float(pca_estimator.estimate_id(features).item())
        results["twonn_overall"] = twonn_id(features)
    else:
        results["pca_overall"] = float("nan")
        results["twonn_overall"] = float("nan")

    if fg_mask.sum() >= 2:
        fg_features = features[fg_mask]
        results["pca_foreground"] = float(pca_estimator.estimate_id(fg_features).item())
        results["twonn_foreground"] = twonn_id(fg_features)
    else:
        results["pca_foreground"] = float("nan")
        results["twonn_foreground"] = float("nan")

    per_class_pca = {}
    per_class_twonn = {}
    for c in range(1, num_classes):
        mask = labels == c
        if mask.sum() >= 2:
            class_features = features[mask]
            per_class_pca[str(c)] = float(pca_estimator.estimate_id(class_features).item())
            per_class_twonn[str(c)] = twonn_id(class_features, max_samples=512)
        else:
            per_class_pca[str(c)] = float("nan")
            per_class_twonn[str(c)] = float("nan")
    results["pca_per_class"] = per_class_pca
    results["twonn_per_class"] = per_class_twonn

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", default=["checkpoint_initial.pth", "checkpoint_last.pth"])
    parser.add_argument("--output", default="id_recompute.json")
    parser.add_argument("--use-gt-boxes", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    config = validate_experiment_config(config)
    num_classes = int(config["model"]["num_classes"])

    _, val_loader = build_detection_loaders(config)

    all_results = {}
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        run_name = run_dir.name
        all_results[run_name] = {}

        model, _, box_head_type = load_model_for_run(run_dir, args.baseline, args.config, device)

        for ckpt_name in args.checkpoints:
            ckpt_path = run_dir / ckpt_name
            if not ckpt_path.exists():
                continue

            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()

            print(f"Collecting features for {run_name} / {ckpt_name} ...")
            features, labels = collect_features(model, val_loader, device, config, args.use_gt_boxes)
            if features is None:
                continue

            print(f"  features: {features.shape}, labels: {labels.shape}")
            ids = compute_ids(features, labels, num_classes)
            all_results[run_name][ckpt_name] = ids
            print(f"  PCA: overall={ids['pca_overall']:.1f}, fg={ids['pca_foreground']:.1f}")
            print(f"  TwoNN: overall={ids['twonn_overall']:.1f}, fg={ids['twonn_foreground']:.1f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
