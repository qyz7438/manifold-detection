r"""Layer-wise + spatial intrinsic-dimension diagnostics.

Extracts features at multiple FPN levels (each is spatial), the standard
MultiScaleRoIAlign output (256x7x7), the box-head fc1/fc2 activations, and the
active-corrected endpoint.  Reports intrinsic dimension globally, per-class,
and optionally per spatial cell in the 7x7 ROI grid.

Example:
    python analyze_spatial_layerwise_geometry.py \
        --config spectral_detection_posttrain/configs/manifold_nwpu.yaml \
        --checkpoint runs/round2100_nwpu_baseline/checkpoint_best.pth \
        --run-name baseline_spatial_id
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.ops as ops

from analyze_layerwise_geometry import (
    _resolve_box_head_layers,
    build_table,
    extract_layerwise_features,
    geometry_for_layer,
    install_active_correction_from_checkpoint_state,
)
from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.methods.manifold.geometry_metrics import estimate_intrinsic_dimension
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--max-geometry-samples", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--id-method", default="pca", choices=["pca", "twonn"])
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-class", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extract-fpn-levels", action="store_true")
    parser.add_argument("--spatial-id-map", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def extract_features_with_fpn_levels(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    max_samples: int,
    extract_fpn_levels: bool,
    spatial_id_map: bool,
    config: dict,
    limit_val: int | None,
) -> dict[str, Any]:
    """Extract roi_pooled, per-FPN-level ROI features, fc1, z, z_corrected."""
    # Reuse existing extraction for roi_pooled / fc1 / z / z_corrected.
    base = extract_layerwise_features(model, val_loader, device, max_samples)

    if not extract_fpn_levels:
        return base

    # Collect per-FPN-level ROI features.
    fpn_buffers: dict[str, list[torch.Tensor]] = {}
    labels_t: torch.Tensor | None = None
    count = 0

    _, val_loader_for_fpn = build_detection_loaders(
        config,
        limit_val=limit_val,
        batch_size=1,
    )

    for images, targets in val_loader_for_fpn:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in tgt.items()} for tgt in targets]

        transformed, _ = model.transform(images, None)
        features = model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, _ = model.rpn(transformed, features, targets)
        sampled_props, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(
            proposals, targets
        )
        labels_t = torch.cat(labels, dim=0)
        fg_mask = labels_t >= 1
        if not fg_mask.any():
            continue

        # Use the first image size as reference for stride calculation.
        img_h, img_w = transformed.image_sizes[0]
        boxes = torch.cat(sampled_props, dim=0)[fg_mask]

        for key, feat in features.items():
            _, _, fh, fw = feat.shape
            stride_h = img_h / fh
            stride_w = img_w / fw
            spatial_scale = 1.0 / ((stride_h + stride_w) / 2.0)

            scaled_boxes = boxes.clone()
            scaled_boxes[:, [0, 2]] /= stride_w
            scaled_boxes[:, [1, 3]] /= stride_h

            roi = ops.roi_align(feat, [scaled_boxes], output_size=(7, 7), spatial_scale=spatial_scale)
            fpn_buffers.setdefault(f"fpn_{key}", []).append(roi.cpu())

        count += fg_mask.sum().item()
        if count >= max_samples:
            break

    # Trim and merge.
    out = dict(base)
    for key, buf in fpn_buffers.items():
        out[key] = torch.cat(buf, dim=0)[:max_samples]
    return out


def spatial_id_map(roi_pooled: torch.Tensor, labels: torch.Tensor, method: str = "pca") -> dict[str, Any]:
    """Compute per-cell intrinsic dimension in the 7x7 ROI grid.

    Returns a dict with:
        - id_map: (7, 7) array of ID values
        - id_map_foreground: (7, 7) array using only foreground labels
    """
    N, C, H, W = roi_pooled.shape
    id_map = torch.full((H, W), float("nan"))
    id_map_fg = torch.full((H, W), float("nan"))
    fg_mask = labels >= 1

    for i in range(H):
        for j in range(W):
            cell = roi_pooled[:, :, i, j]  # (N, C)
            if cell.shape[0] >= 2:
                id_map[i, j] = estimate_intrinsic_dimension(cell, method=method).item()
            if fg_mask.any() and cell[fg_mask].shape[0] >= 2:
                id_map_fg[i, j] = estimate_intrinsic_dimension(cell[fg_mask], method=method).item()

    return {
        "id_map": id_map.tolist(),
        "id_map_foreground": id_map_fg.tolist(),
        "mean_id": float(id_map.nanmean().item()) if not id_map.isnan().all() else float("nan"),
        "mean_id_foreground": float(id_map_fg.nanmean().item()) if not id_map_fg.isnan().all() else float("nan"),
    }


def main() -> None:
    global args
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = torch.device(args.device) if args.device else resolve_device(config)

    _, val_loader = build_detection_loaders(
        config,
        limit_val=args.limit_val,
        batch_size=int(config["posttrain"].get("batch_size", 1)),
    )
    num_classes = int(config["model"]["num_classes"])

    model = build_detector(config)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    checkpoint_metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}

    install_active_correction_from_checkpoint_state(
        model,
        state_dict,
        checkpoint_path=args.checkpoint,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
    )
    model.load_state_dict(state_dict)
    model.eval()

    layer_features = extract_features_with_fpn_levels(
        model,
        val_loader,
        device,
        args.max_geometry_samples,
        args.extract_fpn_levels,
        args.spatial_id_map,
        config,
        args.limit_val,
    )

    table = build_table(
        layer_features, num_classes, args.id_method, args.normalize, args.per_class
    )

    spatial_report = {}
    if args.spatial_id_map and "roi_pooled" in layer_features:
        spatial_report = spatial_id_map(
            layer_features["roi_pooled"],
            layer_features["labels"],
            method=args.id_method,
        )

    report = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "id_method": args.id_method,
        "normalize": args.normalize,
        "per_class": args.per_class,
        "layer_geometry": table,
        "spatial_id_map": spatial_report,
    }

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, run_dir / "spatial_layerwise_geometry.json")

    # Print summary.
    print(f"\nSpatial layer-wise geometry for {args.run_name}")
    print(f"{'Layer':<15} {'Ambient':>10} {'N':>6} {'N_fg':>6} {'ID_fg':>8} {'intra':>8} {'inter':>8}")
    print("-" * 72)
    for layer_name, geom in table.items():
        print(
            f"{layer_name:<15} "
            f"{geom['ambient_dim']:>10} "
            f"{geom['n_samples']:>6} "
            f"{geom['n_foreground']:>6} "
            f"{geom['id_foreground']:>8.2f} "
            f"{geom['intra_mean']:>8.4f} "
            f"{geom['inter_mean']:>8.4f}"
        )

    if spatial_report:
        print(f"\nROI pooled per-cell mean ID (all): {spatial_report['mean_id']:.2f}")
        print(f"ROI pooled per-cell mean ID (fg):  {spatial_report['mean_id_foreground']:.2f}")

    print(f"\nSaved report to {run_dir / 'spatial_layerwise_geometry.json'}")


if __name__ == "__main__":
    main()
