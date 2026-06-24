r"""Standalone analysis of a trained MGL-OPT run.

Loads a detector checkpoint together with saved manifold modules, runs the
validation set, and reports:

- detection metrics (AP50, AP75, ECE, per-class AP)
- box-level IoU diagnostics
- feature-space geometry (intrinsic dimension, compactness, separation)
- before/after active-correction geometry comparison

Example:
    python analyze_manifold_run.py \
        --config spectral_detection_posttrain/configs/manifold_nwpu.yaml \
        --checkpoint runs/nwpu_mglopt_unified_identity_g005_legacy_10ep/checkpoint_best.pth \
        --manifold-modules runs/nwpu_mglopt_unified_identity_g005_legacy_10ep/manifold_modules.pth \
        --run-name analyze_g005
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import (
    evaluate_detection_predictions,
    summarize_iou_diagnostics,
)
from spectral_detection_posttrain.methods.manifold import ManifoldCorrectionPredictor
from spectral_detection_posttrain.methods.manifold.geometry_metrics import (
    compute_manifold_geometry,
    scalar_geometry_report,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Detector checkpoint path.")
    parser.add_argument("--manifold-modules", required=True, help="Path to manifold_modules.pth.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--max-geometry-samples", type=int, default=4096)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    return parser.parse_args()


@torch.no_grad()
def extract_validation_features(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Extract raw and corrected box features from validation RPN proposals.

    Returns ``(raw_features, labels, corrected_features)``.  If the model does
    not use active manifold correction, ``corrected_features`` equals
    ``raw_features``.
    """
    raw_buffer: list[torch.Tensor] = []
    label_buffer: list[torch.Tensor] = []
    corr_buffer: list[torch.Tensor] = []

    active_correction = isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)

    for images, targets in val_loader:
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

        box_features = model.roi_heads.box_roi_pool(features, sampled_props, transformed.image_sizes)
        box_features = model.roi_heads.box_head(box_features)
        labels = torch.cat(labels, dim=0)

        fg_mask = labels >= 1
        if fg_mask.any():
            raw_buffer.append(box_features[fg_mask].cpu())
            label_buffer.append(labels[fg_mask].cpu())
            if active_correction:
                predictor = model.roi_heads.box_predictor
                class_weights = F.one_hot(labels[fg_mask], num_classes=predictor.prototype_bank.num_classes).to(
                    device=device, dtype=box_features.dtype
                )
                transport = predictor.correction_field_from_class_weights(
                    box_features[fg_mask], class_weights
                )
                corr_buffer.append((box_features[fg_mask] + predictor.gamma * transport).cpu())
            else:
                corr_buffer.append(box_features[fg_mask].cpu())

        if sum(t.shape[0] for t in raw_buffer) >= max_samples:
            break

    if not raw_buffer:
        empty = torch.empty((0, 1))
        return empty, empty, empty

    raw = torch.cat(raw_buffer, dim=0)[:max_samples]
    labels = torch.cat(label_buffer, dim=0)[:max_samples]
    corr = torch.cat(corr_buffer, dim=0)[:max_samples]
    return raw, labels, corr


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)

    _, val_loader = build_detection_loaders(
        config,
        limit_train=0,
        limit_val=args.limit_val,
        batch_size=int(config["posttrain"].get("batch_size", 1)),
    )

    num_classes = int(config["model"]["num_classes"])

    # Build detector and load checkpoint.
    model = build_detector(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    # Load manifold modules if the checkpoint is not already wrapped.
    manifold_payload = torch.load(args.manifold_modules, map_location=device)
    if not isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor):
        # If the detector checkpoint does not contain the wrapped predictor,
        # re-install it from the saved module state.
        from spectral_detection_posttrain.methods.manifold import PrototypeBank, TransportHead

        manifold_config = manifold_payload["config"]
        prototype_bank = PrototypeBank(
            num_classes=manifold_config["num_classes"],
            num_prototypes_per_class=manifold_config["num_prototypes"],
            feature_dim=manifold_config["feature_dim"],
        ).to(device)
        prototype_bank.load_state_dict(manifold_payload["prototype_bank"])

        active_head_state = manifold_payload.get("active_transport_head")
        if active_head_state is not None:
            active_head = TransportHead(
                feature_dim=manifold_config["feature_dim"],
                num_prototypes=manifold_config["num_classes"] * manifold_config["num_prototypes"],
                tau=manifold_config["tau"],
            ).to(device)
            active_head.load_state_dict(active_head_state)
            model.roi_heads.box_predictor = ManifoldCorrectionPredictor(
                model.roi_heads.box_predictor,
                prototype_bank=prototype_bank,
                transport_head=active_head,
                gamma=manifold_config["active_correction_gamma"],
                tau=manifold_config["tau"],
                normalize_features=True,
            )

    model.eval()

    # Detection metrics.
    predictions = []
    targets = []
    matched_ious = []
    matched_scores = []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in tgt.items()}
                        for tgt in batch_targets])

    metrics = evaluate_detection_predictions(
        predictions, targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=args.score_threshold,
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
        per_class=True,
        num_classes=num_classes,
    )

    # Geometry metrics.
    raw_features, labels, corr_features = extract_validation_features(
        model, val_loader, device, args.max_geometry_samples
    )
    geometry = compute_manifold_geometry(
        raw_features,
        labels,
        num_classes,
        corrected_features=corr_features,
        method="pca",
        normalize=True,
    )

    report = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "manifold_modules": args.manifold_modules,
        "metrics": metrics,
        "geometry": scalar_geometry_report(geometry),
    }

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, run_dir / "analysis_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
