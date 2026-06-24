r"""Layer-wise manifold geometry diagnostics for detector checkpoints.

Extracts intermediate representations along the detector's ROI stream:

    roi_pooled  ->  box_head fc1  ->  box_head fc2/z  ->  corrected z

and reports intrinsic dimension, intra-class compactness, and inter-class
separation at each layer, both globally and **per class**. The goal is to
identify where the manifold structure degrades as the data flows toward the
classifier.

Example:
    python analyze_layerwise_geometry.py \
        --config spectral_detection_posttrain/configs/manifold_nwpu.yaml \
        --checkpoint runs/round2100_nwpu_baseline/checkpoint_best.pth \
        --run-name baseline_layerwise

For an active-correction run, pass --manifold-modules so the corrected z branch
is also populated:

    python analyze_layerwise_geometry.py \
        --config spectral_detection_posttrain/configs/manifold_nwpu.yaml \
        --checkpoint runs/nwpu_active_sweep_g005_en0001/checkpoint_best.pth \
        --manifold-modules runs/nwpu_active_sweep_g005_en0001/manifold_modules.pth \
        --run-name g005_layerwise
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from spectral_detection_posttrain.core.models.bottleneck_box_head import BottleneckTwoMLPHead
from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.methods.manifold import ManifoldCorrectionPredictor, PrototypeBank, TransportHead
from spectral_detection_posttrain.methods.manifold.geometry_metrics import (
    compute_class_centroids,
    compute_effective_rank,
    compute_intra_class_compactness,
    compute_inter_class_separation,
    compute_manifold_geometry,
    compute_nc1,
    compute_separability_auc,
    compute_spectral_decay,
    estimate_intrinsic_dimension,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


_ACTIVE_PREDICTOR_PREFIX = "roi_heads.box_predictor."
_ACTIVE_PROTOTYPE_KEY = _ACTIVE_PREDICTOR_PREFIX + "prototype_bank.prototypes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise detector manifold geometry.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Detector checkpoint path.")
    parser.add_argument("--manifold-modules", default=None, help="Optional manifold_modules.pth for active correction.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--max-geometry-samples", type=int, default=4096)
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu).")
    parser.add_argument("--id-method", default="pca", choices=["pca", "twonn"])
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="L2-normalize features for compactness/separation.")
    parser.add_argument("--per-class", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute and output per-class geometry metrics.")
    parser.add_argument("--active-correction-gamma", type=float, default=None,
                        help="Override gamma when loading active-correction checkpoints.")
    parser.add_argument("--active-correction-mode", default=None,
                        choices=["residual", "endpoint", "gated_endpoint", "gated-endpoint"],
                        help="Override correction mode when loading active-correction checkpoints.")
    parser.add_argument("--transport-tau", type=float, default=None,
                        help="Override transport softmax temperature for active-correction checkpoints.")
    return parser.parse_args()


def _resolve_box_head_layers(box_head: torch.nn.Module) -> tuple[torch.nn.Module | None, torch.nn.Module | None, torch.nn.Module | None]:
    """Return (fc1, fc2, bottleneck) modules from a box head.

    Handles direct ``TwoMLPHead``, wrappers like ``AFMThenHead``, and
    ``BottleneckTwoMLPHead``.
    """
    fc1 = getattr(box_head, "fc6", None)
    fc2 = getattr(box_head, "fc7", None)
    bottleneck = getattr(box_head, "bottleneck", None)
    if fc1 is None or fc2 is None:
        inner = getattr(box_head, "head", None)
        if inner is not None:
            fc1 = getattr(inner, "fc6", fc1)
            fc2 = getattr(inner, "fc7", fc2)
            bottleneck = getattr(inner, "bottleneck", bottleneck)
    return fc1, fc2, bottleneck


def _load_sibling_manifold_config(checkpoint_path: str | Path) -> dict[str, Any]:
    config_path = Path(checkpoint_path).parent / "manifold_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _active_checkpoint_config(
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: str | Path,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
    gamma_override: float | None = None,
    mode_override: str | None = None,
    tau_override: float | None = None,
) -> dict[str, Any] | None:
    prototypes = state_dict.get(_ACTIVE_PROTOTYPE_KEY)
    if prototypes is None:
        return None
    if prototypes.ndim != 3:
        raise ValueError(f"{_ACTIVE_PROTOTYPE_KEY} must have shape (C, K, D)")

    num_classes, num_prototypes, feature_dim = (int(v) for v in prototypes.shape)
    hidden_weight = state_dict.get(_ACTIVE_PREDICTOR_PREFIX + "transport_head.mlp.0.weight")
    hidden_dim = int(hidden_weight.shape[0]) if hidden_weight is not None else feature_dim

    sibling_config = _load_sibling_manifold_config(checkpoint_path)
    has_endpoint_gate = _ACTIVE_PREDICTOR_PREFIX + "endpoint_gate.weight" in state_dict
    default_mode = "gated_endpoint" if has_endpoint_gate else "residual"
    correction_mode = (
        mode_override
        or sibling_config.get("active_correction_mode")
        or sibling_config.get("correction_mode")
        or default_mode
    )
    correction_mode = str(correction_mode).replace("-", "_")
    if has_endpoint_gate and correction_mode != "gated_endpoint":
        correction_mode = "gated_endpoint"

    gamma = gamma_override
    if gamma is None:
        checkpoint_metadata = checkpoint_metadata or {}
        gamma = checkpoint_metadata.get("active_correction_gamma_epoch")
    if gamma is None:
        gamma = sibling_config.get("active_correction_gamma", sibling_config.get("gamma", 1.0))
    tau = tau_override
    if tau is None:
        tau = sibling_config.get("tau", sibling_config.get("transport_tau", 0.1))

    return {
        "num_classes": num_classes,
        "num_prototypes": num_prototypes,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "gamma": float(gamma),
        "tau": float(tau),
        "correction_mode": correction_mode,
    }


def install_bottleneck_head_from_checkpoint_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> bool:
    """Replace the standard box head with BottleneckTwoMLPHead if checkpoint uses it."""
    if "roi_heads.box_head.bottleneck.weight" not in state_dict:
        return False
    in_channels = int(state_dict["roi_heads.box_head.bottleneck.weight"].shape[1])
    bottleneck_channels = int(state_dict["roi_heads.box_head.bottleneck.weight"].shape[0])
    fc6_weight = state_dict.get("roi_heads.box_head.fc6.weight")
    representation_size = int(fc6_weight.shape[0]) if fc6_weight is not None else 1024
    grid_size = 7
    model.roi_heads.box_head = BottleneckTwoMLPHead(
        in_channels=in_channels,
        bottleneck_channels=bottleneck_channels,
        representation_size=representation_size,
        grid_size=grid_size,
    ).to(device)
    return True


def install_active_correction_from_checkpoint_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    checkpoint_path: str | Path,
    device: torch.device,
    checkpoint_metadata: dict[str, Any] | None = None,
    gamma_override: float | None = None,
    mode_override: str | None = None,
    tau_override: float | None = None,
) -> bool:
    """Install the active correction wrapper required by a saved checkpoint."""
    active_config = _active_checkpoint_config(
        state_dict,
        checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        gamma_override=gamma_override,
        mode_override=mode_override,
        tau_override=tau_override,
    )
    if active_config is None:
        return False

    prototype_bank = PrototypeBank(
        num_classes=active_config["num_classes"],
        num_prototypes_per_class=active_config["num_prototypes"],
        feature_dim=active_config["feature_dim"],
    ).to(device)
    transport_head = TransportHead(
        feature_dim=active_config["feature_dim"],
        num_prototypes=active_config["num_classes"] * active_config["num_prototypes"],
        hidden_dim=active_config["hidden_dim"],
        tau=active_config["tau"],
    ).to(device)
    model.roi_heads.box_predictor = ManifoldCorrectionPredictor(
        model.roi_heads.box_predictor,
        prototype_bank=prototype_bank,
        transport_head=transport_head,
        gamma=active_config["gamma"],
        tau=active_config["tau"],
        normalize_features=True,
        correction_mode=active_config["correction_mode"],
    ).to(device)
    return True


@torch.no_grad()
def extract_layerwise_features(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    max_samples: int,
) -> dict[str, torch.Tensor]:
    """Collect per-layer ROI features from validation RPN proposals.

    Returns a dict with keys:
        - roi_pooled: (N, 256, 7, 7)
        - fc1:        (N, fc1_dim)
        - z:          (N, z_dim)
        - z_corrected: (N, z_dim) or None
        - labels:     (N,)
    """
    roi_buffer: list[torch.Tensor] = []
    bottleneck_buffer: list[torch.Tensor] = []
    fc1_buffer: list[torch.Tensor] = []
    z_buffer: list[torch.Tensor] = []
    corr_buffer: list[torch.Tensor] = []
    label_buffer: list[torch.Tensor] = []

    active_correction = isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)
    fc1_mod, _, bottleneck_mod = _resolve_box_head_layers(model.roi_heads.box_head)
    has_fc1 = fc1_mod is not None
    has_bottleneck = bottleneck_mod is not None

    bottleneck_out: torch.Tensor | None = None
    fc1_out: torch.Tensor | None = None

    def _bottleneck_hook(_module: torch.nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal bottleneck_out
        bottleneck_out = output.detach()

    def _fc1_hook(_module: torch.nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal fc1_out
        fc1_out = output.detach()

    if has_bottleneck:
        bottleneck_mod.register_forward_hook(_bottleneck_hook)
    if has_fc1:
        fc1_mod.register_forward_hook(_fc1_hook)

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

        roi_pooled = model.roi_heads.box_roi_pool(features, sampled_props, transformed.image_sizes)
        bottleneck_out = None
        fc1_out = None

        z = model.roi_heads.box_head(roi_pooled)
        labels_t = torch.cat(labels, dim=0)

        fg_mask = labels_t >= 1
        if fg_mask.any():
            roi_buffer.append(roi_pooled[fg_mask].cpu())
            if has_bottleneck and bottleneck_out is not None:
                bottleneck_buffer.append(bottleneck_out[fg_mask].cpu())
            if fc1_out is not None:
                fc1_buffer.append(fc1_out[fg_mask].cpu())
            z_buffer.append(z[fg_mask].cpu())
            label_buffer.append(labels_t[fg_mask].cpu())

            if active_correction:
                predictor = model.roi_heads.box_predictor
                class_weights = F.one_hot(labels_t[fg_mask], num_classes=predictor.prototype_bank.num_classes).to(
                    device=device, dtype=z.dtype
                )
                transport = predictor.correction_field_from_class_weights(z[fg_mask], class_weights)
                corr_buffer.append((z[fg_mask] + predictor.gamma * transport).cpu())

        if sum(t.shape[0] for t in z_buffer) >= max_samples:
            break

    if not z_buffer:
        empty = torch.empty((0, 1))
        return {
            "roi_pooled": empty,
            "bottleneck": empty,
            "fc1": empty,
            "z": empty,
            "z_corrected": None,
            "labels": empty,
        }

    def _cat_trim(buf: list[torch.Tensor], max_samples: int) -> torch.Tensor:
        if not buf:
            return torch.empty((0, 1))
        out = torch.cat(buf, dim=0)[:max_samples]
        return out

    out: dict[str, torch.Tensor | None] = {
        "roi_pooled": _cat_trim(roi_buffer, max_samples),
        "fc1": _cat_trim(fc1_buffer, max_samples),
        "z": _cat_trim(z_buffer, max_samples),
        "z_corrected": _cat_trim(corr_buffer, max_samples) if active_correction else None,
        "labels": _cat_trim(label_buffer, max_samples),
    }
    if has_bottleneck:
        out["bottleneck"] = _cat_trim(bottleneck_buffer, max_samples)
    return out


def _per_class_inter_distance(
    centroids: torch.Tensor,
    labels: torch.Tensor,
    normalize: bool,
) -> dict[str, float]:
    """For each present foreground class, return mean distance to other class centroids."""
    present = sorted({int(c.item()) for c in labels.unique() if c.item() >= 1})
    result: dict[str, float] = {}
    if len(present) < 2:
        return {str(c): float("nan") for c in present}

    cvecs = centroids[present]
    if normalize:
        cvecs = F.normalize(cvecs, dim=-1)
    dists = torch.cdist(cvecs, cvecs, p=2)
    for i, c in enumerate(present):
        mask = torch.ones(len(present), dtype=torch.bool)
        mask[i] = False
        result[str(c)] = float(dists[i][mask].mean().item())
    return result


def geometry_for_layer(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    method: str,
    normalize: bool,
    per_class: bool,
) -> dict[str, Any]:
    """Compute geometry metrics for a single layer's feature matrix."""
    if features.ndim == 4:
        # ROI pooled: (N, C, H, W) -> (N, C*H*W)
        features = features.flatten(start_dim=1)

    fg_mask = labels >= 1
    result: dict[str, Any] = {
        "ambient_dim": int(features.shape[-1]),
        "n_samples": int(features.shape[0]),
        "n_foreground": int(fg_mask.sum().item()),
    }

    # Overall/foreground ID.
    if features.shape[0] >= 2:
        result["id_overall"] = float(estimate_intrinsic_dimension(features, method=method).item())
    else:
        result["id_overall"] = float("nan")

    if fg_mask.any() and features.shape[0] >= 2:
        result["id_foreground"] = float(estimate_intrinsic_dimension(features[fg_mask], method=method).item())
    else:
        result["id_foreground"] = float("nan")

    # Compactness / separation.
    centroids, counts = compute_class_centroids(features, labels, num_classes)
    result["per_class_count"] = {str(c): int(counts[c].item()) for c in range(num_classes)}

    intra = compute_intra_class_compactness(features, labels, num_classes, centroids, normalize=normalize)
    inter = compute_inter_class_separation(centroids, labels=labels, normalize=normalize)

    result["intra_mean"] = float(intra["intra_mean"].item())
    result["intra_median"] = float(intra["intra_median"].item())
    result["intra_max"] = float(intra["intra_max"].item())
    result["inter_mean"] = float(inter["inter_mean"].item())
    result["inter_min"] = float(inter["inter_min"].item())

    # Effective rank and spectral decay.
    if features.shape[0] >= 2:
        result["effective_rank_overall"] = float(compute_effective_rank(features).item())
        spectral = compute_spectral_decay(features)
        for key, value in spectral.items():
            result[key] = float(value.item())
    else:
        result["effective_rank_overall"] = float("nan")

    if fg_mask.any() and features[fg_mask].shape[0] >= 2:
        result["effective_rank_foreground"] = float(compute_effective_rank(features[fg_mask]).item())
        spectral_fg = compute_spectral_decay(features[fg_mask])
        for key, value in spectral_fg.items():
            result[f"{key}_foreground"] = float(value.item())
    else:
        result["effective_rank_foreground"] = float("nan")

    # NC1 and separability (foreground-based).
    result["nc1_overall"] = float(compute_nc1(features, labels, num_classes).item())
    sep = compute_separability_auc(features, labels, num_classes, centroids=centroids)
    result["separability_overall"] = float(sep["separability_overall"].item())

    if per_class:
        result["per_class_id"] = {
            str(c): float(estimate_intrinsic_dimension(features[labels == c], method=method).item())
            if (labels == c).sum() >= 2 else float("nan")
            for c in range(1, num_classes)
        }
        result["per_class_intra_mean"] = {
            str(c): float(intra["per_class_intra_mean"][c].item())
            if counts[c] > 0 else float("nan")
            for c in range(1, num_classes)
        }
        result["per_class_inter_mean"] = _per_class_inter_distance(centroids, labels, normalize)
        if isinstance(sep["per_class_separability"], dict):
            result["per_class_separability"] = {
                str(c): float(sep["per_class_separability"].get(str(c), float("nan")))
                for c in range(1, num_classes)
            }

    return result


def build_table(
    layer_features: dict[str, torch.Tensor],
    num_classes: int,
    method: str,
    normalize: bool,
    per_class: bool,
) -> dict[str, Any]:
    """Build the layer-wise geometry table."""
    labels = layer_features["labels"]
    table: dict[str, Any] = {}
    for layer_name, features in layer_features.items():
        if layer_name == "labels" or features is None:
            continue
        table[layer_name] = geometry_for_layer(
            features, labels, num_classes, method, normalize, per_class
        )
    return table


def main() -> None:
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

    install_bottleneck_head_from_checkpoint_state(model, state_dict, device=device)

    installed_from_checkpoint = install_active_correction_from_checkpoint_state(
        model,
        state_dict,
        checkpoint_path=args.checkpoint,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
        gamma_override=args.active_correction_gamma,
        mode_override=args.active_correction_mode,
        tau_override=args.transport_tau,
    )

    # Legacy runs may keep active modules outside the detector checkpoint.
    if args.manifold_modules is not None and not installed_from_checkpoint:
        manifold_payload = torch.load(args.manifold_modules, map_location=device, weights_only=False)
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
                correction_mode=manifold_config.get("active_correction_mode", "residual"),
            )

    model.load_state_dict(state_dict)

    model.eval()

    layer_features = extract_layerwise_features(model, val_loader, device, args.max_geometry_samples)
    table = build_table(
        layer_features, num_classes, args.id_method, args.normalize, args.per_class
    )

    report = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "manifold_modules": args.manifold_modules,
        "id_method": args.id_method,
        "normalize": args.normalize,
        "per_class": args.per_class,
        "layer_geometry": table,
    }

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, run_dir / "layerwise_geometry.json")

    # Print compact summary tables.
    print(f"\nLayer-wise geometry for {args.run_name}")
    print(f"{'Layer':<15} {'Ambient':>8} {'N':>6} {'N_fg':>6} {'ID_fg':>8} {'intra':>8} {'inter':>8}")
    print("-" * 70)
    for layer_name, geom in table.items():
        print(
            f"{layer_name:<15} "
            f"{geom['ambient_dim']:>8} "
            f"{geom['n_samples']:>6} "
            f"{geom['n_foreground']:>6} "
            f"{geom['id_foreground']:>8.2f} "
            f"{geom['intra_mean']:>8.4f} "
            f"{geom['inter_mean']:>8.4f}"
        )

    if args.per_class:
        print("\nPer-class foreground ID by layer")
        class_ids = sorted(int(c) for c in table["z"]["per_class_id"].keys())
        header = " | ".join([f"class {c:>2}" for c in class_ids])
        print(f"{'Layer':<15} | {header}")
        print("-" * (20 + len(class_ids) * 10))
        for layer_name, geom in table.items():
            vals = " | ".join(
                f"{geom['per_class_id'].get(str(c), float('nan')):>8.2f}" for c in class_ids
            )
            print(f"{layer_name:<15} | {vals}")

        print("\nPer-class intra-mean by layer")
        print(f"{'Layer':<15} | {header}")
        print("-" * (20 + len(class_ids) * 10))
        for layer_name, geom in table.items():
            vals = " | ".join(
                f"{geom['per_class_intra_mean'].get(str(c), float('nan')):>8.4f}" for c in class_ids
            )
            print(f"{layer_name:<15} | {vals}")

    # Endpoint correction delta: z_corrected vs z (the key metric under the
    # expansion-then-contraction logic).
    if "z_corrected" in table and table["z_corrected"]["n_samples"] > 0:
        z = table["z"]
        zc = table["z_corrected"]
        delta = {
            "id_fg": zc["id_foreground"] - z["id_foreground"],
            "intra_mean": zc["intra_mean"] - z["intra_mean"],
            "inter_mean": zc["inter_mean"] - z["inter_mean"],
        }
        print("\nEndpoint correction delta (z_corrected - z)")
        print(f"  ID_fg : {delta['id_fg']:+.2f}   (negative = dimension dropped)")
        print(f"  intra : {delta['intra_mean']:+.4f}   (negative = more compact)")
        print(f"  inter : {delta['inter_mean']:+.4f}   (positive = better separated)")
        report["endpoint_delta"] = delta
        save_json(report, run_dir / "layerwise_geometry.json")

    print(f"\nSaved full report to {run_dir / 'layerwise_geometry.json'}")


if __name__ == "__main__":
    main()
