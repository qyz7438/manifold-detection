r"""Multi-layer cross-validated intrinsic dimension estimation for detector checkpoints.

For each checkpoint, extracts intermediate ROI-head representations:

    roi_pooled -> fc1/fc6 -> bottleneck (when present) -> z -> z_corrected (when present)

and estimates intrinsic dimension with multiple methods, multiple hyper-parameters,
and repeated random subsampling folds.  Results are saved as JSON and summarized as
plain-text tables plus correlation matrices against AP50/AP75.

Example
-------
    python analyze_id_cross_validation.py \
        --config spectral_detection_posttrain/configs/manifold_nwpu.yaml \
        --baseline runs/nwpu_baseline_best.pth \
        --run-dirs runs/nwpu_head_original_e5 \
            runs/nwpu_head_bottleneck_r128_e5 \
            runs/nwpu_head_bottlenecktwomlp_c64_e5 \
            runs/nwpu_head_convlowdim_c128_e5 \
            runs/nwpu_head_attentionpool_e5 \
        --output id_cross_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from spectral_detection_posttrain.core.models.box_heads import replace_box_head
from spectral_detection_posttrain.core.models.bottleneck_box_head import BottleneckTwoMLPHead
from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.experiments.schema import validate_experiment_config
from spectral_detection_posttrain.methods.manifold import (
    IntrinsicDimEstimator,
    ManifoldCorrectionPredictor,
    PrototypeBank,
    TransportHead,
)
from spectral_detection_posttrain.methods.manifold.geometry_metrics import compute_class_centroids
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


_ACTIVE_PREDICTOR_PREFIX = "roi_heads.box_predictor."
_ACTIVE_PROTOTYPE_KEY = _ACTIVE_PREDICTOR_PREFIX + "prototype_bank.prototypes"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_run_info(run_dir: Path) -> dict[str, Any]:
    """Load manifold_result.json and metadata.json, merging into one dict."""
    info: dict[str, Any] = {}
    for name in ("manifold_result.json", "metadata.json"):
        path = run_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                info.update(json.load(f))
    return info


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
) -> dict[str, Any] | None:
    """Infer active-correction config from a detector state dict (or None)."""
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
        sibling_config.get("active_correction_mode")
        or sibling_config.get("correction_mode")
        or default_mode
    )
    correction_mode = str(correction_mode).replace("-", "_")
    if has_endpoint_gate and correction_mode != "gated_endpoint":
        correction_mode = "gated_endpoint"

    gamma = checkpoint_metadata.get("active_correction_gamma_epoch") if checkpoint_metadata else None
    if gamma is None:
        gamma = sibling_config.get("active_correction_gamma", sibling_config.get("gamma", 1.0))
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


def _install_bottleneck_head_from_checkpoint_state(
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
    from spectral_detection_posttrain.core.models.bottleneck_box_head import BottleneckTwoMLPHead

    model.roi_heads.box_head = BottleneckTwoMLPHead(
        in_channels=in_channels,
        bottleneck_channels=bottleneck_channels,
        representation_size=representation_size,
        grid_size=7,
    ).to(device)
    return True


def _install_active_correction_from_checkpoint_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: str | Path,
    device: torch.device,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> bool:
    """Install the active correction wrapper required by a saved checkpoint."""
    active_config = _active_checkpoint_config(state_dict, checkpoint_path, checkpoint_metadata=checkpoint_metadata)
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


def load_model_for_run(
    run_dir: Path,
    baseline_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Build a detector, replace the head, and install active correction if needed."""
    config = load_config(config_path)
    config = validate_experiment_config(config)

    info = _load_run_info(run_dir)
    box_head_type = info.get("box_head_type", "original")

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
            rank=int(info.get("box_head_rank", 128)),
            conv_channels=int(info.get("box_head_conv_channels", 128)),
            bottleneck_dim=int(info.get("box_head_bottleneck_dim", 512)),
            attention_channels=int(info.get("box_head_attention_channels", 64)),
            copy_compatible_weights=True,
        )

    return model, config, box_head_type


# ---------------------------------------------------------------------------
# Layer resolution / feature extraction
# ---------------------------------------------------------------------------


def _resolve_layer_modules(box_head: torch.nn.Module) -> dict[str, torch.nn.Module | None]:
    """Resolve the modules whose outputs we want to capture.

    Returns a dict with keys ``fc1``, ``bottleneck`` and ``pre_proj``.
    ``pre_proj`` is the low-dimensional spatially-pooled vector for heads that
    do not have a standard fc6 (e.g. ConvLowDimBoxHead, AttentionPoolBoxHead).
    """
    fc1 = getattr(box_head, "fc6", None)
    fc2 = getattr(box_head, "fc7", None)
    bottleneck = getattr(box_head, "bottleneck", None)

    # Unwrap simple wrappers.
    if fc1 is None or fc2 is None:
        inner = getattr(box_head, "head", None)
        if inner is not None:
            fc1 = getattr(inner, "fc6", fc1)
            fc2 = getattr(inner, "fc7", fc2)
            bottleneck = getattr(inner, "bottleneck", bottleneck)

    # For BottleneckBoxHead the bottleneck is bottleneck_down.
    if bottleneck is None and hasattr(box_head, "bottleneck_down"):
        bottleneck = box_head.bottleneck_down
    else:
        inner = getattr(box_head, "head", None)
        if inner is not None and hasattr(inner, "bottleneck_down"):
            bottleneck = inner.bottleneck_down

    # ``pre_proj`` captures the vector just before the final 1024 projection
    # for heads that do not expose a useful ``bottleneck`` module.
    pre_proj: torch.nn.Module | None = None
    if isinstance(box_head, type(None)):
        pass
    elif hasattr(box_head, "global_pool"):
        pre_proj = box_head.global_pool
    elif hasattr(box_head, "attention"):
        # AttentionPoolBoxHead: we will capture pooled features manually.
        pre_proj = "attention_pool_pooled"

    return {"fc1": fc1, "bottleneck": bottleneck, "pre_proj": pre_proj}


@torch.no_grad()
def extract_layerwise_features(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    max_samples: int = 8192,
) -> dict[str, torch.Tensor | None]:
    """Collect per-layer ROI features from validation RPN proposals.

    Collects *all* valid proposals (labels >= 0), not only foreground, so that
    both ``overall`` and ``foreground`` IDs can be computed afterwards.
    """
    model.eval()

    layer_mods = _resolve_layer_modules(model.roi_heads.box_head)
    active_correction = isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)

    captured: dict[str, torch.Tensor | None] = {"fc1": None, "bottleneck": None}

    def _make_hook(name: str):
        def hook(_module, _input, output: torch.Tensor) -> None:
            captured[name] = output.detach()
        return hook

    handles = []
    if layer_mods["fc1"] is not None:
        handles.append(layer_mods["fc1"].register_forward_hook(_make_hook("fc1")))
    if layer_mods["bottleneck"] is not None and isinstance(layer_mods["bottleneck"], torch.nn.Module):
        handles.append(layer_mods["bottleneck"].register_forward_hook(_make_hook("bottleneck")))

    buffers: dict[str, list[torch.Tensor]] = {
        "roi_pooled": [],
        "fc1": [],
        "bottleneck": [],
        "pre_proj": [],
        "z": [],
        "z_corrected": [],
        "labels": [],
    }

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        transformed, _ = model.transform(images, None)
        features = model.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, _ = model.rpn(transformed, features, targets)
        sampled_props, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(
            proposals, targets
        )

        roi_pooled = model.roi_heads.box_roi_pool(features, sampled_props, transformed.image_sizes)
        captured["fc1"] = None
        captured["bottleneck"] = None

        z = model.roi_heads.box_head(roi_pooled)
        labels_t = torch.cat(labels, dim=0)
        valid_mask = labels_t >= 0

        if valid_mask.any():
            buffers["roi_pooled"].append(roi_pooled[valid_mask].cpu())
            buffers["z"].append(z[valid_mask].cpu())
            buffers["labels"].append(labels_t[valid_mask].cpu())

            if captured["fc1"] is not None:
                fc1_out = captured["fc1"][valid_mask]
                if isinstance(layer_mods["fc1"], torch.nn.Linear):
                    fc1_out = F.relu(fc1_out)
                buffers["fc1"].append(fc1_out.cpu())

            if captured["bottleneck"] is not None:
                bneck_out = captured["bottleneck"][valid_mask]
                if isinstance(layer_mods["bottleneck"], (torch.nn.Linear, torch.nn.Conv2d)):
                    bneck_out = F.relu(bneck_out)
                if bneck_out.ndim == 4:
                    bneck_out = bneck_out.flatten(start_dim=1)
                buffers["bottleneck"].append(bneck_out.cpu())

            # ConvLowDimBoxHead / AttentionPoolBoxHead intermediate.
            box_head = model.roi_heads.box_head
            if hasattr(box_head, "global_pool"):
                with torch.no_grad():
                    x = box_head.conv_reduce(roi_pooled[valid_mask])
                    x = box_head.global_pool(x)
                    buffers["pre_proj"].append(x.flatten(start_dim=1).cpu())
            elif hasattr(box_head, "attention"):
                with torch.no_grad():
                    x = roi_pooled[valid_mask]
                    n, c, h, w = x.shape
                    attn = box_head.attention(x).view(n, -1)
                    attn = F.softmax(attn, dim=-1)
                    pooled = (x.view(n, c, h * w) * attn.unsqueeze(1)).sum(dim=-1)
                    buffers["pre_proj"].append(pooled.cpu())

            if active_correction:
                predictor = model.roi_heads.box_predictor
                z_valid = z[valid_mask].to(device)
                cls_onehot = F.one_hot(
                    labels_t[valid_mask].clamp_min(0).to(torch.int64),
                    num_classes=predictor.prototype_bank.num_classes,
                ).to(device=device, dtype=z_valid.dtype)
                transport = predictor.correction_field_from_class_weights(z_valid, cls_onehot)
                z_corrected = z_valid + predictor.gamma * transport
                buffers["z_corrected"].append(z_corrected.cpu())

        if sum(t.shape[0] for t in buffers["labels"]) >= max_samples:
            break

    for h in handles:
        h.remove()

    def _cat_trim(buf: list[torch.Tensor], max_samples: int) -> torch.Tensor | None:
        if not buf:
            return None
        out = torch.cat(buf, dim=0)[:max_samples]
        return out

    out: dict[str, torch.Tensor | None] = {
        "roi_pooled": _cat_trim(buffers["roi_pooled"], max_samples),
        "fc1": _cat_trim(buffers["fc1"], max_samples),
        "bottleneck": _cat_trim(buffers["bottleneck"], max_samples),
        "pre_proj": _cat_trim(buffers["pre_proj"], max_samples),
        "z": _cat_trim(buffers["z"], max_samples),
        "z_corrected": _cat_trim(buffers["z_corrected"], max_samples),
        "labels": _cat_trim(buffers["labels"], max_samples),
    }
    return out


# ---------------------------------------------------------------------------
# Intrinsic dimension estimators
# ---------------------------------------------------------------------------


def _subsample(features: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Deterministic random subsample of at most ``n`` rows."""
    n_total = features.shape[0]
    if n_total <= n:
        return features
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    return features[perm[:n]]


def _pca_id(features: torch.Tensor, threshold: float) -> float:
    """PCA cumulative-variance intrinsic dimension."""
    estimator = IntrinsicDimEstimator(method="pca", pca_variance_threshold=threshold)
    return float(estimator.estimate_id(features).item())


def _pairwise_dist(features: torch.Tensor) -> torch.Tensor:
    """Memory-efficient pairwise distance matrix with self-distance set to inf."""
    dist = torch.cdist(features, features)
    dist.fill_diagonal_(float("inf"))
    return dist


def _twonn_id(dist: torch.Tensor, k: int = 2) -> float:
    """TwoNN estimator from a pre-computed distance matrix."""
    n = dist.shape[0]
    if n <= k:
        return float("nan")
    sorted_dist, _ = torch.topk(dist, k=min(k, n - 1), largest=False, dim=-1)
    r1 = sorted_dist[:, 0].clamp_min(1e-12)
    r2 = sorted_dist[:, 1].clamp_min(1e-12)
    log_mu = torch.log((r2 / r1).clamp_min(1e-12))
    id_est = 1.0 / log_mu.mean()
    return float(id_est.clamp_min(1.0).item())


def _lid_id(dist: torch.Tensor, k: int) -> tuple[float, float]:
    """LID-style local estimator. Returns (median over points, mean over points)."""
    n = dist.shape[0]
    if n <= k:
        return float("nan"), float("nan")
    sorted_dist, _ = torch.topk(dist, k=k, largest=False, dim=-1)
    r1 = sorted_dist[:, 0].clamp_min(1e-12)
    ratios = sorted_dist / r1.unsqueeze(-1)
    log_ratios = torch.log(ratios.clamp_min(1e-12))
    mean_log = log_ratios.mean(dim=-1)
    # Avoid division by zero / negative values caused by numerical issues.
    mean_log = mean_log.clamp_min(1e-12)
    local_id = math.log(k) / mean_log
    local_id = local_id.clamp_min(1.0)
    return float(local_id.median().item()), float(local_id.mean().item())


def compute_cv_ids(
    features: torch.Tensor,
    n_folds: int,
    subsample_size: int,
    base_seed: int,
    pca_thresholds: list[float],
    twonn_k_values: list[int],
    lid_k_values: list[int],
) -> dict[str, Any]:
    """Cross-validated ID estimates for a single feature matrix.

    Returns nested dictionaries with ``mean``, ``std`` and per-fold ``values``.
    """
    n = features.shape[0]
    if n < 2:
        return {
            "pca": {t: {"mean": float("nan"), "std": float("nan"), "folds": []} for t in pca_thresholds},
            "twonn": {k: {"mean": float("nan"), "std": float("nan"), "folds": []} for k in twonn_k_values},
            "lid": {
                k: {"median": {"mean": float("nan"), "std": float("nan")},
                    "mean": {"mean": float("nan"), "std": float("nan")},
                    "folds": []}
                for k in lid_k_values
            },
        }

    size = min(subsample_size, n)

    pca_folds = {t: [] for t in pca_thresholds}
    twonn_folds = {k: [] for k in twonn_k_values}
    lid_folds = {k: [] for k in lid_k_values}

    for fold in range(n_folds):
        seed = base_seed + fold * 10007
        sub = _subsample(features, size, seed)

        for t in pca_thresholds:
            pca_folds[t].append(_pca_id(sub, t))

        dist = _pairwise_dist(sub)
        for k in twonn_k_values:
            twonn_folds[k].append(_twonn_id(dist, k))
        for k in lid_k_values:
            med, mean = _lid_id(dist, k)
            lid_folds[k].append((med, mean))

    def _summarize(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=float)
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "folds": values}

    result = {
        "pca": {t: _summarize(pca_folds[t]) for t in pca_thresholds},
        "twonn": {k: _summarize(twonn_folds[k]) for k in twonn_k_values},
        "lid": {
            k: {
                "median": _summarize([v[0] for v in lid_folds[k]]),
                "mean": _summarize([v[1] for v in lid_folds[k]]),
                "folds": lid_folds[k],
            }
            for k in lid_k_values
        },
    }
    return result


# ---------------------------------------------------------------------------
# Post-processing: tables and correlations
# ---------------------------------------------------------------------------


def _flatten_layer_result(
    layer_result: dict[str, Any],
    scope: str,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten one scope (overall/foreground) of a layer result to scalar columns."""
    out: dict[str, float] = {}
    ids = layer_result.get(scope, {})
    for t, v in ids.get("pca", {}).items():
        out[f"{prefix}pca_{t}"] = v["mean"]
        out[f"{prefix}pca_{t}_std"] = v["std"]
    for k, v in ids.get("twonn", {}).items():
        out[f"{prefix}twonn_k{k}"] = v["mean"]
        out[f"{prefix}twonn_k{k}_std"] = v["std"]
    for k, v in ids.get("lid", {}).items():
        out[f"{prefix}lid_median_k{k}"] = v["median"]["mean"]
        out[f"{prefix}lid_median_k{k}_std"] = v["median"]["std"]
        out[f"{prefix}lid_mean_k{k}"] = v["mean"]["mean"]
        out[f"{prefix}lid_mean_k{k}_std"] = v["mean"]["std"]
    return out


def build_summary_table(results: dict[str, Any], scope: str = "foreground") -> pd.DataFrame:
    """Build a wide DataFrame: one row per (run, layer)."""
    rows = []
    for run_name, run_data in results.items():
        ap50 = run_data.get("ap50", float("nan"))
        ap75 = run_data.get("ap75", float("nan"))
        for layer_name, layer_result in run_data.get("layers", {}).items():
            ambient = layer_result.get("ambient_dim", float("nan"))
            n = layer_result.get("n_samples", float("nan"))
            n_fg = layer_result.get("n_foreground", float("nan"))
            row = {
                "head": run_name,
                "layer": layer_name,
                "AP50": ap50,
                "AP75": ap75,
                "ambient_dim": ambient,
                "n": n,
                "n_fg": n_fg,
            }
            row.update(_flatten_layer_result(layer_result, scope, prefix=""))
            rows.append(row)
    return pd.DataFrame(rows)


def _correlation_table(df: pd.DataFrame, metric_cols: list[str], target_cols: list[str] = None) -> pd.DataFrame:
    """Correlation of numeric ID columns with AP50/AP75 (Pearson and Spearman)."""
    target_cols = target_cols or ["AP50", "AP75"]
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for metric in metric_cols:
            if metric not in df.columns:
                continue
            valid = df[[metric] + target_cols].dropna()
            if len(valid) < 3:
                continue
            for target in target_cols:
                r_p, p_p = pearsonr(valid[metric], valid[target])
                r_s, p_s = spearmanr(valid[metric], valid[target])
                records.append({
                    "metric": metric,
                    "target": target,
                    "pearson_r": r_p,
                    "pearson_p": p_p,
                    "spearman_r": r_s,
                    "spearman_p": p_s,
                    "n": len(valid),
                })
    return pd.DataFrame(records)


def _stability_table(df: pd.DataFrame, scope: str = "foreground") -> pd.DataFrame:
    """For each estimator configuration, report mean fold-std across runs."""
    mean_cols = [c for c in df.columns if any(x in c for x in ["pca_", "twonn_", "lid_"]) and not c.endswith("_std")]
    records = []
    for mean_col in mean_cols:
        std_col = mean_col + "_std"
        if std_col not in df.columns:
            continue
        std_vals = df[std_col].dropna()
        mean_vals = df[mean_col].dropna()
        if len(std_vals) == 0:
            continue
        records.append({
            "metric": mean_col,
            "mean_estimate": mean_vals.mean(),
            "mean_fold_std": std_vals.mean(),
            "cv_across_runs": mean_vals.std() / abs(mean_vals.mean()) if mean_vals.mean() != 0 else float("nan"),
        })
    return pd.DataFrame(records).sort_values("mean_fold_std", na_position="last")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validated multi-layer intrinsic dimension estimation.")
    parser.add_argument("--config", required=True, help="Detector config YAML.")
    parser.add_argument("--baseline", required=True, help="Baseline detector checkpoint.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to analyze.")
    parser.add_argument("--checkpoints", nargs="+", default=["checkpoint_last.pth"], help="Checkpoint names.")
    parser.add_argument("--output", default="id_cross_validation.json", help="Output JSON path.")
    parser.add_argument("--max-samples", type=int, default=8192, help="Max proposals to collect per run.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--subsample-size", type=int, default=4096, help="Samples per fold.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base RNG seed for folds.")
    parser.add_argument("--pca-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95, 0.99])
    parser.add_argument("--twonn-k", nargs="+", type=int, default=[2, 3, 5, 10])
    parser.add_argument("--lid-k", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization before ID estimation.")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = validate_experiment_config(config)
    set_seed(int(config.get("seed", args.base_seed)))
    device = torch.device(args.device) if args.device else resolve_device(config)

    _, val_loader = build_detection_loaders(
        config,
        batch_size=int(config["posttrain"].get("batch_size", 1)),
    )

    normalize = not args.no_normalize
    baseline_path = Path(args.baseline)

    all_results: dict[str, Any] = {}

    for run_dir in map(Path, args.run_dirs):
        run_name = run_dir.name
        info = _load_run_info(run_dir)
        ap50 = info.get("best_val_ap50", float("nan"))
        ap75 = info.get("best_val_ap75", float("nan"))
        box_head_type = info.get("box_head_type", "original")

        all_results[run_name] = {
            "ap50": ap50,
            "ap75": ap75,
            "box_head_type": box_head_type,
            "checkpoints": {},
        }

        model, _, _ = load_model_for_run(run_dir, baseline_path, args.config, device)

        for ckpt_name in args.checkpoints:
            ckpt_path = run_dir / ckpt_name
            if not ckpt_path.exists():
                print(f"Skipping missing checkpoint: {ckpt_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Run: {run_name} | checkpoint: {ckpt_name}")
            print(f"AP50={ap50:.4f} AP75={ap75:.4f} head={box_head_type}")
            print(f"{'='*60}")

            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            checkpoint_metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}

            _install_bottleneck_head_from_checkpoint_state(model, state_dict, device)
            _install_active_correction_from_checkpoint_state(
                model, state_dict, ckpt_path, device, checkpoint_metadata
            )
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            print(f"Extracting layer-wise features (max {args.max_samples}) ...")
            layer_features = extract_layerwise_features(model, val_loader, device, args.max_samples)

            labels = layer_features["labels"]
            fg_mask = labels >= 1 if labels is not None else None

            layer_results: dict[str, Any] = {}
            for layer_name, features in layer_features.items():
                if layer_name == "labels" or features is None:
                    continue

                if features.ndim == 4:
                    features = features.flatten(start_dim=1)

                if normalize:
                    features = F.normalize(features, dim=-1)

                print(f"  Layer {layer_name}: {features.shape}")

                overall_ids = compute_cv_ids(
                    features,
                    n_folds=args.n_folds,
                    subsample_size=args.subsample_size,
                    base_seed=args.base_seed,
                    pca_thresholds=args.pca_thresholds,
                    twonn_k_values=args.twonn_k,
                    lid_k_values=args.lid_k,
                )
                foreground_ids = None
                if fg_mask is not None and fg_mask.any():
                    foreground_ids = compute_cv_ids(
                        features[fg_mask],
                        n_folds=args.n_folds,
                        subsample_size=args.subsample_size,
                        base_seed=args.base_seed + 1,
                        pca_thresholds=args.pca_thresholds,
                        twonn_k_values=args.twonn_k,
                        lid_k_values=args.lid_k,
                    )

                layer_results[layer_name] = {
                    "ambient_dim": int(features.shape[-1]),
                    "n_samples": int(features.shape[0]),
                    "n_foreground": int(fg_mask.sum().item()) if fg_mask is not None else 0,
                    "overall": overall_ids,
                    "foreground": foreground_ids,
                }

            all_results[run_name]["checkpoints"][ckpt_name] = {
                "path": str(ckpt_path),
                "layers": layer_results,
            }

            # Per-checkpoint mini table.
            print(f"\nLayer-wise ID summary (foreground, mean ± std over {args.n_folds} folds)")
            header = (
                f"{'Layer':<15} {'Ambient':>8} {'N':>6} {'N_fg':>6} "
                f"{'PCA0.9':>10} {'PCA0.95':>10} {'TwoNN_k2':>10} {'LIDmed20':>10}"
            )
            print(header)
            print("-" * len(header))
            for layer_name, res in layer_results.items():
                fg = res.get("foreground") or res.get("overall")
                pca_09 = fg["pca"][0.90] if 0.90 in fg["pca"] else {"mean": float("nan"), "std": float("nan")}
                pca_095 = fg["pca"][0.95] if 0.95 in fg["pca"] else {"mean": float("nan"), "std": float("nan")}
                twonn_2 = fg["twonn"][2] if 2 in fg["twonn"] else {"mean": float("nan"), "std": float("nan")}
                lid_20 = fg["lid"][20]["median"] if 20 in fg["lid"] else {"mean": float("nan"), "std": float("nan")}
                print(
                    f"{layer_name:<15} {res['ambient_dim']:>8} {res['n_samples']:>6} "
                    f"{res['n_foreground']:>6} "
                    f"{pca_09['mean']:>5.1f}±{pca_09['std']:<3.1f} "
                    f"{pca_095['mean']:>5.1f}±{pca_095['std']:<3.1f} "
                    f"{twonn_2['mean']:>5.2f}±{twonn_2['std']:<3.2f} "
                    f"{lid_20['mean']:>5.2f}±{lid_20['std']:<3.2f}"
                )

    # Save raw results.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results to {args.output}")

    # Build per-checkpoint summary tables and correlations.
    for ckpt_name in args.checkpoints:
        print(f"\n{'#'*70}")
        print(f"# Summary for {ckpt_name}")
        print(f"{'#'*70}")

        # Re-shape results to one row per (run, layer) for this checkpoint.
        ckpt_results = {}
        for run_name, run_data in all_results.items():
            if ckpt_name not in run_data.get("checkpoints", {}):
                continue
            ckpt_results[run_name] = {
                "ap50": run_data["ap50"],
                "ap75": run_data["ap75"],
                "box_head_type": run_data["box_head_type"],
                "layers": run_data["checkpoints"][ckpt_name]["layers"],
            }

        if not ckpt_results:
            print(f"No data for {ckpt_name}")
            continue

        for scope in ("overall", "foreground"):
            df = build_summary_table(ckpt_results, scope=scope)
            if df.empty:
                continue

            print(f"\n--- Scope: {scope} ---")
            # Compact table with selected columns.
            selected = [
                "head", "layer", "AP50", "AP75",
                "pca_0.9", "pca_0.9_std",
                "pca_0.95", "pca_0.95_std",
                "twonn_k2", "twonn_k2_std",
                "lid_median_k20", "lid_median_k20_std",
            ]
            available = [c for c in selected if c in df.columns]
            print(df[available].to_string(index=False))

            # Correlations with AP50/AP75.
            id_cols = [c for c in df.columns if c not in ("head", "layer", "AP50", "AP75", "ambient_dim", "n", "n_fg") and not c.endswith("_std")]
            corr_df = _correlation_table(df, id_cols, ["AP50", "AP75"])
            if not corr_df.empty:
                print(f"\nTop correlations with AP50/AP75 (Pearson |r|) for {scope}:")
                corr_df["abs_r"] = corr_df["pearson_r"].abs()
                top = corr_df.sort_values("abs_r", ascending=False).head(15)
                print(top[["metric", "target", "pearson_r", "pearson_p", "spearman_r", "n"]].to_string(index=False))

            # Stability: lowest mean fold-std across runs (true cross-fold stability).
            stab_df = _stability_table(df, scope=scope)
            if not stab_df.empty:
                print(f"\nMost stable estimators (lowest mean fold-std) for {scope}:")
                print(stab_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
