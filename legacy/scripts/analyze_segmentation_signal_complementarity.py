"""Offline complementarity analysis for segmentation signals.

Generates synthetic masks with controlled quality degradation, extracts old and
new segmentation signals, then measures which combinations best predict true
mask IoU.  The script outputs correlation matrices and small logistic-regression
ablations so we can pick structurally complementary signals before committing to
a full training integration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from spectral_detection_posttrain.methods.segmentation.signals import (
    activation_centroid_consistency,
    aspect_ratio_plausibility,
    boundary_phase_coherence,
    boundary_reward,
    connected_component_reward,
    dice_reward,
    interior_exterior_texture_contrast,
    mask_iou_reward,
    multi_scale_saliency_consistency,
    nms_survivor_density,
    score_edge_alignment,
)
from spectral_detection_posttrain.methods.segmentation.signals.fft import (
    compute_amplitude_profile,
    compute_lowfreq_phase_stats,
    compute_structure_similarity,
    edge_similarity_score,
    lowfreq_phase_similarity,
    phase_correlation_score,
    spectral_profile_similarity,
)


OUTPUT_DIR = Path("runs/segmentation_signal_complementarity_analysis")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_random_blob_mask(size: int, rng: np.random.Generator) -> torch.Tensor:
    """Create an irregular blob-like binary mask."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    # Random elliptical + harmonic perturbation
    a = 0.3 + rng.random() * 0.3
    b = 0.3 + rng.random() * 0.3
    theta = rng.random() * math.pi
    cx = (rng.random() - 0.5) * 0.6
    cy = (rng.random() - 0.5) * 0.6
    xr = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
    yr = -(xx - cx) * math.sin(theta) + (yy - cy) * math.cos(theta)
    perturbation = (
        0.05 * torch.sin(6 * math.pi * xx + rng.random() * math.pi)
        + 0.05 * torch.cos(6 * math.pi * yy + rng.random() * math.pi)
    )
    mask = (xr / (a + 1e-6)) ** 2 + (yr / (b + 1e-6)) ** 2 <= 1.0 + perturbation
    return mask


def _degrade_mask(
    target: torch.Tensor,
    degradation: str,
    severity: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Create a predicted mask with a controlled failure mode."""
    pred = target.clone()
    h, w = pred.shape

    if degradation == "erosion":
        # Shrink foreground -> lower recall
        k = max(1, int(round(severity * 5)))
        pred = F.max_pool2d(
            pred.float().view(1, 1, h, w), kernel_size=2 * k + 1, stride=1, padding=k
        )
        pred = (pred.view(h, w) > 0.0).bool()
    elif degradation == "dilation":
        # Expand foreground -> lower precision
        k = max(1, int(round(severity * 5)))
        pred = F.max_pool2d(
            pred.float().view(1, 1, h, w), kernel_size=2 * k + 1, stride=1, padding=k
        )
        pred = (pred.view(h, w) > 0.5).bool()
    elif degradation == "shift":
        # Translate mask -> localization error
        dx = int(round(severity * w * 0.15 * (1 if rng.random() > 0.5 else -1)))
        dy = int(round(severity * h * 0.15 * (1 if rng.random() > 0.5 else -1)))
        pred = torch.roll(pred, shifts=(dy, dx), dims=(0, 1))
    elif degradation == "noise":
        # Random pixel flip -> fragmentation
        flip_p = severity * 0.15
        noise = torch.rand(h, w, generator=torch.Generator().manual_seed(int(rng.integers(1 << 30)))) < flip_p
        pred = pred ^ noise
    elif degradation == "drop_component":
        # Remove a random chunk -> broken topology
        if pred.any():
            coords = torch.nonzero(pred, as_tuple=False)
            idx = int(rng.integers(len(coords)))
            cy, cx = coords[idx].tolist()
            radius = int(round(severity * min(h, w) * 0.25))
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            hole = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            pred = pred & ~hole
    else:
        raise ValueError(f"Unknown degradation: {degradation}")

    return pred


def _make_image_for_mask(mask: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Create a simple RGB image where foreground/background differ in texture."""
    h, w = mask.shape
    # Smooth background + structured foreground
    yy, xx = torch.meshgrid(
        torch.linspace(0, 4 * math.pi, h),
        torch.linspace(0, 4 * math.pi, w),
        indexing="ij",
    )
    base = torch.sin(yy) * 0.3 + torch.cos(xx) * 0.3
    noise = torch.rand(3, h, w) * 0.2
    image = torch.stack([base + noise[c] for c in range(3)], dim=0)
    # Foreground is brighter and has orthogonal texture
    fg_pattern = torch.sin(2 * yy + 1.0) * 0.4 + torch.cos(3 * xx - 0.5) * 0.4
    fg_pattern = fg_pattern.unsqueeze(0).expand(3, h, w)
    image = torch.where(mask.unsqueeze(0), image + 0.4 + fg_pattern * 0.3, image)
    return image.clamp(0.0, 1.0)


@dataclass
class Sample:
    image: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    degradation: str
    severity: float
    iou: float


def generate_dataset(
    n_samples: int = 300,
    size: int = 64,
    seed: int = 42,
) -> list[Sample]:
    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator().manual_seed(seed)
    degradations = ["erosion", "dilation", "shift", "noise", "drop_component"]
    samples: list[Sample] = []
    for _ in range(n_samples):
        target = _make_random_blob_mask(size, rng)
        image = _make_image_for_mask(target, rng)
        degradation = rng.choice(degradations)
        severity = float(rng.random() * 0.8 + 0.1)
        pred = _degrade_mask(target, degradation, severity, rng)
        iou = mask_iou_reward(pred, target).item()
        samples.append(Sample(image, pred, target, degradation, severity, iou))
    return samples


def extract_features(sample: Sample) -> dict[str, float]:
    image = sample.image
    pred = sample.pred
    target = sample.target

    feats: dict[str, float] = {}

    # Geometry / verifiable (need target)
    feats["mask_iou"] = mask_iou_reward(pred, target).item()
    feats["dice"] = dice_reward(pred, target).item()
    feats["boundary_f1"] = boundary_reward(pred, target, tolerance=2).item()
    feats["connected_component"] = connected_component_reward(pred, target).item()

    # FFT (need target for some)
    feats["phase_correlation"] = phase_correlation_score(image, pred, target).item()
    feats["edge_similarity"] = edge_similarity_score(image, pred, target).item()
    feats["structure_similarity"] = compute_structure_similarity(image, pred, target).item()
    feats["lowfreq_phase_similarity"] = lowfreq_phase_similarity(image, pred, target).item()
    feats["spectral_profile_similarity"] = spectral_profile_similarity(image, pred, target, num_bins=16).item()

    # Interpretable / self-contained (no target)
    feats["boundary_phase_coherence"] = boundary_phase_coherence(image, pred).item()
    feats["interior_exterior_texture_contrast"] = interior_exterior_texture_contrast(image, pred).item()
    feats["multi_scale_saliency_consistency"] = multi_scale_saliency_consistency(image, pred).item()
    feats["score_edge_alignment"] = score_edge_alignment(image, pred).item()
    feats["activation_centroid_consistency"] = activation_centroid_consistency(image, pred).item()
    feats["aspect_ratio_plausibility"] = aspect_ratio_plausibility(pred).item()
    # nms_survivor_density needs neighbor masks; pass empty -> 0 by design
    feats["nms_survivor_density"] = nms_survivor_density(pred, []).item()

    return feats


def _feature_matrix(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_dicts = [extract_features(s) for s in samples]
    keys = list(feature_dicts[0].keys())
    X = np.array([[fd[k] for k in keys] for fd in feature_dicts], dtype=np.float64)
    y = np.array([s.iou for s in samples], dtype=np.float64)
    return X, y, keys


def _safe_impute(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with column median; signals like aspect_ratio may be 0."""
    X = np.where(np.isfinite(X), X, np.nan)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def analyze_correlations(X: np.ndarray, keys: list[str]) -> dict:
    corr = np.corrcoef(X.T)
    # Find least correlated pairs (potential complementary signals)
    triu = np.triu(np.abs(corr), k=1)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs.append((keys[i], keys[j], float(triu[i, j])))
    pairs.sort(key=lambda x: x[2])
    return {
        "correlation_matrix": {ki: {kj: float(corr[i, j]) for j, kj in enumerate(keys)} for i, ki in enumerate(keys)},
        "least_correlated_pairs": pairs[:10],
        "most_correlated_pairs": pairs[-10:][::-1],
    }


def _recall_at_precision(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_precision: float = 0.70,
) -> tuple[float, float]:
    """Return recall at target precision and the threshold used."""
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve drops the last threshold; append a sentinel
    thresholds = np.append(thresholds, 1.0)
    valid = precision >= target_precision
    if not valid.any():
        return 0.0, 0.0
    best_idx = int(np.argmax(recall[valid]))
    return float(recall[valid][best_idx]), float(thresholds[valid][best_idx])


def evaluate_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label: str,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    # Regression task: Spearman correlation and MSE
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr

    reg = Ridge(alpha=1.0, random_state=seed)
    reg.fit(X_train, y_train)
    val_pred = reg.predict(X_val)
    spearman = float(spearmanr(y_val, val_pred)[0])
    mse = float(np.mean((y_val - val_pred) ** 2))

    y_bin_train = (y_train >= 0.5).astype(int)
    y_bin_val = (y_val >= 0.5).astype(int)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train, y_bin_train)
    proba = clf.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_bin_val, proba))
    ap = float(average_precision_score(y_bin_val, proba))
    r_at_p70, threshold_p70 = _recall_at_precision(y_bin_val, proba, target_precision=0.70)

    return {
        "label": label,
        "features": feature_names,
        "n_features": len(feature_names),
        "val_mse": mse,
        "val_spearman": spearman,
        "val_auc": auc,
        "val_ap": ap,
        "val_r_at_p70": r_at_p70,
        "val_threshold_p70": threshold_p70,
    }


def run_ablations(X: np.ndarray, y: np.ndarray, keys: list[str]) -> list[dict]:
    results: list[dict] = []

    # Single-signal baselines
    for i, key in enumerate(keys):
        results.append(evaluate_feature_set(X[:, [i]], y, [key], f"single:{key}"))

    # Predefined complementary combinations
    combinations = [
        ("single:boundary_phase_coherence", ["boundary_phase_coherence"]),
        ("single:interior_exterior_texture_contrast", ["interior_exterior_texture_contrast"]),
        ("single:score_edge_alignment", ["score_edge_alignment"]),
        ("single:multi_scale_saliency_consistency", ["multi_scale_saliency_consistency"]),
        ("single:activation_centroid_consistency", ["activation_centroid_consistency"]),
        ("single:aspect_ratio_plausibility", ["aspect_ratio_plausibility"]),
        ("new_top3", ["boundary_phase_coherence", "interior_exterior_texture_contrast", "score_edge_alignment"]),
        ("new_top3 + fft", ["boundary_phase_coherence", "interior_exterior_texture_contrast", "score_edge_alignment", "phase_correlation", "structure_similarity", "spectral_profile_similarity"]),
        ("new_top3 + spatial_self", ["boundary_phase_coherence", "interior_exterior_texture_contrast", "score_edge_alignment", "multi_scale_saliency_consistency", "activation_centroid_consistency", "aspect_ratio_plausibility"]),
        ("fft_only", ["phase_correlation", "edge_similarity", "structure_similarity", "lowfreq_phase_similarity", "spectral_profile_similarity"]),
        ("all_new7", ["boundary_phase_coherence", "interior_exterior_texture_contrast", "multi_scale_saliency_consistency", "score_edge_alignment", "activation_centroid_consistency", "aspect_ratio_plausibility", "nms_survivor_density"]),
        ("all_self_supervised", keys),
    ]

    for label, names in combinations:
        idx = [keys.index(n) for n in names if n in keys]
        if len(idx) == len(names):
            results.append(evaluate_feature_set(X[:, idx], y, names, label))

    return results


def main() -> None:
    _ensure_output_dir()
    print("Generating synthetic segmentation dataset...")
    samples = generate_dataset(n_samples=400, size=64, seed=42)
    print(f"Generated {len(samples)} samples")

    print("Extracting features...")
    X, y, keys = _feature_matrix(samples)
    X = _safe_impute(X)
    print(f"Features: {keys}")

    # Exclude direct-GT signals from features used for prediction.
    gt_signals = {"mask_iou", "dice", "boundary_f1", "connected_component"}
    self_supervised_keys = [k for k in keys if k not in gt_signals]
    X_ss = X[:, [keys.index(k) for k in self_supervised_keys]]

    print("Analyzing correlations among self-supervised signals...")
    corr_analysis = analyze_correlations(X_ss, self_supervised_keys)

    print("Running ablations (predicting GT IoU from self-supervised signals)...")
    ablations = run_ablations(X_ss, y, self_supervised_keys)

    # Add one sanity check: how much do we lose vs using GT signals?
    ablations.append(evaluate_feature_set(X[:, [keys.index(k) for k in ["mask_iou"]]], y, ["mask_iou"], "oracle:mask_iou"))
    ablations.append(evaluate_feature_set(X[:, [keys.index(k) for k in gt_signals]], y, list(gt_signals), "oracle:all_gt_signals"))

    report = {
        "n_samples": len(samples),
        "degradations": sorted({s.degradation for s in samples}),
        "iou_mean": float(np.mean(y)),
        "iou_std": float(np.std(y)),
        "self_supervised_signals": self_supervised_keys,
        "gt_signals": sorted(gt_signals),
        "correlation_analysis": corr_analysis,
        "ablations": ablations,
    }

    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {report_path}")

    # Print summary table
    print("\n=== Ablations (ranked by Val Spearman) ===")
    ablations_sorted = sorted(ablations, key=lambda r: r["val_spearman"], reverse=True)
    print(f"{'Label':<40} {'n':>3} {'MSE':>7} {'Spear':>7} {'AUC':>7} {'AP':>7} {'R@P0.7':>7}")
    print("-" * 85)
    for r in ablations_sorted:
        print(
            f"{r['label']:<40} {r['n_features']:>3} "
            f"{r['val_mse']:>7.4f} {r['val_spearman']:>7.4f} "
            f"{r['val_auc']:>7.4f} {r['val_ap']:>7.4f} {r.get('val_r_at_p70', -1.0):>7.4f}"
        )

    print("\n=== Most complementary pairs (lowest absolute correlation) ===")
    for a, b, c in corr_analysis["least_correlated_pairs"][:8]:
        print(f"  {a}  <->  {b}  |r|={c:.3f}")

    print("\n=== Most redundant pairs (highest absolute correlation) ===")
    for a, b, c in corr_analysis["most_correlated_pairs"][:8]:
        print(f"  {a}  <->  {b}  |r|={c:.3f}")


if __name__ == "__main__":
    main()
