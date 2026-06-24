"""FFT/Manifold discriminative power in classifier's uncertain region [0.3, 0.7].

Hypothesis: If geometric signal (FFT/manifold features) provides MORE value
where the classifier is uncertain (conf in [0.3, 0.7]), then dAUC in the
uncertain group should exceed dAUC across all proposals.

Uses precomputed raw features from scripts/manifold_fft_results/raw_features.npz
which contains 3224 proposals with 768-dim FFT features, IoU, confidence, and image IDs.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_NPZ = Path("scripts/manifold_fft_results/raw_features.npz")
OUT_JSON = Path("scripts/manifold_fft_results/uncertain_region_analysis.json")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
npz = np.load(RAW_NPZ)
X_raw: np.ndarray = npz["X_raw"]          # (3224, 768)
ious: np.ndarray = npz["ious"]            # (3224,)
confs: np.ndarray = npz["confs"]          # (3224,)
img_ids: np.ndarray = npz["img_ids"]      # (3224,)

M = X_raw.shape[0]

# Labels: TP if IoU >= 0.5, FP if IoU < 0.3 (strict, ignore ambiguous 0.3-0.5)
is_tp = ious >= 0.5
is_fp = ious < 0.3
strict_mask = is_tp | is_fp
y = is_tp.astype(int)

# Uncertain region: confidence in [0.3, 0.7]
uncertain_mask = (confs >= 0.3) & (confs <= 0.7)
strict_uncertain = strict_mask & uncertain_mask

print(f"Total proposals: {M}")
print(f"TP (IoU>=0.5): {is_tp.sum()}, FP (IoU<0.3): {is_fp.sum()}, Ambiguous: {(~is_tp & ~is_fp).sum()}")
print(f"Uncertain [0.3,0.7]: {uncertain_mask.sum()}")
print(f"  TP in uncertain: {(uncertain_mask & is_tp).sum()}")
print(f"  FP in uncertain: {(uncertain_mask & is_fp).sum()}")
print(f"  Ambiguous in uncertain: {(uncertain_mask & ~is_tp & ~is_fp).sum()}")
print(f"Strict uncertain (TP/FP only): {strict_uncertain.sum()}")

# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
# PCA to 50 dims (same as existing pipeline)
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_raw)
print(f"PCA(50) variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# 1. FFT raw 768 Euclidean distance to TP centroid
tp_centroid_raw = X_raw[is_tp].mean(axis=0)
fft_dist = np.linalg.norm(X_raw - tp_centroid_raw, axis=1)

# 2. Isomap(6) embedding distance
print("Computing Isomap(6)...")
iso = Isomap(n_neighbors=15, n_components=6)
X_iso = iso.fit_transform(X_pca)
tp_centroid_iso = X_iso[is_tp].mean(axis=0)
iso_dist = np.linalg.norm(X_iso - tp_centroid_iso, axis=1)

# 3. LLE(6) embedding distance
print("Computing LLE(6)...")
lle = LocallyLinearEmbedding(n_neighbors=15, n_components=6, method="standard", random_state=42)
X_lle = lle.fit_transform(X_pca)
tp_centroid_lle = X_lle[is_tp].mean(axis=0)
lle_dist = np.linalg.norm(X_lle - tp_centroid_lle, axis=1)

# 4. KDE density score (fit on TP, score all)
print("Computing KDE density...")
kde = KernelDensity(bandwidth="scott", kernel="gaussian")
kde.fit(X_pca[is_tp])
kde_score = kde.score_samples(X_pca)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fit_lr(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, LogisticRegression | None]:
    """Fit logistic regression and return AUC on masked subset."""
    Xm = X[mask]
    ym = y[mask]
    if len(np.unique(ym)) < 2:
        return float("nan"), None
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(Xm, ym)
    prob = lr.predict_proba(Xm)[:, 1]
    auc = roc_auc_score(ym, prob)
    return float(auc), lr


def pair_consistency_score(
    score: np.ndarray,
    ious: np.ndarray,
    mask: np.ndarray,
    uncertain_mask: np.ndarray,
    img_ids: np.ndarray,
    y: np.ndarray,
) -> tuple[float, int, float, int]:
    """Compute pair consistency: score-order agrees with IoU-order for mixed-label pairs.

    Returns (pc_all, n_all, pc_uncertain, n_uncertain).
    """
    agree_all = 0
    n_all = 0
    agree_unc = 0
    n_unc = 0

    unique_imgs = np.unique(img_ids[mask])
    for img in unique_imgs:
        img_mask = (img_ids == img) & mask
        idxs = np.where(img_mask)[0]
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                # Only mixed-label pairs (TP vs FP)
                if y[a] == y[b]:
                    continue
                n_all += 1
                score_order = score[a] > score[b]
                iou_order = ious[a] > ious[b]
                if score_order == iou_order:
                    agree_all += 1

                # Both in uncertain region and strict
                if uncertain_mask[a] and uncertain_mask[b] and mask[a] and mask[b]:
                    n_unc += 1
                    if score_order == iou_order:
                        agree_unc += 1

    pc_all = agree_all / n_all if n_all else 0.0
    pc_unc = agree_unc / n_unc if n_unc else 0.0
    return pc_all, n_all, pc_unc, n_unc


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
features: dict[str, np.ndarray] = {
    "conf_only": confs.reshape(-1, 1),
    "conf_fft": np.column_stack([confs, fft_dist]),
    "conf_isomap": np.column_stack([confs, iso_dist]),
    "conf_lle": np.column_stack([confs, lle_dist]),
    "conf_kde": np.column_stack([confs, kde_score]),
}

# ---------------------------------------------------------------------------
# AUC + Pair Consistency analysis
# ---------------------------------------------------------------------------
results: list[dict[str, Any]] = []
base_all: float | None = None
base_unc: float | None = None

print("\n" + "=" * 90)
print("AUC + PAIR CONSISTENCY ANALYSIS")
print("=" * 90)
print(f"{'Method':<20s} | {'AUC(all)':>9s} | {'AUC(unc)':>9s} | {'dAUC(all)':>10s} | {'dAUC(unc)':>10s} | {'PC(all)':>8s} | {'PC(unc)':>8s}")
print("-" * 90)

for name, Xf in features.items():
    auc_all, _ = fit_lr(Xf, y, strict_mask)
    auc_unc, _ = fit_lr(Xf, y, strict_uncertain)

    if name == "conf_only":
        base_all = auc_all
        base_unc = auc_unc

    # Pair consistency: use the auxiliary feature as score (or confidence for baseline)
    score = Xf[:, 1] if Xf.shape[1] > 1 else confs
    pc_all, n_all, pc_unc, n_unc = pair_consistency_score(
        score, ious, strict_mask, uncertain_mask, img_ids, y
    )

    d_all = auc_all - (base_all or 0.0)
    d_unc = auc_unc - (base_unc or 0.0)

    print(
        f"{name:<20s} | {auc_all:>9.4f} | {auc_unc:>9.4f} | {d_all:>+10.4f} | {d_unc:>+10.4f} | {pc_all:>8.3f} | {pc_unc:>8.3f}"
    )

    results.append({
        "method": name,
        "auc_all": float(auc_all),
        "auc_uncertain": float(auc_unc),
        "delta_auc_all": float(d_all),
        "delta_auc_uncertain": float(d_unc),
        "pair_consistency_all": float(pc_all),
        "pair_consistency_all_n": int(n_all),
        "pair_consistency_uncertain": float(pc_unc),
        "pair_consistency_uncertain_n": int(n_unc),
    })

# ---------------------------------------------------------------------------
# Hypothesis test
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("HYPOTHESIS TEST: dAUC(uncertain) > dAUC(all)?")
print("=" * 90)

verdicts = []
for r in results:
    if r["method"] == "conf_only":
        continue
    supported = r["delta_auc_uncertain"] > r["delta_auc_all"]
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    verdicts.append({
        "method": r["method"],
        "supported": supported,
        "delta_all": r["delta_auc_all"],
        "delta_uncertain": r["delta_auc_uncertain"],
    })
    print(
        f"  {r['method']:<20s} | d_all={r['delta_auc_all']:+.4f} | d_unc={r['delta_auc_uncertain']:+.4f} | {verdict}"
    )

# ---------------------------------------------------------------------------
# Summary statistics for uncertain region
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("UNCERTAIN REGION DESCRIPTIVE STATISTICS")
print("=" * 90)

uncertain_tp = uncertain_mask & is_tp
uncertain_fp = uncertain_mask & is_fp

print(f"Uncertain TP count: {uncertain_tp.sum()}")
print(f"Uncertain FP count: {uncertain_fp.sum()}")
print(f"Uncertain ambiguous count: {(uncertain_mask & ~is_tp & ~is_fp).sum()}")

if uncertain_tp.sum() > 0 and uncertain_fp.sum() > 0:
    # Compare feature distributions in uncertain region
    for feat_name, feat_vec in [
        ("confidence", confs),
        ("fft_dist", fft_dist),
        ("isomap_dist", iso_dist),
        ("lle_dist", lle_dist),
        ("kde_score", kde_score),
    ]:
        tp_vals = feat_vec[uncertain_tp]
        fp_vals = feat_vec[uncertain_fp]
        d = (np.mean(tp_vals) - np.mean(fp_vals)) / np.sqrt(
            (np.var(tp_vals, ddof=1) + np.var(fp_vals, ddof=1)) / 2 + 1e-12
        )
        try:
            u_stat, p_val = stats.mannwhitneyu(tp_vals, fp_vals, alternative="two-sided")
        except ValueError:
            p_val = float("nan")
        print(
            f"  {feat_name:<15s} | TP mean={np.mean(tp_vals):.4f}, FP mean={np.mean(fp_vals):.4f} | Cohen's d={d:+.3f} | p={p_val:.4e}"
        )

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_data = {
    "N_total": int(M),
    "N_strict": int(strict_mask.sum()),
    "N_uncertain": int(uncertain_mask.sum()),
    "N_strict_uncertain": int(strict_uncertain.sum()),
    "results": results,
    "verdicts": verdicts,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved results to: {OUT_JSON}")
print("=" * 90)
