"""Find hidden good boxes: low-confidence proposals that are actually TP,
using precomputed Isomap embeddings.

Hidden good box definition:
- confidence in [0.1, 0.5] (classifier is uncertain / afraid to label)
- Isomap(6) distance to TP cluster center < median (manifold says it's good)
- actual IoU > 0.5 (it really IS a good box!)

Control group: "manifold says good but actually bad"
- confidence in [0.1, 0.5]
- Isomap distance < median
- IoU < 0.3

Output: statistics + scatter plot saved to
scripts/manifold_fft_results/hidden_good_boxes.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_NPZ = Path("scripts/manifold_fft_results/raw_features.npz")
EMB_NPZ = Path("scripts/manifold_nonlinear_results/embeddings.npz")
OUT_DIR = Path("scripts/manifold_fft_results")
OUT_PNG = OUT_DIR / "hidden_good_boxes.png"
OUT_JSON = OUT_DIR / "hidden_good_boxes.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
npz_raw = np.load(RAW_NPZ)
X_raw: np.ndarray = npz_raw["X_raw"]          # (3224, 768)
ious: np.ndarray = npz_raw["ious"]            # (3224,)
confs: np.ndarray = npz_raw["confs"]          # (3224,)
img_ids: np.ndarray = npz_raw["img_ids"]      # (3224,)

npz_emb = np.load(EMB_NPZ)
# Use precomputed Isomap(6) embeddings if available
if "Isomap" in npz_emb:
    X_iso = npz_emb["Isomap"].astype(np.float32)
else:
    # Recompute Isomap(6) from PCA(50) as fallback
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_raw)
    iso = Isomap(n_neighbors=15, n_components=6)
    X_iso = iso.fit_transform(X_pca)

M = X_raw.shape[0]

# ---------------------------------------------------------------------------
# Compute Isomap distance to TP cluster center
# ---------------------------------------------------------------------------
is_tp = ious >= 0.5
tp_centroid_iso = X_iso[is_tp].mean(axis=0)
iso_dist = np.linalg.norm(X_iso - tp_centroid_iso, axis=1)
iso_dist_median = np.median(iso_dist)

# ---------------------------------------------------------------------------
# Define regions
# ---------------------------------------------------------------------------
# Uncertain region: confidence in [0.1, 0.5]
uncertain_mask = (confs >= 0.1) & (confs <= 0.5)

# Manifold says good: Isomap distance < median
manifold_good_mask = iso_dist < iso_dist_median

# Actually good: IoU > 0.5
actually_good_mask = ious > 0.5

# Actually bad: IoU < 0.3 (strict)
actually_bad_mask = ious < 0.3

# Hidden good boxes: all three conditions
hidden_good_mask = uncertain_mask & manifold_good_mask & actually_good_mask

# Control group: manifold says good but actually bad
control_bad_mask = uncertain_mask & manifold_good_mask & actually_bad_mask

# For comparison: all uncertain proposals
all_uncertain = uncertain_mask.sum()
all_uncertain_tp = (uncertain_mask & actually_good_mask).sum()
all_uncertain_fp = (uncertain_mask & actually_bad_mask).sum()

N_hidden_good = int(hidden_good_mask.sum())
N_control_bad = int(control_bad_mask.sum())

# Percentage of hidden good among all uncertain proposals
pct_hidden_good = 100.0 * N_hidden_good / all_uncertain if all_uncertain else 0.0
# Percentage of hidden good among uncertain TP
pct_hidden_good_of_unc_tp = 100.0 * N_hidden_good / all_uncertain_tp if all_uncertain_tp else 0.0

# ---------------------------------------------------------------------------
# Per-image distribution
# ---------------------------------------------------------------------------
unique_imgs = np.unique(img_ids)
per_img_stats = []
for img in unique_imgs:
    img_mask = img_ids == img
    n_total = int(img_mask.sum())
    n_uncertain = int((img_mask & uncertain_mask).sum())
    n_hidden = int((img_mask & hidden_good_mask).sum())
    n_control = int((img_mask & control_bad_mask).sum())
    per_img_stats.append({
        "img_id": int(img),
        "n_total": n_total,
        "n_uncertain": n_uncertain,
        "n_hidden_good": n_hidden,
        "n_control_bad": n_control,
    })

# Sort by hidden good count descending
per_img_stats.sort(key=lambda x: x["n_hidden_good"], reverse=True)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("=" * 70)
print("HIDDEN GOOD BOXES ANALYSIS")
print("=" * 70)
print(f"Total proposals: {M}")
print(f"Uncertain region [0.1, 0.5]: {all_uncertain}")
print(f"  -> TP in uncertain: {all_uncertain_tp}")
print(f"  -> FP in uncertain: {all_uncertain_fp}")
print(f"  -> Ambiguous (0.3-0.5 IoU): {all_uncertain - all_uncertain_tp - all_uncertain_fp}")
print()
print(f"Hidden good boxes (low conf + manifold good + IoU>0.5): {N_hidden_good}")
print(f"  -> % of uncertain region: {pct_hidden_good:.2f}%")
print(f"  -> % of uncertain TP:     {pct_hidden_good_of_unc_tp:.2f}%")
print()
print(f"Control group (low conf + manifold good + IoU<0.3): {N_control_bad}")
print()
print("Top 10 images by hidden good count:")
for i, s in enumerate(per_img_stats[:10]):
    print(f"  img {s['img_id']:3d}: {s['n_hidden_good']} hidden, {s['n_control_bad']} control, {s['n_uncertain']} uncertain / {s['n_total']} total")

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

# Background: all proposals (light gray)
ax.scatter(confs, iso_dist, c="#cccccc", s=8, alpha=0.4, label="All proposals", zorder=1)

# Uncertain TP (blue, medium)
uncertain_tp_mask = uncertain_mask & actually_good_mask
ax.scatter(confs[uncertain_tp_mask], iso_dist[uncertain_tp_mask], c="#3498db", s=20, alpha=0.6, label="Uncertain TP (IoU>0.5)", zorder=2)

# Uncertain FP (red, medium)
uncertain_fp_mask = uncertain_mask & actually_bad_mask
ax.scatter(confs[uncertain_fp_mask], iso_dist[uncertain_fp_mask], c="#e74c3c", s=20, alpha=0.6, label="Uncertain FP (IoU<0.3)", zorder=2)

# Hidden good boxes (green, large, with edge)
ax.scatter(confs[hidden_good_mask], iso_dist[hidden_good_mask], c="#2ecc71", s=80, edgecolors="black", linewidths=1.2, label=f"Hidden good (N={N_hidden_good})", zorder=5)

# Control bad (orange, large, with edge)
ax.scatter(confs[control_bad_mask], iso_dist[control_bad_mask], c="#f39c12", s=80, edgecolors="black", linewidths=1.2, label=f"Control bad (N={N_control_bad})", zorder=5)

# Reference lines
ax.axhline(iso_dist_median, color="black", linestyle="--", linewidth=1, alpha=0.7, label=f"Isomap median distance")
ax.axvline(0.1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

ax.set_xlabel("Confidence (classifier score)", fontsize=12)
ax.set_ylabel("Isomap(6) distance to TP centroid", fontsize=12)
ax.set_title("Hidden Good Boxes: Low-confidence proposals that are actually TP\n(Isomap manifold distance < median)", fontsize=13)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0, np.percentile(iso_dist, 99.5))

# Annotation box
ann_text = (
    f"Hidden good: {N_hidden_good} ({pct_hidden_good:.1f}% of uncertain)\n"
    f"Control bad: {N_control_bad}\n"
    f"Isomap median: {iso_dist_median:.4f}"
)
ax.text(0.02, 0.98, ann_text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved scatter plot to: {OUT_PNG}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
out_data = {
    "N_total": int(M),
    "N_uncertain": int(all_uncertain),
    "N_uncertain_tp": int(all_uncertain_tp),
    "N_uncertain_fp": int(all_uncertain_fp),
    "N_hidden_good": N_hidden_good,
    "N_control_bad": N_control_bad,
    "pct_hidden_good_of_uncertain": round(pct_hidden_good, 4),
    "pct_hidden_good_of_uncertain_tp": round(pct_hidden_good_of_unc_tp, 4),
    "isomap_median_distance": float(round(iso_dist_median, 6)),
    "per_image_stats": per_img_stats,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print(f"Saved JSON to: {OUT_JSON}")
print("=" * 70)

# Verdict
if N_hidden_good > 10 and pct_hidden_good > 5.0:
    print(f"VERDICT: SUPPORTED — Isomap manifold geometry CAN discover classifier-misjudged good boxes.")
else:
    print(f"VERDICT: NOT SUPPORTED — Hidden good boxes are too rare to be actionable.")
print("=" * 70)
