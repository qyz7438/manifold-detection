"""Control-IoU geometric signal validation for FFT spectral features.

Question: After controlling for IoU, can FFT geometric features (local intrinsic
dimension + PCA distance to TP centroid) independently discriminate TP vs FP?

Steps:
1. Load baseline checkpoint, extract all val proposals
2. Compute per-proposal: 7168-dim raw FFT amplitude, IoU, TP/FP label
3. PCA -> 50 dims
4. Per-IoU-bin analysis: measure TLE (k=30) and PCA distance to TP centroid
5. Report Cohen's d for TP-vs-FP separation within each bin
6. Verdict: Cohen's d > 0.3 => viable independent signal for RLVR reward
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

warnings.filterwarnings("ignore")
set_seed(42)
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
model = build_detector(
    {
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
        }
    }
).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

# Hooks to capture ROI crops and proposals
box_head_in, roi_crops, sampled_props = {}, {}, {}
model.roi_heads.box_head.register_forward_pre_hook(
    lambda m, args: box_head_in.update({"x": args[0]})
)
model.roi_heads.box_roi_pool.register_forward_pre_hook(
    lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
)
model.roi_heads.box_roi_pool.register_forward_hook(
    lambda m, i, o: roi_crops.update({"c": o.clone()})
)

# ---------------------------------------------------------------------------
# 2. Extract proposals + raw FFT + IoU + confidence
# ---------------------------------------------------------------------------
_, vl = build_penn_fudan_loaders_320(batch_size=1)

all_iou = []
all_conf = []
all_is_tp = []
all_fft_amp = []  # list of (N, 7168) numpy arrays per image

for img, tgt in tqdm(vl, desc="Extracting"):
    img_d = [img[0].to(DEV)]
    tgt_d = [{k: v.to(DEV) for k, v in tgt[0].items()}]
    box_head_in.clear()
    roi_crops.clear()
    sampled_props.clear()

    with torch.no_grad():
        model(img_d, tgt_d)

    rf = box_head_in.get("x")
    crops = roi_crops.get("c")
    sp_raw = sampled_props.get("p")
    if rf is None or crops is None or sp_raw is None or rf.shape[0] == 0:
        continue

    sp_cat = torch.cat(sp_raw, dim=0)
    bf = model.roi_heads.box_head(rf)
    reg_out = model.roi_heads.box_predictor.bbox_pred(bf)
    person_deltas = reg_out[:, 2:6]
    decoded = decode_boxes(sp_cat, person_deltas)
    gt_boxes = tgt_d[0]["boxes"]
    N = sp_cat.shape[0]

    # Raw FFT amplitude per bin: (N, C, H, W) where H=W=7
    # rfft2 on (7,7) -> (7, 4) complex values (7//2+1 = 4)
    # Amplitude: (N, C, 7, 4) -> flatten -> (N, C*7*4) = (N, 7168)
    f = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")  # (N, C, 7, 4)
    amp_raw = torch.abs(f)  # (N, C, 7, 4)
    amp_flat = amp_raw.view(N, -1).cpu().numpy()  # (N, 7168)

    if len(gt_boxes) == 0:
        for i in range(N):
            all_iou.append(0.0)
            all_conf.append(0.0)
            all_is_tp.append(False)
        all_fft_amp.append(amp_flat)
        continue

    ious = box_iou(decoded, gt_boxes)  # (N, G)
    best_iou, best_gt = ious.max(dim=1)  # (N,)
    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]

    for i in range(N):
        iou_i = best_iou[i].item()
        all_iou.append(iou_i)
        all_conf.append(conf[i].item())
        # TP: IoU >= 0.5, FP: IoU < 0.3 (strict, ignore 0.3-0.5 ambiguous)
        all_is_tp.append(iou_i >= 0.5)
    all_fft_amp.append(amp_flat)

# Concatenate all proposals
fft_mat = np.concatenate(all_fft_amp, axis=0)  # (M, 7168)
iou_vec = np.array(all_iou)
conf_vec = np.array(all_conf)
is_tp_vec = np.array(all_is_tp)
M = fft_mat.shape[0]

# Strict FP mask: IoU < 0.3
is_fp_vec = iou_vec < 0.3

print(f"\nTotal proposals: {M}")
print(f"TP (IoU>=0.5): {is_tp_vec.sum()}")
print(f"FP (IoU<0.3): {is_fp_vec.sum()}")
print(f"Ambiguous (0.3<=IoU<0.5): {(~is_tp_vec & ~is_fp_vec).sum()}")

# ---------------------------------------------------------------------------
# 3. PCA to 50 dims
# ---------------------------------------------------------------------------
pca = PCA(n_components=50, random_state=42)
pca_fft = pca.fit_transform(fft_mat)  # (M, 50)
var_ratio = pca.explained_variance_ratio_.sum()
print(f"PCA(50) variance explained: {var_ratio:.4f}")

# ---------------------------------------------------------------------------
# 4. Compute per-proposal metrics
#    a) Local intrinsic dimension (TLE, k=30)
#    b) PCA distance to TP centroid
# ---------------------------------------------------------------------------

# 4a. Local intrinsic dimension via TLE (TwoNN estimator from skdim)
# TLE uses TwoNN: ratio of distances to k-th and 2k-th neighbor
k_tle = 30
local_id = np.full(M, np.nan)

try:
    import skdim

    tle = skdim.id.TLE()
    # fit_transform_pw returns array of local IDs
    local_id = tle.fit_transform_pw(pca_fft, n_neighbors=k_tle)
    print(f"Local ID computed (TLE, k={k_tle})")
except Exception as e:
    print(f"skdim TLE failed: {e}")
    # Fallback: manual TwoNN estimator on PCA space
    print("Using manual TwoNN fallback...")
    nn = NearestNeighbors(n_neighbors=k_tle * 2 + 1, metric="euclidean")
    nn.fit(pca_fft)
    dists, _ = nn.kneighbors(pca_fft)  # (M, 2k+1)
    # TwoNN: r = dist_k / dist_2k, ID = -log(k/2k) / log(r) = log(2) / log(r)
    # Use k=10 for stability (2k=20)
    k_nn = 10
    r = dists[:, k_nn] / (dists[:, 2 * k_nn] + 1e-12)
    r = np.clip(r, 1e-6, 1 - 1e-6)
    local_id = np.log(2.0) / np.log(r)
    # Clip extreme outliers
    local_id = np.clip(local_id, 0.5, 50.0)
    print(f"Manual TwoNN fallback completed")

# 4b. PCA distance to TP centroid
tp_centroid = pca_fft[is_tp_vec].mean(axis=0)  # (50,)
tp_pca_dist = np.linalg.norm(pca_fft - tp_centroid, axis=1)  # (M,)

# ---------------------------------------------------------------------------
# 5. Per-IoU-bin analysis: Cohen's d for TP vs FP separation
# ---------------------------------------------------------------------------

def cohens_d(x1, x2):
    """Compute Cohen's d (pooled std)."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std


# Define IoU bins
bins = [
    (0.00, 0.30, "FP_only_0.0-0.3"),  # Pure FP region, no TP
    (0.30, 0.50, "ambiguous_0.3-0.5"),  # Excluded from strict TP/FP
    (0.50, 0.70, "low_TP_0.5-0.7"),
    (0.70, 0.90, "high_TP_0.7-0.9"),
    (0.90, 1.01, "perfect_TP_0.9-1.0"),
]

print(f"\n{'='*70}")
print("PER-IoU-BIN ANALYSIS: TP vs FP geometric discrimination")
print(f"{'='*70}")
print("(Note: TP/FP defined by IoU threshold, so bins are mutually exclusive.")
print(" This section shows descriptive stats per bin; key test is Section 6.)")

results = []
for lo, hi, name in bins:
    mask = (iou_vec >= lo) & (iou_vec < hi)
    n_total = mask.sum()
    n_tp = (mask & is_tp_vec).sum()
    n_fp = (mask & is_fp_vec).sum()

    print(f"\n--- Bin {name} ---")
    print(f"  Total: {n_total}, TP: {n_tp}, FP: {n_fp}")

    if n_tp < 5 or n_fp < 5:
        print(f"  SKIP: insufficient samples (need >=5 each)")
        continue

    # Metric 1: Local intrinsic dimension
    id_tp = local_id[mask & is_tp_vec]
    id_fp = local_id[mask & is_fp_vec]
    d_id = cohens_d(id_tp, id_fp)
    d_id_abs = abs(d_id)

    # Metric 2: PCA distance to TP centroid
    dist_tp = tp_pca_dist[mask & is_tp_vec]
    dist_fp = tp_pca_dist[mask & is_fp_vec]
    d_dist = cohens_d(dist_tp, dist_fp)
    d_dist_abs = abs(d_dist)

    # Mann-Whitney U test (non-parametric)
    try:
        u_id, p_id = stats.mannwhitneyu(id_tp, id_fp, alternative="two-sided")
    except ValueError:
        p_id = np.nan
    try:
        u_dist, p_dist = stats.mannwhitneyu(dist_tp, dist_fp, alternative="two-sided")
    except ValueError:
        p_dist = np.nan

    print(f"  Local ID      — TP mean={np.mean(id_tp):.3f}, FP mean={np.mean(id_fp):.3f}, Cohen's d={d_id:+.3f} (|d|={d_id_abs:.3f}), p={p_id:.4e}")
    print(f"  PCA dist      — TP mean={np.mean(dist_tp):.3f}, FP mean={np.mean(dist_fp):.3f}, Cohen's d={d_dist:+.3f} (|d|={d_dist_abs:.3f}), p={p_dist:.4e}")

    results.append({
        "bin": name,
        "n_tp": int(n_tp),
        "n_fp": int(n_fp),
        "d_local_id": float(d_id),
        "d_local_id_abs": float(d_id_abs),
        "d_pca_dist": float(d_dist),
        "d_pca_dist_abs": float(d_dist_abs),
        "p_local_id": float(p_id),
        "p_pca_dist": float(p_dist),
    })

# ---------------------------------------------------------------------------
# 6. KEY TEST: Control IoU — partial correlation & residual analysis
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("KEY TEST A: Partial correlation — geometry vs TP/FP | IoU, conf")
print(f"{'='*70}")

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

strict_mask = is_tp_vec | is_fp_vec
y = is_tp_vec[strict_mask].astype(int)  # 1=TP, 0=FP
X_iou_conf = np.column_stack([iou_vec[strict_mask], conf_vec[strict_mask]])

# 6a. Partial correlation via residual regression
# Regress local_id on [iou, conf], then correlate residual with TP/FP
reg_id = LinearRegression().fit(X_iou_conf, local_id[strict_mask])
id_resid = local_id[strict_mask] - reg_id.predict(X_iou_conf)

reg_dist = LinearRegression().fit(X_iou_conf, tp_pca_dist[strict_mask])
dist_resid = tp_pca_dist[strict_mask] - reg_dist.predict(X_iou_conf)

# Point-biserial correlation (Pearson with binary y)
r_id_partial, p_id_partial = stats.pearsonr(id_resid, y)
r_dist_partial, p_dist_partial = stats.pearsonr(dist_resid, y)

print(f"Local ID residual  ~ TP/FP: r={r_id_partial:+.4f} (p={p_id_partial:.4e})")
print(f"PCA dist residual  ~ TP/FP: r={r_dist_partial:+.4f} (p={p_dist_partial:.4e})")

# 6b. Logistic regression: IoU+conf vs IoU+conf+geometry
# Model 1: IoU + conf only
X_base = X_iou_conf
lr_base = LogisticRegression(max_iter=1000, solver='lbfgs')
lr_base.fit(X_base, y)
auc_base = roc_auc_score(y, lr_base.predict_proba(X_base)[:, 1])

# Model 2: IoU + conf + local_id
X_id = np.column_stack([iou_vec[strict_mask], conf_vec[strict_mask], local_id[strict_mask]])
lr_id = LogisticRegression(max_iter=1000, solver='lbfgs')
lr_id.fit(X_id, y)
auc_id = roc_auc_score(y, lr_id.predict_proba(X_id)[:, 1])

# Model 3: IoU + conf + PCA dist
X_dist = np.column_stack([iou_vec[strict_mask], conf_vec[strict_mask], tp_pca_dist[strict_mask]])
lr_dist = LogisticRegression(max_iter=1000, solver='lbfgs')
lr_dist.fit(X_dist, y)
auc_dist = roc_auc_score(y, lr_dist.predict_proba(X_dist)[:, 1])

# Model 4: IoU + conf + both geometry
X_both = np.column_stack([iou_vec[strict_mask], conf_vec[strict_mask], local_id[strict_mask], tp_pca_dist[strict_mask]])
lr_both = LogisticRegression(max_iter=1000, solver='lbfgs')
lr_both.fit(X_both, y)
auc_both = roc_auc_score(y, lr_both.predict_proba(X_both)[:, 1])

print(f"\nLogistic regression AUC:")
print(f"  IoU + conf only:        {auc_base:.4f}")
print(f"  + local ID:             {auc_id:.4f}  (delta = {auc_id - auc_base:+.4f})")
print(f"  + PCA dist:             {auc_dist:.4f}  (delta = {auc_dist - auc_base:+.4f})")
print(f"  + both geometry:        {auc_both:.4f}  (delta = {auc_both - auc_base:+.4f})")

# 6c. Within-TP and within-FP: does geometry correlate with IoU?
print(f"\n{'='*70}")
print("KEY TEST B: Within-group geometry ~ IoU correlation")
print(f"{'='*70}")

# Within TP: geometry vs IoU (if geometry carries info beyond binary TP/FP)
if is_tp_vec.sum() > 10:
    r_id_tp, p_id_tp = stats.pearsonr(local_id[is_tp_vec], iou_vec[is_tp_vec])
    r_dist_tp, p_dist_tp = stats.pearsonr(tp_pca_dist[is_tp_vec], iou_vec[is_tp_vec])
    print(f"Within TP (n={is_tp_vec.sum()}):")
    print(f"  Local ID ~ IoU:  r={r_id_tp:+.4f} (p={p_id_tp:.4e})")
    print(f"  PCA dist ~ IoU:  r={r_dist_tp:+.4f} (p={p_dist_tp:.4e})")

if is_fp_vec.sum() > 10:
    r_id_fp, p_id_fp = stats.pearsonr(local_id[is_fp_vec], iou_vec[is_fp_vec])
    r_dist_fp, p_dist_fp = stats.pearsonr(tp_pca_dist[is_fp_vec], iou_vec[is_fp_vec])
    print(f"Within FP (n={is_fp_vec.sum()}):")
    print(f"  Local ID ~ IoU:  r={r_id_fp:+.4f} (p={p_id_fp:.4e})")
    print(f"  PCA dist ~ IoU:  r={r_dist_fp:+.4f} (p={p_dist_fp:.4e})")

# 6d. Confidence-matched TP vs FP (as before, but now as supplementary)
print(f"\n{'='*70}")
print("KEY TEST C: Confidence-matched TP vs FP (supplementary)")
print(f"{'='*70}")
conf_quintiles = np.percentile(conf_vec, [0, 20, 40, 60, 80, 100])
matched_results = []
for q in range(5):
    lo_q, hi_q = conf_quintiles[q], conf_quintiles[q + 1]
    mask_q = (conf_vec >= lo_q) & (conf_vec < hi_q) if q < 4 else (conf_vec >= lo_q) & (conf_vec <= hi_q)
    mask_tp_q = mask_q & is_tp_vec
    mask_fp_q = mask_q & is_fp_vec
    n_tp_q = mask_tp_q.sum()
    n_fp_q = mask_fp_q.sum()
    if n_tp_q < 5 or n_fp_q < 5:
        print(f"  Q{q+1} [{lo_q:.3f}-{hi_q:.3f}]: TP={n_tp_q}, FP={n_fp_q} - skip")
        continue
    id_tp_q = local_id[mask_tp_q]
    id_fp_q = local_id[mask_fp_q]
    dist_tp_q = tp_pca_dist[mask_tp_q]
    dist_fp_q = tp_pca_dist[mask_fp_q]
    d_id_q = cohens_d(id_tp_q, id_fp_q)
    d_dist_q = cohens_d(dist_tp_q, dist_fp_q)
    print(f"  Q{q+1} [{lo_q:.3f}-{hi_q:.3f}]: TP={n_tp_q}, FP={n_fp_q} | ID d={abs(d_id_q):.3f}, DIST d={abs(d_dist_q):.3f}")
    matched_results.append({
        "quintile": q + 1,
        "d_id_abs": abs(d_id_q),
        "d_dist_abs": abs(d_dist_q),
    })

# ---------------------------------------------------------------------------
# 7. Global summary: correlation of geometry with IoU (continuous)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("GLOBAL CORRELATION: Geometry vs IoU (continuous)")
print(f"{'='*70}")

# Use only strict TP and FP (exclude ambiguous 0.3-0.5)
strict_mask = is_tp_vec | is_fp_vec

if strict_mask.sum() > 10:
    # Local ID vs IoU
    r_id_iou, p_id_iou = stats.pearsonr(local_id[strict_mask], iou_vec[strict_mask])
    rho_id_iou, _ = stats.spearmanr(local_id[strict_mask], iou_vec[strict_mask])
    print(f"Local ID ~ IoU:    Pearson r={r_id_iou:.4f} (p={p_id_iou:.4e}), Spearman rho={rho_id_iou:.4f}")

    # PCA dist vs IoU
    r_dist_iou, p_dist_iou = stats.pearsonr(tp_pca_dist[strict_mask], iou_vec[strict_mask])
    rho_dist_iou, _ = stats.spearmanr(tp_pca_dist[strict_mask], iou_vec[strict_mask])
    print(f"PCA dist ~ IoU:    Pearson r={r_dist_iou:.4f} (p={p_dist_iou:.4e}), Spearman rho={rho_dist_iou:.4f}")

    # Local ID vs confidence
    r_id_conf, p_id_conf = stats.pearsonr(local_id[strict_mask], conf_vec[strict_mask])
    print(f"Local ID ~ Conf:   Pearson r={r_id_conf:.4f} (p={p_id_conf:.4e})")

    # PCA dist vs confidence
    r_dist_conf, p_dist_conf = stats.pearsonr(tp_pca_dist[strict_mask], conf_vec[strict_mask])
    print(f"PCA dist ~ Conf:   Pearson r={r_dist_conf:.4f} (p={p_dist_conf:.4e})")

# ---------------------------------------------------------------------------
# 8. Verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

# Collect all Cohen's d values
all_d_id = [r["d_local_id_abs"] for r in results if not np.isnan(r["d_local_id_abs"])]
all_d_dist = [r["d_pca_dist_abs"] for r in results if not np.isnan(r["d_pca_dist_abs"])]
all_d_matched_id = [r["d_id_abs"] for r in matched_results] if matched_results else []
all_d_matched_dist = [r["d_dist_abs"] for r in matched_results] if matched_results else []

max_d_id = max(all_d_id) if all_d_id else 0.0
max_d_dist = max(all_d_dist) if all_d_dist else 0.0
max_d_matched_id = max(all_d_matched_id) if all_d_matched_id else 0.0
max_d_matched_dist = max(all_d_matched_dist) if all_d_matched_dist else 0.0

print(f"Max |Cohen's d| - Local ID (per-bin):       {max_d_id:.3f}")
print(f"Max |Cohen's d| - PCA dist (per-bin):        {max_d_dist:.3f}")
print(f"Max |Cohen's d| - Local ID (conf-matched):   {max_d_matched_id:.3f}")
print(f"Max |Cohen's d| - PCA dist (conf-matched):   {max_d_matched_dist:.3f}")

# Partial correlation effect sizes (Cohen's conventions for r: 0.1=small, 0.3=medium, 0.5=large)
print(f"\nPartial correlation (residualized IoU+conf):")
print(f"  Local ID residual ~ TP/FP:  r={r_id_partial:+.4f}  (|r|={abs(r_id_partial):.4f})")
print(f"  PCA dist residual ~ TP/FP:  r={r_dist_partial:+.4f}  (|r|={abs(r_dist_partial):.4f})")

# Logistic regression AUC improvements
print(f"\nLogistic regression AUC improvement:")
print(f"  Base (IoU+conf):            {auc_base:.4f}")
print(f"  + local ID:                 {auc_id:.4f}  (delta={auc_id - auc_base:+.4f})")
print(f"  + PCA dist:                 {auc_dist:.4f}  (delta={auc_dist - auc_base:+.4f})")
print(f"  + both geometry:            {auc_both:.4f}  (delta={auc_both - auc_base:+.4f})")

# Threshold: 0.3 = small effect for Cohen's d, 0.1 = small for Pearson r
threshold_d = 0.3
threshold_r = 0.1
# NOTE: conf-matched Cohen's d does NOT control IoU (TP/FP still separated by IoU threshold)
# True "control IoU" evidence comes from partial correlation and within-group analysis
any_viable = (
    abs(r_id_partial) >= threshold_r
    or abs(r_dist_partial) >= threshold_r
    or (auc_id - auc_base) > 0.01
    or (auc_dist - auc_base) > 0.01
)

if any_viable:
    print(f"\n>>> VERDICT: VIABLE (|partial r| >= {threshold_r} or AUC delta > 0.01)")
    print(">>> FFT geometric features carry independent discriminative signal after controlling IoU/confidence.")
    print(">>> Recommendation: USE as auxiliary RLVR reward dimension.")
else:
    print(f"\n>>> VERDICT: NOT VIABLE (|partial r| < {threshold_r} and AUC delta <= 0.01)")
    print(">>> FFT geometric features do NOT carry independent signal after controlling IoU/confidence.")
    print(">>> The apparent separation in conf-matched analysis is fully explained by IoU (TP/FP are defined by IoU threshold).")
    print(">>> Recommendation: DO NOT USE as RLVR reward — signal is fully explained by IoU.")

print(f"{'='*70}")
