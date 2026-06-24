"""Misclassified proposal analysis for FFT geometric features.

Answers the researcher's core question:
"For MISCLASSIFIED samples (FP with high confidence, FN with low confidence),
is geometric signal stronger than overall?"

Steps:
1. Load baseline checkpoint
2. Extract all val proposals with decoded boxes, IoU, confidence
3. Define misclassification categories:
   - True Positive (TP): IoU >= 0.5, conf >= 0.5 (correctly detected)
   - False Positive (FP): IoU < 0.3, conf >= 0.5 (high conf, wrong location)
   - False Negative (FN): IoU >= 0.5, conf < 0.5 (low conf, correct location)
   - Ambiguous: 0.3 <= IoU < 0.5 (excluded from main analysis)
4. Compute FFT amplitude features (7168-dim) on ROI crops
5. PCA to 50 dims (fit on TP only, apply to all — no leakage)
6. Measure geometric discrimination:
   - Cohen's d for TP vs FP, TP vs FN, FP vs FN
   - Partial correlation controlling for IoU and conf
   - Within-misclassified-group analysis
7. Test relaxed IoU threshold: TP @ 0.3 vs 0.5
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
from sklearn.linear_model import LinearRegression
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEV}")
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

# Hooks
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
# 2. Extract proposals + FFT + IoU + confidence
# ---------------------------------------------------------------------------
_, vl = build_penn_fudan_loaders_320(batch_size=1)

all_iou = []
all_conf = []
all_is_tp_05 = []  # TP @ IoU >= 0.5
all_is_tp_03 = []  # TP @ IoU >= 0.3 (relaxed)
all_is_fp = []     # FP @ IoU < 0.3
all_fft_amp = []

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

    # FFT amplitude: (N, 256, 7, 4) -> (N, 7168)
    f = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")
    amp_raw = torch.abs(f)
    amp_flat = amp_raw.view(N, -1).cpu().numpy()

    if len(gt_boxes) == 0:
        for i in range(N):
            all_iou.append(0.0)
            all_conf.append(0.0)
            all_is_tp_05.append(False)
            all_is_tp_03.append(False)
            all_is_fp.append(True)
        all_fft_amp.append(amp_flat)
        continue

    ious = box_iou(decoded, gt_boxes)
    best_iou, best_gt = ious.max(dim=1)
    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]

    for i in range(N):
        iou_i = best_iou[i].item()
        conf_i = conf[i].item()
        all_iou.append(iou_i)
        all_conf.append(conf_i)
        all_is_tp_05.append(iou_i >= 0.5)
        all_is_tp_03.append(iou_i >= 0.3)
        all_is_fp.append(iou_i < 0.3)
    all_fft_amp.append(amp_flat)

# Concatenate
fft_mat = np.concatenate(all_fft_amp, axis=0)
iou_vec = np.array(all_iou)
conf_vec = np.array(all_conf)
is_tp_05 = np.array(all_is_tp_05)
is_tp_03 = np.array(all_is_tp_03)
is_fp = np.array(all_is_fp)
M = fft_mat.shape[0]

print(f"\nTotal proposals: {M}")
print(f"TP (IoU>=0.5): {is_tp_05.sum()}")
print(f"TP (IoU>=0.3, relaxed): {is_tp_03.sum()}")
print(f"FP (IoU<0.3): {is_fp.sum()}")
print(f"Ambiguous (0.3<=IoU<0.5): {(~is_tp_03 & ~is_fp).sum()}")

# ---------------------------------------------------------------------------
# 3. PCA — fit on TP only, apply to all (no data leakage)
# ---------------------------------------------------------------------------
tp_mask_for_pca = is_tp_05
if tp_mask_for_pca.sum() < 50:
    print(f"WARNING: Only {tp_mask_for_pca.sum()} TP samples, PCA may be unstable")
    tp_mask_for_pca = is_tp_03  # fall back to relaxed TP

pca = PCA(n_components=50, random_state=42)
pca.fit(fft_mat[tp_mask_for_pca])
pca_fft = pca.transform(fft_mat)
var_ratio = pca.explained_variance_ratio_.sum()
print(f"PCA(50) fit on TP-only, variance explained: {var_ratio:.4f}")

# ---------------------------------------------------------------------------
# 4. Define misclassification categories
# ---------------------------------------------------------------------------
# Using conf threshold = 0.5 for "high" vs "low" confidence
conf_thresh = 0.5

# Category definitions
cat_tp = is_tp_05 & (conf_vec >= conf_thresh)       # Correct detection
cat_fp = is_fp & (conf_vec >= conf_thresh)           # Misdetected: high conf, wrong location
cat_fn = is_tp_05 & (conf_vec < conf_thresh)         # Missed: low conf, good location (but wait...)
# Actually FN should be: good location but NOT detected by model (not in proposals or classified as bg)
# In our proposal-level analysis, FN = proposals with good IoU but low confidence
cat_ambig = ~is_tp_03 & ~is_fp                       # 0.3 <= IoU < 0.5

# Also define: "near-miss" = IoU 0.3-0.5 with high conf (potential FP that are close)
cat_near_miss = (~is_tp_03 & ~is_fp) & (conf_vec >= conf_thresh)

print(f"\n{'='*70}")
print("MISCLASSIFICATION CATEGORIES")
print(f"{'='*70}")
print(f"TP (IoU>=0.5, conf>=0.5):     {cat_tp.sum()}")
print(f"FP (IoU<0.3, conf>=0.5):       {cat_fp.sum()}")
print(f"FN-proxy (IoU>=0.5, conf<0.5): {cat_fn.sum()}")
print(f"Ambiguous (0.3<=IoU<0.5):     {cat_ambig.sum()}")
print(f"Near-miss (0.3<=IoU<0.5, conf>=0.5): {cat_near_miss.sum()}")

# ---------------------------------------------------------------------------
# 5. Geometric metrics
# ---------------------------------------------------------------------------
# PCA distance to TP centroid
tp_centroid = pca_fft[cat_tp].mean(axis=0)
tp_pca_dist = np.linalg.norm(pca_fft - tp_centroid, axis=1)

# Local intrinsic dimension (manual TwoNN)
k_nn = 10
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=k_nn * 2 + 1, metric="euclidean")
nn.fit(pca_fft)
dists, _ = nn.kneighbors(pca_fft)
r = dists[:, k_nn] / (dists[:, 2 * k_nn] + 1e-12)
r = np.clip(r, 1e-6, 1 - 1e-6)
local_id = np.log(2.0) / np.log(r)
local_id = np.clip(local_id, 0.5, 50.0)

# ---------------------------------------------------------------------------
# 6. Cohen's d analysis
# ---------------------------------------------------------------------------
def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std

print(f"\n{'='*70}")
print("COHEN'S D: GEOMETRIC DISCRIMINATION BETWEEN CATEGORIES")
print(f"{'='*70}")

comparisons = [
    ("TP", "FP", cat_tp, cat_fp),
    ("TP", "FN-proxy", cat_tp, cat_fn),
    ("TP", "Near-miss", cat_tp, cat_near_miss),
    ("FP", "Near-miss", cat_fp, cat_near_miss),
    ("FN-proxy", "Near-miss", cat_fn, cat_near_miss),
]

for name1, name2, mask1, mask2 in comparisons:
    n1, n2 = mask1.sum(), mask2.sum()
    if n1 < 5 or n2 < 5:
        print(f"\n{name1} vs {name2}: SKIP (n1={n1}, n2={n2})")
        continue

    d_id = cohens_d(local_id[mask1], local_id[mask2])
    d_dist = cohens_d(tp_pca_dist[mask1], tp_pca_dist[mask2])

    # Mann-Whitney U
    u_id, p_id = stats.mannwhitneyu(local_id[mask1], local_id[mask2], alternative="two-sided")
    u_dist, p_dist = stats.mannwhitneyu(tp_pca_dist[mask1], tp_pca_dist[mask2], alternative="two-sided")

    print(f"\n{name1} (n={n1}) vs {name2} (n={n2}):")
    print(f"  Local ID  — {name1} mean={np.mean(local_id[mask1]):.3f}, {name2} mean={np.mean(local_id[mask2]):.3f}, Cohen's d={d_id:+.3f} (|d|={abs(d_id):.3f}), p={p_id:.4e}")
    print(f"  PCA dist  — {name1} mean={np.mean(tp_pca_dist[mask1]):.3f}, {name2} mean={np.mean(tp_pca_dist[mask2]):.3f}, Cohen's d={d_dist:+.3f} (|d|={abs(d_dist):.3f}), p={p_dist:.4e}")

# ---------------------------------------------------------------------------
# 7. Partial correlation: geometry vs misclassification | IoU, conf
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("PARTIAL CORRELATION: GEOMETRY vs MISCLASSIFICATION | IoU, conf")
print(f"{'='*70}")

# Define binary targets
y_fp = cat_fp.astype(int)  # 1 = FP (high conf, low IoU), 0 = others
y_fn = cat_fn.astype(int)  # 1 = FN-proxy (low conf, high IoU), 0 = others
y_mis = (cat_fp | cat_fn).astype(int)  # 1 = any misclassification

for y_name, y_vec in [("FP", y_fp), ("FN-proxy", y_fn), ("Any-misclass", y_mis)]:
    mask = y_vec >= 0  # all valid
    X = np.column_stack([iou_vec[mask], conf_vec[mask]])

    # Residualize both geometry and target
    reg_id = LinearRegression().fit(X, local_id[mask])
    id_resid = local_id[mask] - reg_id.predict(X)

    reg_dist = LinearRegression().fit(X, tp_pca_dist[mask])
    dist_resid = tp_pca_dist[mask] - reg_dist.predict(X)

    reg_y = LinearRegression().fit(X, y_vec[mask])
    y_resid = y_vec[mask] - reg_y.predict(X)

    r_id, p_id = stats.pearsonr(id_resid, y_resid)
    r_dist, p_dist = stats.pearsonr(dist_resid, y_resid)

    print(f"\n{y_name} (n={mask.sum()}, pos={y_vec.sum()}):")
    print(f"  Local ID residual  ~ {y_name}: r={r_id:+.4f} (p={p_id:.4e})")
    print(f"  PCA dist residual  ~ {y_name}: r={r_dist:+.4f} (p={p_dist:.4e})")

# ---------------------------------------------------------------------------
# 8. Within-misclassified analysis: does geometry correlate with IoU?
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("WITHIN-GROUP CORRELATION: Geometry vs IoU")
print(f"{'='*70}")

for name, mask in [("TP", cat_tp), ("FP", cat_fp), ("FN-proxy", cat_fn), ("Near-miss", cat_near_miss)]:
    if mask.sum() < 10:
        print(f"\n{name}: SKIP (n={mask.sum()})")
        continue
    r_id, p_id = stats.pearsonr(local_id[mask], iou_vec[mask])
    r_dist, p_dist = stats.pearsonr(tp_pca_dist[mask], iou_vec[mask])
    r_conf_id, p_conf_id = stats.pearsonr(local_id[mask], conf_vec[mask])
    r_conf_dist, p_conf_dist = stats.pearsonr(tp_pca_dist[mask], conf_vec[mask])
    print(f"\n{name} (n={mask.sum()}):")
    print(f"  Local ID ~ IoU:  r={r_id:+.4f} (p={p_id:.4e})")
    print(f"  PCA dist ~ IoU:  r={r_dist:+.4f} (p={p_dist:.4e})")
    print(f"  Local ID ~ Conf: r={r_conf_id:+.4f} (p={p_conf_id:.4e})")
    print(f"  PCA dist ~ Conf: r={r_conf_dist:+.4f} (p={p_conf_dist:.4e})")

# ---------------------------------------------------------------------------
# 9. Relaxed IoU threshold analysis
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("RELAXED IoU THRESHOLD: TP@0.3 vs TP@0.5")
print(f"{'='*70}")

# New categories with relaxed threshold
cat_tp_03 = is_tp_03 & (conf_vec >= conf_thresh)
cat_fp_03 = (~is_tp_03) & (conf_vec >= conf_thresh)  # everything not TP@0.3 with high conf

print(f"TP@0.3 (conf>=0.5): {cat_tp_03.sum()}")
print(f"FP@0.3 (conf>=0.5): {cat_fp_03.sum()}")

if cat_tp_03.sum() >= 5 and cat_fp_03.sum() >= 5:
    d_id_03 = cohens_d(local_id[cat_tp_03], local_id[cat_fp_03])
    d_dist_03 = cohens_d(tp_pca_dist[cat_tp_03], tp_pca_dist[cat_fp_03])
    u_id_03, p_id_03 = stats.mannwhitneyu(local_id[cat_tp_03], local_id[cat_fp_03], alternative="two-sided")
    u_dist_03, p_dist_03 = stats.mannwhitneyu(tp_pca_dist[cat_tp_03], tp_pca_dist[cat_fp_03], alternative="two-sided")
    print(f"\nTP@0.3 vs FP@0.3:")
    print(f"  Local ID  — Cohen's d={d_id_03:+.3f} (|d|={abs(d_id_03):.3f}), p={p_id_03:.4e}")
    print(f"  PCA dist  — Cohen's d={d_dist_03:+.3f} (|d|={abs(d_dist_03):.3f}), p={p_dist_03:.4e}")

# Compare: TP@0.5 vs TP@0.3 (newly included near-miss)
cat_newly_tp = is_tp_03 & ~is_tp_05  # IoU 0.3-0.5, now considered TP
if cat_tp.sum() >= 5 and cat_newly_tp.sum() >= 5:
    d_id_new = cohens_d(local_id[cat_tp], local_id[cat_newly_tp])
    d_dist_new = cohens_d(tp_pca_dist[cat_tp], tp_pca_dist[cat_newly_tp])
    print(f"\nTP@0.5 vs newly-TP@0.3 (IoU 0.3-0.5):")
    print(f"  Local ID  — Cohen's d={d_id_new:+.3f} (|d|={abs(d_id_new):.3f})")
    print(f"  PCA dist  — Cohen's d={d_dist_new:+.3f} (|d|={abs(d_dist_new):.3f})")
    print(f"  Newly-TP mean IoU={np.mean(iou_vec[cat_newly_tp]):.3f}, conf={np.mean(conf_vec[cat_newly_tp]):.3f}")

# ---------------------------------------------------------------------------
# 10. Verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

print("""
Key questions answered:
1. Does geometry discriminate misclassified (FP high-conf, FN low-conf) from correct?
2. Is signal stronger in misclassified group than overall?
3. Does relaxing IoU threshold to 0.3 reveal new signal?

Thresholds: Cohen's d > 0.3 = small effect, |partial r| > 0.1 = small effect
""")

print(f"{'='*70}")
