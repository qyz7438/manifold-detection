"""FFT spectral manifold independence verification — FIXED VERSION.

Controls for IoU and confidence, then tests whether FFT distances
carry independent discriminative signal for "same GT vs different GT".

Fixes over v1:
- PCA fit ONLY on TP (IoU > 0.5), then transform ALL proposals
- Use pingouin.partial_corr for true partial correlation (not manual residual)
- Raw 7168-dim baseline (no PCA) included
- UMAP(10) included if available
- Borderline proposals (IoU 0.3–0.5) kept, not excluded

Conclusion threshold: |partial_corr| > 0.10 => viable.
"""
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from torchvision.ops import box_iou, nms
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
    extract_perchan_fft,
)
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed

warnings.filterwarnings("ignore")
set_seed(42)
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"

# Optional UMAP
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Optional pingouin (install if missing)
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except Exception:
    HAS_PINGOUIN = False

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
model = bm().to(DEV)
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
# 2. Extract proposals + FFT + IoU + confidence
# ---------------------------------------------------------------------------
_, vl = build_penn_fudan_loaders_320(batch_size=1)

all_iou = []
all_conf = []
all_gt_id = []
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

    # Raw FFT amplitude per bin: (N, C, 7, 4) -> (N, 7168)
    f = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")  # (N, C, 7, 4)
    amp_raw = torch.abs(f)  # (N, C, 7, 4)
    amp_flat = amp_raw.view(N, -1).cpu().numpy()  # (N, 7168)

    if len(gt_boxes) == 0:
        for i in range(N):
            all_iou.append(0.0)
            all_conf.append(0.0)
            all_gt_id.append(-1)
        all_fft_amp.append(amp_flat)
        continue

    ious = box_iou(decoded, gt_boxes)  # (N, G)
    best_iou, best_gt = ious.max(dim=1)  # (N,)
    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]

    for i in range(N):
        all_iou.append(best_iou[i].item())
        all_conf.append(conf[i].item())
        all_gt_id.append(best_gt[i].item())
    all_fft_amp.append(amp_flat)

# Concatenate all proposals
fft_mat = np.concatenate(all_fft_amp, axis=0)  # (M, 7168)
iou_vec = np.array(all_iou)
conf_vec = np.array(all_conf)
gt_vec = np.array(all_gt_id)
M = fft_mat.shape[0]
print(f"\nTotal proposals: {M}")

# ---------------------------------------------------------------------------
# 3. Dimensionality reduction
# ---------------------------------------------------------------------------
# 3a. PCA: fit ONLY on TP (IoU > 0.5), transform ALL
tp_mask = iou_vec > 0.5
print(f"TP proposals (IoU>0.5): {tp_mask.sum()}")
print(f"Borderline (0.3<IoU<=0.5): {((iou_vec > 0.3) & (iou_vec <= 0.5)).sum()}")
print(f"FP (IoU<=0.3): {(iou_vec <= 0.3).sum()}")

pca = PCA(n_components=50, random_state=42)
pca.fit(fft_mat[tp_mask])
pca_fft = pca.transform(fft_mat)  # (M, 50)
var_ratio = pca.explained_variance_ratio_.sum()
print(f"PCA(50) variance explained (fit on TP only): {var_ratio:.4f}")

# 3b. UMAP(10) if available — fit on TP, transform all
umap_fft = None
if HAS_UMAP:
    umap_reducer = umap.UMAP(n_components=10, random_state=42, metric="euclidean")
    umap_reducer.fit(fft_mat[tp_mask])
    umap_fft = umap_reducer.transform(fft_mat)
    print(f"UMAP(10) fitted on TP, transformed all.")
else:
    print("UMAP not available (pip install umap-learn), skipping.")

# 3c. Raw 7168-dim (no reduction)
print(f"Raw 7168-dim ready.")

# ---------------------------------------------------------------------------
# 4. Pairwise analysis: sample proposal pairs within same image
# ---------------------------------------------------------------------------
img_boundaries = [0]
for arr in all_fft_amp:
    img_boundaries.append(img_boundaries[-1] + arr.shape[0])

img_id = np.zeros(M, dtype=int)
for idx in range(len(img_boundaries) - 1):
    start, end = img_boundaries[idx], img_boundaries[idx + 1]
    img_id[start:end] = idx

np.random.seed(42)
max_pairs_per_img = 2000
pair_data = []

for idx in range(len(img_boundaries) - 1):
    start, end = img_boundaries[idx], img_boundaries[idx + 1]
    n = end - start
    if n < 2:
        continue
    idxs = np.arange(start, end)
    gts = gt_vec[start:end]
    n_pairs = min(max_pairs_per_img, n * (n - 1) // 2)
    if n_pairs <= 0:
        continue
    chosen = set()
    attempts = 0
    while len(chosen) < n_pairs and attempts < n_pairs * 10:
        i, j = np.random.choice(idxs, 2, replace=False)
        if i > j:
            i, j = j, i
        chosen.add((i, j))
        attempts += 1
    for i, j in chosen:
        same_gt = 1 if (gt_vec[i] == gt_vec[j] and gt_vec[i] >= 0) else 0
        iou_diff = abs(iou_vec[i] - iou_vec[j])
        conf_diff = abs(conf_vec[i] - conf_vec[j])
        pca_dist = np.linalg.norm(pca_fft[i] - pca_fft[j])
        raw_dist = np.linalg.norm(fft_mat[i] - fft_mat[j])
        pair_data.append({
            "same_gt": same_gt,
            "iou_diff": iou_diff,
            "conf_diff": conf_diff,
            "pca_dist": pca_dist,
            "raw_dist": raw_dist,
        })

if HAS_UMAP:
    for k, (i, j) in enumerate([(p["i"], p["j"]) for p in pair_data]):
        # Need to re-add i,j to pair_data for UMAP; do it inline above
        pass

# Rebuild with UMAP distances if available
pair_data = []
for idx in range(len(img_boundaries) - 1):
    start, end = img_boundaries[idx], img_boundaries[idx + 1]
    n = end - start
    if n < 2:
        continue
    idxs = np.arange(start, end)
    n_pairs = min(max_pairs_per_img, n * (n - 1) // 2)
    if n_pairs <= 0:
        continue
    chosen = set()
    attempts = 0
    while len(chosen) < n_pairs and attempts < n_pairs * 10:
        i, j = np.random.choice(idxs, 2, replace=False)
        if i > j:
            i, j = j, i
        chosen.add((i, j))
        attempts += 1
    for i, j in chosen:
        same_gt = 1 if (gt_vec[i] == gt_vec[j] and gt_vec[i] >= 0) else 0
        iou_diff = abs(iou_vec[i] - iou_vec[j])
        conf_diff = abs(conf_vec[i] - conf_vec[j])
        pca_dist = np.linalg.norm(pca_fft[i] - pca_fft[j])
        raw_dist = np.linalg.norm(fft_mat[i] - fft_mat[j])
        d = {
            "same_gt": same_gt,
            "iou_diff": iou_diff,
            "conf_diff": conf_diff,
            "pca_dist": pca_dist,
            "raw_dist": raw_dist,
        }
        if HAS_UMAP:
            d["umap_dist"] = np.linalg.norm(umap_fft[i] - umap_fft[j])
        pair_data.append(d)

print(f"Sampled pairs: {len(pair_data)}")

# ---------------------------------------------------------------------------
# 5. Partial correlation analysis
# ---------------------------------------------------------------------------
same_gt = np.array([p["same_gt"] for p in pair_data])
iou_diff = np.array([p["iou_diff"] for p in pair_data])
conf_diff = np.array([p["conf_diff"] for p in pair_data])
pca_dist = np.array([p["pca_dist"] for p in pair_data])
raw_dist = np.array([p["raw_dist"] for p in pair_data])

if HAS_UMAP:
    umap_dist = np.array([p["umap_dist"] for p in pair_data])

# Helper: compute partial correlation via pingouin or fallback

def safe_partial_corr(x, y, covar1, covar2):
    """Return partial correlation r, p-value, and method name."""
    if HAS_PINGOUIN:
        df = {
            "x": x,
            "y": y,
            "covar1": covar1,
            "covar2": covar2,
        }
        # pingouin.partial_corr expects DataFrame
        import pandas as pd
        pdf = pd.DataFrame(df)
        try:
            res = pg.partial_corr(data=pdf, x="x", y="y", covar=["covar1", "covar2"], method="pearson")
            r = res["r"].values[0]
            pval = res["p-unc"].values[0]
            return float(r), float(pval), "pingouin"
        except Exception as e:
            print(f"  pingouin failed ({e}), falling back to manual residual")
    # Fallback: manual residual (semi-partial approximation)
    from sklearn.linear_model import LinearRegression
    X = np.column_stack([covar1, covar2])
    reg_x = LinearRegression().fit(X, x)
    reg_y = LinearRegression().fit(X, y)
    x_res = x - reg_x.predict(X)
    y_res = y - reg_y.predict(X)
    r = np.corrcoef(x_res, y_res)[0, 1]
    # Approximate p via Fisher transform
    n = len(x)
    if abs(r) < 1.0:
        z = np.arctanh(r) * np.sqrt(n - 3)
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        pval = 0.0
    return float(r), float(pval), "manual_residual"

results = {}

# 5a. Raw 7168 dim
print("\n--- Raw 7168-dim Euclidean distance ---")
r, p, method = safe_partial_corr(raw_dist, same_gt, iou_diff, conf_diff)
results["raw_7168"] = {"r": r, "p": p, "method": method}
print(f"  partial_corr (raw_dist ~ same_gt | iou_diff, conf_diff): r={r:.4f}, p={p:.4e} [{method}]")

# 5b. PCA(50, fit on TP)
print("\n--- PCA(50), fit on TP only ---")
r, p, method = safe_partial_corr(pca_dist, same_gt, iou_diff, conf_diff)
results["pca50_tpfit"] = {"r": r, "p": p, "method": method}
print(f"  partial_corr (pca_dist ~ same_gt | iou_diff, conf_diff): r={r:.4f}, p={p:.4e} [{method}]")

# 5c. UMAP(10) if available
if HAS_UMAP:
    print("\n--- UMAP(10), fit on TP only ---")
    r, p, method = safe_partial_corr(umap_dist, same_gt, iou_diff, conf_diff)
    results["umap10_tpfit"] = {"r": r, "p": p, "method": method}
    print(f"  partial_corr (umap_dist ~ same_gt | iou_diff, conf_diff): r={r:.4f}, p={p:.4e} [{method}]")

# ---------------------------------------------------------------------------
# 6. Conclusion
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
for name, vals in results.items():
    flag = "✓ VIABLE" if abs(vals["r"]) > 0.10 else "✗ NOT VIABLE"
    print(f"  {name:<20s}: r={vals['r']:+.4f}  p={vals['p']:.2e}  {flag}  [{vals['method']}]")
print(f"{'='*60}")

# Also print simple correlations for context
print("\nSimple correlations (for context, NOT controlling covariates):")
print(f"  raw_dist  ~ same_gt: {np.corrcoef(raw_dist, same_gt)[0,1]:+.4f}")
print(f"  pca_dist  ~ same_gt: {np.corrcoef(pca_dist, same_gt)[0,1]:+.4f}")
if HAS_UMAP:
    print(f"  umap_dist ~ same_gt: {np.corrcoef(umap_dist, same_gt)[0,1]:+.4f}")

print(f"\nThreshold: |partial_corr| > 0.10 => viable independent signal")
