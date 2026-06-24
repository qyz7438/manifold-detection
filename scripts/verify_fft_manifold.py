"""FFT spectral manifold independence verification.

Controls for IoU and confidence, then tests whether PCA-reduced FFT distances
carry independent discriminative signal for "same GT vs different GT".

Steps:
1. Load baseline checkpoint
2. Run inference on 34 val images, extract per-proposal:
   - IoU with GT (max IoU per proposal)
   - confidence
   - 7168-dim FFT amplitude spectrum (extract_perchan_fft amplitude bands)
3. PCA to 6 dims (local intrinsic dimension estimate)
4. Compute pairwise Euclidean distances in PCA space
5. Partial correlation: PCA distance ~ same_GT | IoU, confidence
6. Also test no-PCA (raw z-score 768-dim) for comparison
7. Conclusion: partial_corr > 0.15 => viable, < 0.05 => not viable
"""
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

    # Raw FFT amplitude per bin
    # crops: (N, C, H, W) where H=W=7
    f = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")  # (N, C, 7, 4)
    amp_raw = torch.abs(f)  # (N, C, 7, 4)
    amp_flat = amp_raw.view(N, -1).cpu().numpy()  # (N, C*7*4) = (N, 7168)

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

# z-score normalization
scaler = StandardScaler()
fft_z = scaler.fit_transform(fft_mat)

# ---------------------------------------------------------------------------
# 3. Evaluate three configurations
# ---------------------------------------------------------------------------

def evaluate_config(X: np.ndarray, label: str, pca_dim: int | None) -> dict:
    """Sample pairs and compute partial correlation for a given embedding."""
    print(f"\n{'='*60}")
    print(f"[Config] {label}  (dim={X.shape[1]})")
    print(f"{'='*60}")

    # Reconstruct image boundaries
    img_boundaries = [0]
    for arr in all_fft_amp:
        img_boundaries.append(img_boundaries[-1] + arr.shape[0])

    img_id = np.zeros(M, dtype=int)
    for idx in range(len(img_boundaries) - 1):
        start, end = img_boundaries[idx], img_boundaries[idx + 1]
        img_id[start:end] = idx

    # Sample pairs: within same image, same GT vs different GT
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
            dist = np.linalg.norm(X[i] - X[j])
            pair_data.append({
                "same_gt": same_gt,
                "iou_diff": iou_diff,
                "conf_diff": conf_diff,
                "dist": dist,
            })

    print(f"Sampled pairs: {len(pair_data)}")

    # Partial correlation: distance ~ same_GT | IoU_diff, conf_diff
    same_gt = np.array([p["same_gt"] for p in pair_data])
    iou_diff = np.array([p["iou_diff"] for p in pair_data])
    conf_diff = np.array([p["conf_diff"] for p in pair_data])
    dist = np.array([p["dist"] for p in pair_data])

    from sklearn.linear_model import LinearRegression

    Xreg = np.column_stack([iou_diff, conf_diff])
    reg = LinearRegression().fit(Xreg, dist)
    dist_resid = dist - reg.predict(Xreg)

    reg2 = LinearRegression().fit(Xreg, same_gt)
    same_gt_resid = same_gt - reg2.predict(Xreg)

    partial_corr = np.corrcoef(dist_resid, same_gt_resid)[0, 1]
    simple_corr = np.corrcoef(dist, same_gt)[0, 1]

    print(f"Partial correlation (dist ~ same_GT | IoU_diff, conf_diff): {partial_corr:.4f}")
    print(f"Simple correlation (dist ~ same_GT):                    {simple_corr:.4f}")

    return {
        "label": label,
        "pca_dim": pca_dim,
        "embedding_dim": int(X.shape[1]),
        "n_pairs": len(pair_data),
        "partial_corr": float(partial_corr),
        "simple_corr": float(simple_corr),
    }


results = []

# Config 1: PCA=50 (original)
pca50 = PCA(n_components=50, random_state=42)
pca50_fft = pca50.fit_transform(fft_z)
var50 = pca50.explained_variance_ratio_.sum()
print(f"PCA(50) variance explained: {var50:.4f}")
results.append(evaluate_config(pca50_fft, "PCA=50", 50))

# Config 2: PCA=6 (local intrinsic dimension estimate)
pca6 = PCA(n_components=6, random_state=42)
pca6_fft = pca6.fit_transform(fft_z)
var6 = pca6.explained_variance_ratio_.sum()
print(f"PCA(6) variance explained: {var6:.4f}")
results.append(evaluate_config(pca6_fft, "PCA=6", 6))

# Config 3: No PCA (raw z-score)
results.append(evaluate_config(fft_z, "Raw z-score (no PCA)", None))

# ---------------------------------------------------------------------------
# 4. Summary comparison
# ---------------------------------------------------------------------------

print(f"\n{'='*70}")
print("SUMMARY COMPARISON TABLE")
print(f"{'='*70}")
print(f"{'Config':<25s} {'Partial r':>12s} {'Simple r':>12s} {'Variance':>12s}")
print(f"{'-'*70}")
for r in results:
    var_str = f"{var50:.4f}" if r["pca_dim"] == 50 else (f"{var6:.4f}" if r["pca_dim"] == 6 else "N/A")
    print(f"{r['label']:<25s} {r['partial_corr']:12.4f} {r['simple_corr']:12.4f} {var_str:>12s}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
for r in results:
    pc = abs(r["partial_corr"])
    if pc > 0.15:
        conclusion = "可行 (viable) — FFT 流形方向携带独立判别信号"
    elif pc < 0.05:
        conclusion = "不可行 (not viable) — 控制 IoU/confidence 后无独立信号"
    else:
        conclusion = "边缘 (marginal) — 信号微弱但非零"
    print(f"  {r['label']}: {conclusion}")

print(f"{'='*70}")
