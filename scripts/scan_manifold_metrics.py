"""Scan manifold distance metrics for DPO pair consistency.

Test 5 metrics on baseline val set (no model training):
  PCA distance, UMAP geodesic, k-NN local density, spectral cluster assign, diffusion distance.
Pair consistency = P(metric ranking agrees with IoU ranking) within same GT.
Target: >70% → worth DPO experiment.
"""
import sys, torch, numpy as np
import torch.nn.functional as F
from torchvision.ops import box_iou
from collections import defaultdict
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320, decode_boxes, extract_perchan_fft,
)
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed
from tqdm import tqdm
try: import umap
except: umap = None

set_seed(42); DEV, CKPT = "cuda", "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
model = bm().to(DEV); ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()

sampled_props, box_head_in = {}, {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, a: sampled_props.update({"p": [a_.clone() for a_ in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m, a: box_head_in.update({"x": a[0]}))

_, vl = build_penn_fudan_loaders_320(batch_size=1)

# Collect: [iou, conf, fft_vec, gt_id] per proposal
all_iou, all_conf, all_fft, all_gt_id = [], [], [], []
for img, tgt in tqdm(vl, desc="Collecting proposals"):
    img_d = [img[0].to(DEV)]; tgt_d = [{k: v.to(DEV) for k, v in tgt[0].items()}]
    sampled_props.clear(); box_head_in.clear()
    with torch.no_grad(): model(img_d, tgt_d)
    rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
    if rf is None or sp_raw is None or rf.shape[0] == 0: continue

    bf = model.roi_heads.box_head(rf)
    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]
    reg = model.roi_heads.box_predictor.bbox_pred(bf)
    sp_cat = torch.cat(sp_raw, dim=0); decoded = decode_boxes(sp_cat, reg[:, 2:6])
    gt = tgt_d[0]["boxes"]

    if len(gt) > 0:
        iou_mat = box_iou(decoded, gt)
        best_iou, best_gt = iou_mat.max(dim=1)
        roi_crops = model.roi_heads.box_roi_pool(model.backbone(img_d)["0"], [sp_cat], [(img_d[0].shape[-2], img_d[0].shape[-1])])
        f = extract_perchan_fft(roi_crops); ch = f.shape[1] // 6
        fft_vec = f[:, :3*ch].reshape(len(best_iou), -1).cpu().numpy()
    else:
        best_iou = torch.zeros(len(sp_cat)); best_gt = torch.zeros(len(sp_cat), dtype=torch.long)
        fft_vec = np.zeros((len(sp_cat), 1))

    all_iou.append(best_iou.cpu().numpy())
    all_conf.append(conf.cpu().numpy())
    all_fft.append(fft_vec)
    all_gt_id.append(best_gt.cpu().numpy())

iou = np.concatenate(all_iou); conf = np.concatenate(all_conf)
fft = np.concatenate(all_fft); gt_id = np.concatenate(all_gt_id)
print(f"\nProposals: {len(iou)}  TP(IoU>0.5): {(iou>0.5).sum()}  FP(IoU<0.3): {(iou<0.3).sum()}  Borderline: {((iou>=0.3)&(iou<0.5)).sum()}")

# --- PCA distance metric ---
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42).fit(fft)
pca_vec = pca.transform(fft)
tp_center_pca = pca_vec[iou > 0.5].mean(axis=0)
pca_dist = np.linalg.norm(pca_vec - tp_center_pca, axis=1)

# --- k-NN local density ---
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=30, metric="euclidean").fit(pca_vec)
knn_dist, _ = nn.kneighbors(pca_vec)
local_density = 1.0 / (knn_dist.mean(axis=1) + 1e-8)

# --- Spectral clustering assignment ---
from sklearn.cluster import SpectralClustering
try:
    spec = SpectralClustering(n_clusters=3, random_state=42, n_init=10, affinity="nearest_neighbors").fit(pca_vec)
    spec_labels = spec.labels_
except:
    spec_labels = np.zeros(len(pca_vec))

# --- Diffusion distance (approximation via diffusion maps) ---
from sklearn.manifold import spectral_embedding
diffusion_vec = spectral_embedding(NearestNeighbors(n_neighbors=30).fit(pca_vec).kneighbors_graph(pca_vec, mode="connectivity"),
                                    n_components=10, random_state=42)
tp_center_diff = diffusion_vec[iou > 0.5].mean(axis=0)
diff_dist = np.linalg.norm(diffusion_vec - tp_center_diff, axis=1)

# --- UMAP geodesic (if available) ---
if umap is not None:
    umap_obj = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.3, random_state=42).fit(fft)
    umap_vec = umap_obj.transform(fft)
    tp_center_umap = umap_vec[iou > 0.5].mean(axis=0)
    umap_dist = np.linalg.norm(umap_vec - tp_center_umap, axis=1)
else:
    umap_dist = pca_dist  # fallback

metrics = {
    "PCA_dist": pca_dist,
    "local_density": -local_density,  # negated: high density = good
    "spec_label": spec_labels,
    "diff_dist": diff_dist,
}
if umap is not None: metrics["UMAP_dist"] = umap_dist

# --- Pair consistency: within same GT, how often does metric ranking agree with IoU? ---
print(f"\n{'Metric':<16s} {'All_pairs':>8s} {'Consist%':>10s} {'Borderline%':>12s}")

for name, metric in metrics.items():
    if name == "spec_label":  # cluster label as 0/1/2 score
        score = metric
    else:
        score = metric

    total_pairs, agree_all, agree_bl = 0, 0, 0
    for gid in np.unique(gt_id):
        if gid < 0: continue
        mask = gt_id == gid
        n = mask.sum()
        if n < 2: continue
        for i in range(n):
            for j in range(i+1, n):
                idx_i, idx_j = np.where(mask)[0][i], np.where(mask)[0][j]
                iou_order = iou[idx_i] > iou[idx_j]
                metric_order = score[idx_i] > score[idx_j]
                total_pairs += 1
                if iou_order == metric_order:
                    agree_all += 1
                    if 0.3 <= iou[idx_i] < 0.5 or 0.3 <= iou[idx_j] < 0.5:
                        agree_bl += 1

    bl_pairs = sum(1 for gid in np.unique(gt_id) if gid >= 0
                   for i in range((gt_id == gid).sum())
                   for j in range(i+1, (gt_id == gid).sum())
                   if 0.3 <= iou[(gt_id == gid)][i] < 0.5 or 0.3 <= iou[(gt_id == gid)][j] < 0.5)
    print(f"{name:<16s} {total_pairs:8d} {100*agree_all/total_pairs:9.1f}% {100*agree_bl/bl_pairs if bl_pairs>0 else 0:11.1f}%")

# --- Hard: IoU-based vs metric-based pair selection overlap ---
print(f"\n{'Metric':<16s} {'Pair_overlap%':>14s}")
for name, metric in metrics.items():
    score = metric if name != "spec_label" else metric
    iou_pairs, met_pairs = set(), set()
    for gid in np.unique(gt_id):
        if gid < 0: continue; mask = gt_id == gid; n = mask.sum()
        if n < 2: continue
        idxs = np.where(mask)[0]
        # IoU pair: best vs worst
        iou_best, iou_worst = idxs[iou[mask].argmax()], idxs[iou[mask].argmin()]
        iou_pairs.add((iou_best, iou_worst))
        # Metric pair
        met_best, met_worst = idxs[score[mask].argmax()], idxs[score[mask].argmin()]
        met_pairs.add((met_best, met_worst))
    overlap = len(iou_pairs & met_pairs)
    total = len(iou_pairs)
    print(f"{name:<16s} {100*overlap/total:13.1f}%")

best_name = max(metrics, key=lambda n: 100*(sum(1 for gid in np.unique(gt_id) if gid>=0 for i in range((gt_id==gid).sum()) for j in range(i+1,(gt_id==gid).sum()) if (iou[(gt_id==gid)][i]>iou[(gt_id==gid)][j])==(metrics[n][(gt_id==gid)][i]>metrics[n][(gt_id==gid)][j])))/max(1,sum(1 for gid in np.unique(gt_id) if gid>=0 for i in range((gt_id==gid).sum()) for j in range(i+1,(gt_id==gid).sum()))))
print(f"\nBest metric: {best_name}")
print("Target: >70% consistency → worth DPO experiment")
