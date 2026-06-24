"""Static validation: Phase-only reconstruction edge + manifold on PennFudan val."""
import sys, torch, numpy as np
import torch.nn.functional as F
from torchvision.ops import box_iou
from collections import defaultdict
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

set_seed(42); DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
model = bm().to(DEV); ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()

sampled_props, box_head_in = {}, {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, a: sampled_props.update({"p": [x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m, a: box_head_in.update({"x": a[0]}))

_, vl = build_penn_fudan_loaders_320(batch_size=1)

edge_scores, recon_vecs, ious, confs, gt_ids = [], [], [], [], []

for img, tgt in vl:
    img_d = [img[0].to(DEV)]; tgt_d = [{k: v.to(DEV) for k, v in tgt[0].items()}]
    sampled_props.clear(); box_head_in.clear()
    with torch.no_grad(): model(img_d, tgt_d)
    rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
    if rf is None or sp_raw is None or rf.shape[0] == 0: continue

    bf = model.roi_heads.box_head(rf)
    conf = F.softmax(model.roi_heads.box_predictor.cls_score(bf), dim=-1)[:, 1]
    reg = model.roi_heads.box_predictor.bbox_pred(bf)
    sp_cat = torch.cat(sp_raw, dim=0)
    decoded = model.roi_heads.box_predictor.bbox_pred(bf)[:, 2:6]
    from spectral_detection_posttrain.experiments.runner_utils import decode_boxes
    decoded_boxes = decode_boxes(sp_cat, decoded)
    gt = tgt_d[0]["boxes"]

    if len(gt) > 0:
        iou_mat = box_iou(decoded_boxes, gt)
        best_iou, best_gt = iou_mat.max(dim=1)
    else:
        best_iou = torch.zeros(len(sp_cat)); best_gt = torch.full((len(sp_cat),), -1, dtype=torch.long)

    # Phase-only reconstruction for each proposal
    full_img = img_d[0]  # (3, H, W)
    for i in range(len(sp_cat)):
        x1, y1, x2, y2 = sp_cat[i].long()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full_img.shape[2], x2), min(full_img.shape[1], y2)
        if x2 <= x1 or y2 <= y1: continue
        crop = full_img[:, y1:y2, x1:x2].unsqueeze(0)  # (1, 3, h, w)
        crop = F.interpolate(crop, (64, 64), mode='bilinear', align_corners=False).squeeze(0)  # (3, 64, 64)

        fft = torch.fft.rfft2(crop, dim=(-2, -1))
        phase = torch.angle(fft)
        fft_po = torch.exp(1j * phase)  # magnitude = 1, phase preserved
        recon = torch.fft.irfft2(fft_po, s=(64, 64))

        # Edge score (Sobel on grayscale)
        gray = 0.299*recon[0] + 0.587*recon[1] + 0.114*recon[2]
        sobel_x = torch.zeros_like(gray)
        sobel_y = torch.zeros_like(gray)
        sobel_x[1:-1, 1:-1] = (gray[2:, 1:-1] - gray[:-2, 1:-1]) / 8
        sobel_y[1:-1, 1:-1] = (gray[1:-1, 2:] - gray[1:-1, :-2]) / 8
        edge = (sobel_x**2 + sobel_y**2).sqrt()

        edge_scores.append(edge.mean().item())
        recon_vecs.append(recon.reshape(-1).cpu().numpy())
        ious.append(best_iou[i].item())
        confs.append(conf[i].item())
        gt_ids.append(best_gt[i].item())

edge_scores = np.array(edge_scores)
recon_vecs = np.stack(recon_vecs, axis=0)
ious = np.array(ious); confs = np.array(confs); gt_ids = np.array(gt_ids)
print(f"Proposals: {len(ious)}")

# Pair consistency: within same GT, does metric ranking agree with IoU ranking?
def pair_consistency(metric, ious, gt_ids):
    agree, total = 0, 0
    for gid in np.unique(gt_ids):
        if gid < 0: continue
        mask = gt_ids == gid; n = mask.sum()
        if n < 2: continue
        idxs = np.where(mask)[0]
        for i in range(n):
            for j in range(i+1, n):
                iou_order = ious[idxs[i]] > ious[idxs[j]]
                met_order = metric[idxs[i]] > metric[idxs[j]]
                total += 1
                if iou_order == met_order: agree += 1
    return 100 * agree / total if total > 0 else 0

# A: Edge score
pc_edge = pair_consistency(edge_scores, ious, gt_ids)
print(f"Phase-only edge scalar: {pc_edge:.1f}%")

# B: 12288-dim manifold
# Fit on TP only
tp_mask = ious > 0.5
scaler = StandardScaler().fit(recon_vecs[tp_mask])
scaled = scaler.transform(recon_vecs)
pca = PCA(n_components=50, whiten=True, random_state=42).fit(scaled[tp_mask])
whitened = pca.transform(scaled)
tp_center = np.median(whitened[tp_mask], axis=0)
manifold_dist = np.linalg.norm(whitened - tp_center, axis=1)
pc_manifold = pair_consistency(-manifold_dist, ious, gt_ids)
print(f"Phase-only recon manifold (12288→50): {pc_manifold:.1f}%")

# C: Raw image (control) — crop original, same pipeline
raw_vecs = []
for img, tgt in vl:
    img_d = [img[0].to(DEV)]
    sampled_props.clear(); box_head_in.clear()
    with torch.no_grad(): model(img_d, [{k: v.to(DEV) for k, v in tgt[0].items()}])
    sp_raw = sampled_props.get("p")
    if sp_raw is None: continue
    sp_cat = torch.cat(sp_raw, dim=0)
    full_img = img_d[0]
    for i in range(len(sp_cat)):
        x1, y1, x2, y2 = sp_cat[i].long()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full_img.shape[2], x2), min(full_img.shape[1], y2)
        if x2 <= x1 or y2 <= y1: continue
        crop = full_img[:, y1:y2, x1:x2].unsqueeze(0)
        crop = F.interpolate(crop, (64, 64), mode='bilinear', align_corners=False).squeeze(0)
        raw_vecs.append(crop.reshape(-1).cpu().numpy())
raw_vecs = np.stack(raw_vecs, axis=0)[:len(ious)]
scaler_r = StandardScaler().fit(raw_vecs[tp_mask])
scaled_r = scaler_r.transform(raw_vecs)
pca_r = PCA(n_components=50, whiten=True, random_state=42).fit(scaled_r[tp_mask])
whitened_r = pca_r.transform(scaled_r)
tp_center_r = np.median(whitened_r[tp_mask], axis=0)
manifold_dist_r = np.linalg.norm(whitened_r - tp_center_r, axis=1)
pc_raw = pair_consistency(-manifold_dist_r, ious, gt_ids)
print(f"Raw image manifold (12288→50): {pc_raw:.1f}%")

print(f"\nBenchmark: Isomap(6) ROI-FFT = 60.6%")
