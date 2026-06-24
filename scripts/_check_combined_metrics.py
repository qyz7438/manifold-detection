"""Check if combining energy + geometric metrics improves FN prediction."""
import sys
import torch, numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; SEED = 42
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(SEED)

model = build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

fpn_feats_dict = {}
model.backbone.register_forward_hook(lambda m, i, o: fpn_feats_dict.update({"f": {k: o[k] for k in o if k != "pool"}}))
box_pool = model.roi_heads.box_roi_pool

tl, vl = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})

def compute_energy(fft_f):
    ch = fft_f.shape[1] // 6
    a_lo = fft_f[:, 0*ch:1*ch].sum(dim=1)
    a_total = a_lo + fft_f[:, 1*ch:2*ch].sum(dim=1) + fft_f[:, 2*ch:3*ch].sum(dim=1) + 1e-8
    return 2 * (a_lo / a_total) - 1

def extract_perchan_fft(x):
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X**2+Y**2); R = r.max().clamp_min(1e-6); rn = r/R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp*lo).flatten(2).sum(2); a_md = (amp*md).flatten(2).sum(2); a_hi = (amp*hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi], dim=1)

all_data = []

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]
    img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    fpn_feats_dict.clear()
    with torch.no_grad(): preds = model(imgs_d)
    fpn = fpn_feats_dict.get("f")
    if fpn is None: continue

    for i_img in range(len(imgs_d)):
        gt_boxes = tgts[i_img]["boxes"].to(DEV)
        pred_boxes = preds[i_img]["boxes"]

        gt_list = [gt_boxes]
        pooled_gt = box_pool(fpn, gt_list, [img_shape])
        fft_f = extract_perchan_fft(pooled_gt)
        energy = compute_energy(fft_f)

        if len(pred_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            best_iou, _ = ious.max(dim=0)
        else:
            best_iou = torch.zeros(len(gt_boxes))

        for gi in range(len(gt_boxes)):
            box = gt_boxes[gi]; w = box[2]-box[0]; h = box[3]-box[1]
            all_data.append({
                "iou": best_iou[gi].item(),
                "energy": energy[gi].item(),
                "area": (w*h).item(),
                "cx": ((box[0]+box[2])/2).item(),
                "cy": ((box[1]+box[3])/2).item(),
                "edge": min(box[0], img_shape[1]-box[2], box[1], img_shape[0]-box[3]).item(),
                "aspect": (w/max(h,1)).item(),
            })

print(f"N={len(all_data)} GT boxes")

# Normalize features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_raw = np.column_stack([
    [d["energy"] for d in all_data],
    [d["area"] for d in all_data],
    [d["cx"] for d in all_data],
    [d["cy"] for d in all_data],
    [d["edge"] for d in all_data],
    [d["aspect"] for d in all_data],
])
y_iou = np.array([d["iou"] for d in all_data])
y_fn = (y_iou < 0.5).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Fit on full data (n=91, 7 FN) for diagnostic purposes
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y_fn)
y_prob = clf.predict_proba(X)[:, 1]

from sklearn.metrics import roc_auc_score
auc_full = roc_auc_score(y_fn, y_prob)
print(f"\nCombined (energy+area+cx+cy+edge+aspect): AUC={auc_full:.4f} (in-sample)")
print(f"  Coefficients: energy={clf.coef_[0][0]:+.3f} area={clf.coef_[0][1]:+.3f} "
      f"cx={clf.coef_[0][2]:+.3f} cy={clf.coef_[0][3]:+.3f} "
      f"edge={clf.coef_[0][4]:+.3f} aspect={clf.coef_[0][5]:+.3f}")

# Energy-only
clf_en = LogisticRegression(max_iter=1000)
clf_en.fit(X[:, 0:1], y_fn)
auc_en = roc_auc_score(y_fn, clf_en.predict_proba(X[:, 0:1])[:, 1])
print(f"Energy-only:                   AUC={auc_en:.4f}")

# Test all combinations on full data
import itertools
print(f"\n=== All combinations (in-sample AUC) ===")
combos = []
for k in range(1, 7):
    for combo in itertools.combinations(range(6), k):
        Xc = X[:, list(combo)]
        c = LogisticRegression(max_iter=1000).fit(Xc, y_fn)
        auc = roc_auc_score(y_fn, c.predict_proba(Xc)[:, 1])
        combos.append((auc, combo))
combos.sort(key=lambda x: -x[0])

names = ["energy", "area", "cx", "cy", "edge", "aspect"]
for i, (auc, combo) in enumerate(combos[:10]):
    cn = "+".join([names[j] for j in combo])
    print(f"  {cn:<35s} AUC={auc:.4f}")
