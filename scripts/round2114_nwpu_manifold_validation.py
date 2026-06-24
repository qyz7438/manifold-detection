import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import decode_boxes
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
SEED = 42
BATCH = 1
MAX_SIZE = 480
NUM_CLASSES = 11
DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
CKPT_PATH = Path("runs/round2100_nwpu_baseline/checkpoint_best.pth")
CALIB_PKL = Path("runs/round2113_nwpu_manifold_calib.pkl")

set_seed(SEED)

# ---------------------------------------------------------------------------
# NWPU dataset (val only)
# ---------------------------------------------------------------------------

class NWPUDataset(Dataset):
    def __init__(self, root, coco_json, img_ids, max_size):
        self.root = Path(root)
        self.max_size = max_size
        self.coco = json.loads(Path(coco_json).read_text())
        self.img_infos = {img["id"]: img for img in self.coco["images"] if img["id"] in img_ids}
        self.img_ids = list(self.img_infos.keys())
        anns = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in img_ids:
                anns.setdefault(ann["image_id"], []).append(ann)
        self.anns = anns

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists():
            img_path = self.root / "negative image set" / info["file_name"]
        img = Image.open(str(img_path)).convert("RGB")
        img_t = TF.to_tensor(img)
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        _, H, W = img_t.shape
        if max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
            new_h, new_w = int(H * scale), int(W * scale)
            img_t = F.interpolate(img_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear").squeeze(0)
            target["boxes"] = target["boxes"] * scale
        return img_t, target


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# FFT manifold feature extraction (matches round2113 exactly)
# ---------------------------------------------------------------------------

def extract_amp_lo_perchan_stats(roi_features):
    """Extract amp_lo per-channel stats from ROI features (before box_head).

    Returns (N, C*3) numpy array: per-channel [mean, std, max] over 28 freq bins.
    """
    fft = torch.fft.rfft2(roi_features, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)
    freq_h = torch.fft.fftfreq(roi_features.shape[-2], device=roi_features.device)
    freq_w = torch.fft.rfftfreq(roi_features.shape[-1], device=roi_features.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = radius / radius.max().clamp_min(1e-6)
    lo_mask = (radius <= 0.3).float().unsqueeze(0).unsqueeze(0)
    amp_lo = amp * lo_mask
    flat = amp_lo.reshape(amp_lo.shape[0], amp_lo.shape[1], -1)
    mu = flat.mean(dim=-1)
    sg = flat.std(dim=-1)
    mx = flat.max(dim=-1).values
    return torch.cat([mu, sg, mx], dim=1).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Collect proposals on val set with IoU>0.1 threshold
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_proposals(model, val_loader, device):
    """Collect per-proposal IoU, confidence, and amp_lo features on val set."""
    model.eval()
    records = []
    sampled_props, box_head_in = {}, {}

    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    for images, targets in tqdm(val_loader, desc="inference"):
        images_d = [img.to(device) for img in images]
        tgts_t = [{k: v.to(device) for k, v in t.items()} for t in targets]
        sampled_props.clear()
        box_head_in.clear()

        model(images_d, tgts_t)

        sp_raw = sampled_props.get("p")
        rf = box_head_in.get("x")
        if sp_raw is None or rf is None or rf.shape[0] == 0:
            continue

        # classifier confidence: max foreground logit (same as round2113)
        bf = model.roi_heads.box_head(rf)
        cls_logits = model.roi_heads.box_predictor.cls_score(bf)
        fg_max, _ = cls_logits[:, 1:].max(dim=-1)  # (N,)
        conf = fg_max.cpu()

        # IoU computation: class-agnostic deltas using class 1 (airplane)
        sp_cat = torch.cat(sp_raw, dim=0)
        N = sp_cat.shape[0]
        reg_out = model.roi_heads.box_predictor.bbox_pred(bf[:N])
        deltas = reg_out[:, 4:8]  # class 1 deltas (same as round2113)
        decoded = decode_boxes(sp_cat, deltas)

        # amp_lo per-channel stats (N, C*3)
        amp_lo_feats = extract_amp_lo_perchan_stats(rf[:N])

        offset = 0
        for i_img, p_img in enumerate(sp_raw):
            n_p = p_img.shape[0]
            if n_p == 0:
                continue
            gt = tgts_t[i_img]["boxes"]
            gt_labels = tgts_t[i_img]["labels"]

            iou = torch.zeros(n_p)
            gt_idx = torch.full((n_p,), -1, dtype=torch.long)
            if len(gt) > 0:
                iou_mat = box_iou(decoded[offset:offset + n_p], gt)
                iou, best_gt = iou_mat.max(dim=1)
                gt_idx = best_gt
                iou = iou.cpu()
                gt_idx = gt_idx.cpu()

            records.append({
                "iou": iou,
                "conf": conf[offset:offset + n_p],
                "feat": amp_lo_feats[offset:offset + n_p],
                "gt_idx": gt_idx,
                "boxes": sp_cat[offset:offset + n_p].cpu(),
            })
            offset += n_p

    return records


# ---------------------------------------------------------------------------
# Group analysis
# ---------------------------------------------------------------------------

def analyze_group(records, tp_fn, fp_fn, scaler, pca, k):
    """tp_fn(iou) -> bool, fp_fn(iou) -> bool."""
    all_conf = []
    all_iou = []
    all_feat = []
    pair_agree = 0
    pair_total = 0

    for rec in records:
        iou = rec["iou"]
        conf = rec["conf"]
        feat = rec["feat"]
        n = len(iou)
        for i in range(n):
            all_feat.append(feat[i])
            all_conf.append(float(conf[i].item()))
            all_iou.append(float(iou[i].item()))

    X = np.stack(all_feat, axis=0)
    # Apply precomputed scaler + PCA
    Xs = scaler.transform(X)
    X_pca = pca.transform(Xs)[:, :k]

    # manifold distance to TP median
    tp_median = CALIB["tp_median"]
    d = np.linalg.norm(X_pca - tp_median, axis=1)
    # invert: smaller distance = higher score
    manifold_score = -d

    # Build binary labels per group
    tp_mask = np.array([tp_fn(i) for i in all_iou])
    fp_mask = np.array([fp_fn(i) for i in all_iou])

    # Filter to group members
    group_mask = tp_mask | fp_mask
    if group_mask.sum() < 10:
        return None

    y = tp_mask[group_mask].astype(int)
    c = np.array(all_conf)[group_mask]
    m = manifold_score[group_mask]

    # AUC(confidence)
    auc_conf = roc_auc_score(y, c) if len(np.unique(y)) > 1 else 0.5
    # AUC(confidence + manifold_dist)
    # Since manifold_score is negative distance, adding to conf gives combined score
    combined = c + m
    auc_comb = roc_auc_score(y, combined) if len(np.unique(y)) > 1 else 0.5

    # Pair consistency: for all pairs within same GT, does manifold order match IoU order?
    # We only compute on records (per-image) to avoid cross-image pairing
    for rec in records:
        iou = rec["iou"].numpy()
        conf_img = rec["conf"].numpy()
        feat_img = rec["feat"]
        gt_idx = rec["gt_idx"].numpy()
        groups = {}
        for p in range(len(iou)):
            if gt_idx[p] >= 0:
                groups.setdefault(int(gt_idx[p]), []).append(p)
        for gid, idxs in groups.items():
            if len(idxs) < 2:
                continue
            # filter to group members
            valid = [idx for idx in idxs if tp_fn(iou[idx]) or fp_fn(iou[idx])]
            if len(valid) < 2:
                continue
            # extract features for valid
            feats = feat_img[valid]
            Xv = np.stack(feats, axis=0)
            Xvs = scaler.transform(Xv)
            Xvp = pca.transform(Xvs)[:, :k]
            dv = np.linalg.norm(Xvp - tp_median, axis=1)
            mv = -dv
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    a, b = valid[i], valid[j]
                    metric_order = mv[i] > mv[j]
                    iou_order = iou[a] > iou[b]
                    if metric_order == iou_order:
                        pair_agree += 1
                    pair_total += 1

    pair_consistency = pair_agree / pair_total if pair_total > 0 else 0.0

    return {
        "n": int(group_mask.sum()),
        "n_tp": int(tp_mask.sum()),
        "n_fp": int(fp_mask.sum()),
        "auc_conf": float(auc_conf),
        "auc_comb": float(auc_comb),
        "delta_auc": float(auc_comb - auc_conf),
        "pair_consistency": float(pair_consistency),
        "pair_total": int(pair_total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading calibration...")
    import pickle
    global CALIB
    CALIB = pickle.load(open(CALIB_PKL, "rb"))
    scaler = CALIB["scaler"]
    pca = CALIB["pca"]
    k = CALIB["k"]
    print(f"  PCA k={k}, n_tp={CALIB['n_tp']}")

    print("Loading model...")
    model = build_detector({
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_fpn",
            "model_name": "fasterrcnn_mobilenet_v3_large_fpn",
            "pretrained": True,
            "num_classes": NUM_CLASSES,
            "min_size": MAX_SIZE,
            "max_size": MAX_SIZE,
        }
    }).to(DEV)
    ckpt = torch.load(CKPT_PATH, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded checkpoint epoch={ckpt.get('epoch', '?')} AP50={ckpt.get('ap50', '?')}")

    print("Loading val data...")
    coco = json.loads(ANNOT.read_text())
    all_ids = list(
        set(
            img["id"]
            for img in coco["images"]
            if (DATA / "positive image set" / img["file_name"]).exists()
        )
    )
    np.random.seed(42)
    np.random.shuffle(all_ids)
    n_train = int(0.7 * len(all_ids))
    val_ids = set(all_ids[n_train:])
    print(f"  Val: {len(val_ids)}")

    val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    print("Collecting proposals (IoU>0.1)...")
    records = collect_proposals(model, val_loader, DEV)
    print(f"  -> {len(records)} images with proposals")

    # Flatten all proposals for stats
    total_props = sum(len(r["iou"]) for r in records)
    total_tp_01 = sum(((r["iou"] > 0.1).sum().item()) for r in records)
    total_tp_05 = sum(((r["iou"] > 0.5).sum().item()) for r in records)
    print(f"  Total proposals: {total_props}")
    print(f"  IoU>0.1: {total_tp_01}  IoU>0.5: {total_tp_05}")

    # Group definitions
    groups = {
        "UltraRelaxed": (lambda i: i > 0.1, lambda i: i < 0.1),
        "Standard": (lambda i: i > 0.5, lambda i: i < 0.3),
    }

    results = {}
    for name, (tp_fn, fp_fn) in groups.items():
        print(f"\nAnalyzing {name}...")
        r = analyze_group(records, tp_fn, fp_fn, scaler, pca, k)
        if r is None:
            print(f"  -> insufficient samples")
            continue
        results[name] = r
        print(f"  n={r['n']} (TP={r['n_tp']} FP={r['n_fp']})")
        print(f"  AUC(conf)      = {r['auc_conf']:.4f}")
        print(f"  AUC(conf+manif)= {r['auc_comb']:.4f}")
        print(f"  Delta AUC      = {r['delta_auc']:+.4f} ({r['delta_auc']*100:+.2f}%)")
        print(f"  Pair consistency = {r['pair_consistency']*100:.1f}% ({r['pair_total']} pairs)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"{name}:")
        print(f"  Delta AUC        = {r['delta_auc']*100:+.2f}%")
        print(f"  Pair consistency = {r['pair_consistency']*100:.1f}%")
    print("=" * 60)

    # Save results
    out_path = Path("runs/round2114_nwpu_manifold_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
