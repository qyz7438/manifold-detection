"""验证：放松 IoU 阈值后，流形几何能否在 confidence 不可靠的 borderline 区域提供独立区分力。

Plan: 2.113-audit
验证方式（GPT-5.5 审计确认）：
  - TP = IoU[0.3, 0.5), FP = IoU[0.1, 0.3)
  - 只用 confidence + FFT 做特征（不用 IoU）
  - 按 image split（同一 image 内 proposal 配对）

在 NWPU val 集上：
  1. 加载 round2100 baseline checkpoint
  2. 提取所有 proposal（IoU>0.1 全部保留）
  3. 定义三组：
       Standard:   TP>0.5, FP<0.3
       Relaxed:    TP>0.3, FP<0.1
       Borderline: TP=IoU[0.3,0.5), FP=IoU[0.1,0.3)
  4. 每组只用 confidence 做 baseline 分类器
  5. 对比 confidence vs confidence+FFT(manifold) 的区分力

评估指标：
  - AUC: confidence only vs confidence + manifold distance
  - Partial AUC at low FPR (0-20%)
  - 在 confidence-matched 子集上的 TPR 差异

输出：清楚报告每个分组下 manifold 的增量贡献。
如果 Borderline 组有显著增量（ΔAUC>0.02），说明“几何信号在分类器不确定区域有价值”成立。
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import decode_boxes
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

warnings.filterwarnings("ignore", category=UserWarning)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_SIZE = 480
NUM_CLASSES = 11
DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
CKPT_PATH = Path("runs/round2100_nwpu_baseline/checkpoint_best.pth")
RUN_DIR = Path("runs/round2113_audit_manifold_verify")

set_seed(SEED)

# ---------------------------------------------------------------------------
# NWPU Dataset
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
            nh, nw = int(H * scale), int(W * scale)
            img_t = F.interpolate(img_t.unsqueeze(0), size=(nh, nw), mode="bilinear").squeeze(0)
            target["boxes"] = target["boxes"] * scale
        return img_t, target


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model + Feature extraction
# ---------------------------------------------------------------------------


def build_model():
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
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_amp_lo_perchan_stats(roi_features):
    """amp_lo per-channel stats: (N, C, H, W) -> (N, C*3) [mean, std, max]."""
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


@torch.no_grad()
def collect_proposals(model, val_loader):
    """Collect per-proposal: IoU, confidence, amp_lo features, GT index, image id.

    Returns list of dicts per image:
      - iou: (N,)
      - conf: (N,)
      - feat: (N, D)  amp_lo per-channel stats
      - gt_idx: (N,)
      - img_id: int
      - boxes: (N, 4)
    """
    records = []
    sampled_props, box_head_in = {}, {}

    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    for images, targets in tqdm(val_loader, desc="collect proposals"):
        images_d = [img.to(DEV) for img in images]
        tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
        sampled_props.clear()
        box_head_in.clear()

        model(images_d, tgts_t)

        sp_raw = sampled_props.get("p")
        rf = box_head_in.get("x")
        if sp_raw is None or rf is None or rf.shape[0] == 0:
            continue

        # confidence: max foreground logit (multi-class)
        bf = model.roi_heads.box_head(rf)
        cls_logits = model.roi_heads.box_predictor.cls_score(bf)
        conf = F.softmax(cls_logits, dim=-1)[:, 1:].max(dim=-1).values.cpu()

        sp_cat = torch.cat(sp_raw, dim=0)
        N = sp_cat.shape[0]
        reg_out = model.roi_heads.box_predictor.bbox_pred(bf[:N])
        # class-agnostic deltas: use class 1 (airplane) as in round2113
        deltas = reg_out[:, 4:8]
        decoded = decode_boxes(sp_cat, deltas)

        # FFT features
        feats = extract_amp_lo_perchan_stats(rf[:N])

        offset = 0
        for i_img, p_img in enumerate(sp_raw):
            n_p = p_img.shape[0]
            if n_p == 0:
                continue
            gt = tgts_t[i_img]["boxes"]
            img_id = int(tgts_t[i_img]["image_id"][0].item())

            iou = torch.zeros(n_p)
            gt_idx = torch.full((n_p,), -1, dtype=torch.long)
            if len(gt) > 0:
                iou_mat = box_iou(decoded[offset:offset + n_p], gt)
                iou, best_gt = iou_mat.max(dim=1)
                iou = iou.cpu()
                gt_idx = best_gt.cpu()

            records.append({
                "iou": iou.numpy(),
                "conf": conf[offset:offset + n_p].numpy(),
                "feat": feats[offset:offset + n_p],
                "gt_idx": gt_idx.numpy(),
                "img_id": img_id,
                "boxes": sp_cat[offset:offset + n_p].cpu().numpy(),
            })
            offset += n_p

    return records


# ---------------------------------------------------------------------------
# Grouping & Pair building (within same image, same GT)
# ---------------------------------------------------------------------------


def build_all_proposals(records):
    """Flatten all proposals into a single array with group keys."""
    all_props = []
    for rec in records:
        img_id = rec["img_id"]
        for i in range(len(rec["iou"])):
            all_props.append({
                "img_id": img_id,
                "gt_idx": int(rec["gt_idx"][i]),
                "iou": float(rec["iou"][i]),
                "conf": float(rec["conf"][i]),
                "feat": rec["feat"][i],
            })
    return all_props


def group_by_image_gt(props):
    """Group proposals by (img_id, gt_idx)."""
    groups = defaultdict(list)
    for p in props:
        if p["gt_idx"] >= 0:
            groups[(p["img_id"], p["gt_idx"])].append(p)
    return groups


# ---------------------------------------------------------------------------
# Calibration: fit scaler + PCA on TP (IoU>0.5) from training set
# ---------------------------------------------------------------------------


def build_calibration(train_records):
    """Fit StandardScaler + PCA(whiten) on TP features from training records."""
    tp_feats = []
    for rec in train_records:
        mask = rec["iou"] > 0.5
        if mask.any():
            tp_feats.append(rec["feat"][mask])
    if not tp_feats:
        raise RuntimeError("No TP features for calibration")
    X = np.concatenate(tp_feats, axis=0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(50, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=k, whiten=True, random_state=42)
    pca.fit(Xs)
    tp_whitened = pca.transform(Xs)
    tp_median = np.median(tp_whitened, axis=0)
    return scaler, pca, tp_median, k


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_pauc(y_true, y_score, fpr_range=(0.0, 0.2)):
    """Partial AUC in a specific FPR range."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find indices within range
    idx = (fpr >= fpr_range[0]) & (fpr <= fpr_range[1])
    if idx.sum() < 2:
        return 0.0
    fpr_sub = fpr[idx]
    tpr_sub = tpr[idx]
    # Add boundaries
    if fpr_sub[0] > fpr_range[0]:
        fpr_sub = np.concatenate([[fpr_range[0]], fpr_sub])
        tpr_sub = np.concatenate([[tpr_sub[0]], tpr_sub])
    if fpr_sub[-1] < fpr_range[1]:
        fpr_sub = np.concatenate([fpr_sub, [fpr_range[1]]])
        tpr_sub = np.concatenate([tpr_sub, [tpr_sub[-1]]])
    return auc(fpr_sub, tpr_sub)


def evaluate_group(props, scaler, pca, tp_median, label_fn):
    """Evaluate a group of proposals with a given label function.

    label_fn(p) -> 1 (TP-like) or 0 (FP-like)

    Returns dict with metrics for:
      - confidence only
      - manifold distance only
      - confidence + manifold (concatenated as 2D feature for logistic-style scoring)
    """
    # Filter to proposals with valid labels
    labeled = [(p, label_fn(p)) for p in props]
    labeled = [(p, lbl) for p, lbl in labeled if lbl is not None]
    if len(labeled) < 10:
        return None

    feats = np.stack([p["feat"] for p, _ in labeled], axis=0)
    confs = np.array([p["conf"] for p, _ in labeled], dtype=np.float64)
    y_true = np.array([lbl for _, lbl in labeled], dtype=np.int32)

    # Manifold distance to TP median
    if scaler is not None and pca is not None:
        fw = pca.transform(scaler.transform(feats))
        dists = np.linalg.norm(fw - tp_median, axis=1)
        # Convert distance to score: closer = higher score = more TP-like
        manifold_score = -dists
    else:
        manifold_score = np.zeros(len(feats))

    # Confidence-only score
    conf_score = confs

    # Combined score: simple weighted sum (treat as 2D feature, use logistic-style)
    # Normalize both to [0, 1] roughly, then sum
    def normalize_score(s):
        s_min, s_max = s.min(), s.max()
        if s_max - s_min < 1e-8:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)

    combined_score = normalize_score(conf_score) + normalize_score(manifold_score)

    results = {}
    for name, score in [
        ("conf_only", conf_score),
        ("manifold_only", manifold_score),
        ("conf_plus_manifold", combined_score),
    ]:
        try:
            auc_full = roc_auc_score(y_true, score)
        except ValueError:
            auc_full = 0.5
        pauc = compute_pauc(y_true, score, fpr_range=(0.0, 0.2))
        results[name] = {"auc": float(auc_full), "pauc_0_20": float(pauc)}

    # Confidence-matched TPR analysis: stratify by confidence deciles, compare TPR
    # For each decile where conf is similar, compare manifold score's ability to rank TP vs FP
    tpr_diffs = []
    n_bins = 5
    conf_bins = np.percentile(confs, np.linspace(0, 100, n_bins + 1))
    for i in range(n_bins):
        lo, hi = conf_bins[i], conf_bins[i + 1]
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if mask.sum() < 5:
            continue
        y_sub = y_true[mask]
        s_sub = manifold_score[mask]
        # TPR at 50% threshold of score within this bin
        thr = np.median(s_sub)
        tp = ((s_sub >= thr) & (y_sub == 1)).sum()
        fn = ((s_sub < thr) & (y_sub == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # For conf-only, same threshold approach (median)
        c_sub = conf_score[mask]
        thr_c = np.median(c_sub)
        tp_c = ((c_sub >= thr_c) & (y_sub == 1)).sum()
        fn_c = ((c_sub < thr_c) & (y_sub == 1)).sum()
        tpr_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        tpr_diffs.append(tpr - tpr_c)

    results["tpr_diff_conf_matched"] = {
        "mean": float(np.mean(tpr_diffs)) if tpr_diffs else 0.0,
        "std": float(np.std(tpr_diffs, ddof=1)) if len(tpr_diffs) > 1 else 0.0,
        "n_bins": len(tpr_diffs),
    }

    results["n_samples"] = len(labeled)
    results["n_positive"] = int(y_true.sum())
    return results


# ---------------------------------------------------------------------------
# Label functions for the three groups
# ---------------------------------------------------------------------------


def label_standard(p):
    """Standard: TP>0.5, FP<0.3."""
    if p["iou"] > 0.5:
        return 1
    if p["iou"] < 0.3:
        return 0
    return None


def label_relaxed(p):
    """Relaxed: TP>0.3, FP<0.1."""
    if p["iou"] > 0.3:
        return 1
    if p["iou"] < 0.1:
        return 0
    return None


def label_borderline(p):
    """Borderline: TP in [0.3, 0.5), FP in [0.1, 0.3)."""
    if 0.3 <= p["iou"] < 0.5:
        return 1
    if 0.1 <= p["iou"] < 0.3:
        return 0
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print("[1/6] Loading round2100 baseline model...")
    model = build_model()

    # 2. Build NWPU val loader (same split as round2100)
    print("[2/6] Building NWPU val loader...")
    coco = json.loads(ANNOT.read_text())
    all_ids = list(
        set(
            img["id"]
            for img in coco["images"]
            if Path(DATA / "positive image set" / img["file_name"]).exists()
        )
    )
    np.random.seed(42)
    np.random.shuffle(all_ids)
    n_train = int(0.7 * len(all_ids))
    train_ids = set(all_ids[:n_train])
    val_ids = set(all_ids[n_train:])
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")

    val_loader = DataLoader(
        NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE),
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # 3. Collect proposals on val set (IoU>0.1 all retained)
    print("[3/6] Collecting proposals on val set...")
    val_records = collect_proposals(model, val_loader)
    print(f"  -> {len(val_records)} images with proposals")

    # Also collect on train set for calibration
    print("  Collecting proposals on train set for calibration...")
    train_loader = DataLoader(
        NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE),
        batch_size=2,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    train_records = collect_proposals(model, train_loader)
    print(f"  -> {len(train_records)} images with proposals")

    # 4. Build calibration on training set TP (IoU>0.5)
    print("[4/6] Building calibration (scaler + PCA) on train TP...")
    scaler, pca, tp_median, k = build_calibration(train_records)
    print(f"  -> PCA k={k}, TP median shape={tp_median.shape}")

    # 5. Flatten and group val proposals
    print("[5/6] Grouping val proposals by (image, GT)...")
    all_props = build_all_proposals(val_records)
    groups = group_by_image_gt(all_props)
    print(f"  -> {len(all_props)} total proposals, {len(groups)} (img, GT) groups")

    # 6. Evaluate per group
    print("[6/6] Evaluating AUC / pAUC / TPR-diff per group...")
    print()

    group_results = {}
    for group_name, label_fn in [
        ("Standard", label_standard),
        ("Relaxed", label_relaxed),
        ("Borderline", label_borderline),
    ]:
        # Collect all proposals across all groups that satisfy the label function
        eligible = []
        for key, gprops in groups.items():
            for p in gprops:
                lbl = label_fn(p)
                if lbl is not None:
                    eligible.append(p)

        if len(eligible) < 10:
            print(f"  {group_name}: insufficient samples ({len(eligible)}), skipping")
            continue

        r = evaluate_group(eligible, scaler, pca, tp_median, label_fn)
        group_results[group_name] = r

        print(f"  === {group_name} ===  n={r['n_samples']} pos={r['n_positive']}")
        for score_name in ["conf_only", "manifold_only", "conf_plus_manifold"]:
            s = r[score_name]
            print(f"    {score_name:<20s} AUC={s['auc']:.4f}  pAUC(0-20%)={s['pauc_0_20']:.4f}")
        delta_auc = r["conf_plus_manifold"]["auc"] - r["conf_only"]["auc"]
        delta_pauc = r["conf_plus_manifold"]["pauc_0_20"] - r["conf_only"]["pauc_0_20"]
        print(f"    ΔAUC (conf+manifold vs conf) = {delta_auc:+.4f}")
        print(f"    ΔpAUC (conf+manifold vs conf) = {delta_pauc:+.4f}")
        tprd = r["tpr_diff_conf_matched"]
        print(f"    TPR diff (conf-matched bins) = {tprd['mean']:+.4f} ± {tprd['std']:.4f} (n_bins={tprd['n_bins']})")
        print()

    # 7. Summary report
    print("=" * 70)
    print("SUMMARY: Manifold Incremental Contribution by Group")
    print("=" * 70)
    for group_name in ["Standard", "Relaxed", "Borderline"]:
        r = group_results.get(group_name)
        if r is None:
            continue
        delta_auc = r["conf_plus_manifold"]["auc"] - r["conf_only"]["auc"]
        delta_pauc = r["conf_plus_manifold"]["pauc_0_20"] - r["conf_only"]["pauc_0_20"]
        tprd = r["tpr_diff_conf_matched"]
        print(f"\n{group_name}:")
        print(f"  ΔAUC        = {delta_auc:+.4f}")
        print(f"  ΔpAUC(0-20%)= {delta_pauc:+.4f}")
        print(f"  TPR diff    = {tprd['mean']:+.4f} ± {tprd['std']:.4f}")
        if group_name == "Borderline":
            verdict = "SIGNIFICANT" if delta_auc > 0.02 else "MARGINAL" if delta_auc > 0.01 else "NONE"
            print(f"  Verdict     = {verdict}  (threshold ΔAUC>0.02 for 'geometry signal valuable')")

    # Save results
    out_path = RUN_DIR / "manifold_verify_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(group_results, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n[Done] Results saved to {out_path}")


if __name__ == "__main__":
    main()
