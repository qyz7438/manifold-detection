"""Plan 2.113: NWPU Manifold DPO — amp_lo manifold distance-based pair selection.

Adapts round2112 manifold DPO pipeline for NWPU (800 img, 10 cls, multi-class).
Key differences from round2112:
  - NWPU 11-class (10 fg + bg) with MAX_SIZE=480
  - Uses NWPUDataset + COCO annotation (from round2109)
  - DPO score = fg_max (max foreground logit) for multi-class
  - IoU computed with class-agnostic deltas (same as round2109)
  - Calibration pass runs on full NWPU training set
"""
import copy, json, pickle, shutil, sys
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.experiments.runner_utils import decode_boxes, unfreeze_rlvr
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
SEEDS = [42, 123, 456]
EPOCHS = 8
BATCH, MAX_SIZE, NUM_CLASSES = 2, 480, 11
DPO_WEIGHT, KL_WEIGHT, BETA = 0.1, 0.01, 5.0

DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
CKPT_PATH = "runs/round2100_nwpu_baseline/checkpoint_best.pth"
CALIB_PKL = Path("runs/round2113_nwpu_manifold_calib.pkl")


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

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]; info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists(): img_path = self.root / "negative image set" / info["file_name"]
        img = Image.open(str(img_path)).convert("RGB"); img_t = TF.to_tensor(img)
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h]); labels.append(ann["category_id"])
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([img_id])}
        _, H, W = img_t.shape
        if max(H, W) > self.max_size:
            scale = self.max_size / max(H, W); nh, nw = int(H * scale), int(W * scale)
            img_t = F.interpolate(img_t.unsqueeze(0), size=(nh, nw), mode="bilinear").squeeze(0)
            target["boxes"] = target["boxes"] * scale
        return img_t, target


def collate(batch): return tuple(zip(*batch))


def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD([{"params": body, "lr": 0.0001}, {"params": head, "lr": 0.001}],
                           lr=0.001, momentum=0.9, weight_decay=0.0005)


def compute_iou_and_gt(sp_raw, bf, box_predictor, tgts_t):
    """Multi-class per-proposal max IoU + matched GT index."""
    sp_cat = torch.cat(sp_raw, dim=0); N = sp_cat.shape[0]
    with torch.no_grad():
        reg = box_predictor.bbox_pred(bf[:N])
        # Class-agnostic: use class 1 deltas (airplane — rough but consistent)
        deltas = reg[:, 4:8]
        decoded = decode_boxes(sp_cat, deltas)
    iou = torch.zeros(N, device=DEV); gt_idx = torch.zeros(N, dtype=torch.long, device=DEV)
    off = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p == 0: continue
        gt = tgts_t[i_img]["boxes"]
        if len(gt) > 0:
            i = box_iou(decoded[off:off+n_p], gt)
            iou[off:off+n_p], gt_idx[off:off+n_p] = i.max(dim=1)
        off += n_p
    return iou, gt_idx


def extract_amp_lo_perchan_stats(roi_features):
    """Extract amp_lo per-channel stats from ROI features (before box_head).

    Args:
        roi_features: (N, C, H, W) tensor, e.g. (N, 256, 7, 7)

    Returns:
        (N, C*3) numpy array: per-channel [mean, std, max] over 28 freq bins.
    """
    fft = torch.fft.rfft2(roi_features, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)  # (N, C, Hf, Wf)

    freq_h = torch.fft.fftfreq(roi_features.shape[-2], device=roi_features.device)
    freq_w = torch.fft.rfftfreq(roi_features.shape[-1], device=roi_features.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = radius / radius.max().clamp_min(1e-6)

    lo_mask = (radius <= 0.3).float().unsqueeze(0).unsqueeze(0)  # (1, 1, Hf, Wf)
    amp_lo = amp * lo_mask  # (N, C, Hf, Wf)

    flat = amp_lo.reshape(amp_lo.shape[0], amp_lo.shape[1], -1)  # (N, C, 28)
    mu = flat.mean(dim=-1)      # (N, C)
    sg = flat.std(dim=-1)       # (N, C)
    mx = flat.max(dim=-1).values  # (N, C)
    return torch.cat([mu, sg, mx], dim=1).detach().cpu().numpy()  # (N, C*3)


@torch.no_grad()
def collect_calibration_features(model, loader, device):
    """Run one frozen pass over the training set and collect all TP amp_lo features."""
    model.eval()
    all_feats, all_ious, all_gtidx = [], [], []

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))

    for imgs, tgts in tqdm(loader, desc="calib", leave=False):
        imgs_d = [i.to(device) for i in imgs]
        tgts_t = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        sampled_props.clear(); box_head_in.clear()
        _ = model(imgs_d, tgts_t)
        rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
        if rf is None or sp_raw is None or rf.shape[0] == 0:
            continue
        bf = model.roi_heads.box_head(rf)
        bp = model.roi_heads.box_predictor
        iou_p, gt_idx = compute_iou_and_gt(sp_raw, bf, bp, tgts_t)
        feats = extract_amp_lo_perchan_stats(rf)
        all_feats.append(feats)
        all_ious.append(iou_p.cpu().numpy())
        all_gtidx.append(gt_idx.cpu().numpy())

    if not all_feats:
        return None, None, None
    feats = np.concatenate(all_feats, axis=0)
    ious = np.concatenate(all_ious, axis=0)
    gtidx = np.concatenate(all_gtidx, axis=0)
    return feats, ious, gtidx


def build_or_load_calibration():
    """Build global calibration (StandardScaler + PCA + tp_median) from full NWPU training set.

    If the pickle already exists, load it. Otherwise run a frozen baseline pass,
    collect all TP amp_lo features, fit scaler+PCA(50, whiten=True), compute
    the whitened TP median, and save.
    """
    if CALIB_PKL.exists():
        with open(CALIB_PKL, "rb") as f:
            calib = pickle.load(f)
        print(f"[calib] Loaded from {CALIB_PKL}")
        return calib["scaler"], calib["pca"], calib["tp_median"]

    print("[calib] Running frozen baseline pass to collect global TP features...")
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
    for p in model.parameters():
        p.requires_grad = False

    # Build NWPU train loader
    coco = json.loads(ANNOT.read_text())
    all_ids = list(set(img["id"] for img in coco["images"] if Path(DATA / "positive image set" / img["file_name"]).exists()))
    np.random.seed(42); np.random.shuffle(all_ids)
    nt = int(0.7 * len(all_ids))
    train_ids = set(all_ids[:nt])
    tl = DataLoader(NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)

    feats, ious, gtidx = collect_calibration_features(model, tl, DEV)
    if feats is None:
        raise RuntimeError("Calibration failed: no features collected")

    tp_mask = ious > 0.5
    n_tp = int(tp_mask.sum())
    print(f"[calib] Total proposals: {len(feats)}, TP (IoU>0.5): {n_tp}")
    if n_tp < 10:
        raise RuntimeError(f"Calibration failed: only {n_tp} TP samples, need >=10")

    tp_feats = feats[tp_mask]
    scaler = StandardScaler()
    tp_scaled = scaler.fit_transform(tp_feats)
    k = min(50, tp_scaled.shape[0] - 1, tp_scaled.shape[1])
    pca = PCA(n_components=k, whiten=True, random_state=42)
    pca.fit(tp_scaled)
    tp_whitened = pca.transform(tp_scaled)
    tp_median = np.median(tp_whitened, axis=0)

    calib = {"scaler": scaler, "pca": pca, "tp_median": tp_median, "n_tp": n_tp, "k": k}
    CALIB_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PKL, "wb") as f:
        pickle.dump(calib, f)
    print(f"[calib] Saved to {CALIB_PKL} (k={k}, n_tp={n_tp})")
    return scaler, pca, tp_median


def run_one(cfg_name, mode, seed, scaler, pca, tp_median):
    run_name = f"round2113_{cfg_name}_s{seed}"; set_seed(seed)
    coco = json.loads(ANNOT.read_text())
    all_ids = list(set(img["id"] for img in coco["images"] if Path(DATA / "positive image set" / img["file_name"]).exists()))
    np.random.seed(42); np.random.shuffle(all_ids)
    nt = int(0.7 * len(all_ids))
    train_ids, val_ids = set(all_ids[:nt]), set(all_ids[nt:])
    tl = DataLoader(NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    vl = DataLoader(NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

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
    unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))

    opt = build_opt(model); baseline_bp = baseline_model.roi_heads.box_predictor
    run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only"; is_dpo = mode == "dpo"
    h, best_ap50 = [], -1.0
    diag = {"dpo_loss": [], "conf_shift": [], "n_pairs": []}

    for ep in range(1, EPOCHS + 1):
        model.train(); td, tdpo, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}", leave=False):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear()
            ld = model(imgs_d, tgts_t); det = sum(ld.values())
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
            dpo = torch.tensor(0.0, device=DEV); kl = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf); cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                # Multi-class: use max foreground logit as "confidence"
                fg_max, _ = cls_logits[:, 1:].max(dim=-1)  # (N,)

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_logits = baseline_bp.cls_score(baseline_bf)
                    b_fg_max, _ = baseline_logits[:, 1:].max(dim=-1)

                # IoU + GT matching (using baseline for stable matching)
                iou_p, gt_idx = compute_iou_and_gt(sp_raw, baseline_bf, baseline_bp, tgts_t)
                N = min(cls_logits.shape[0], iou_p.shape[0])

                # ---- Manifold distance-based pair selection ----
                # Extract amp_lo per-channel stats (N, C*3)
                amp_lo_feats = extract_amp_lo_perchan_stats(rf[:N])  # numpy (N, C*3)

                # Whiten all features and compute distance to TP median
                if scaler is not None and pca is not None:
                    all_w = pca.transform(scaler.transform(amp_lo_feats))  # (N, k)
                    dists = np.linalg.norm(all_w - tp_median, axis=1)  # (N,)
                else:
                    dists = np.zeros(N)

                # DPO: pair chosen=closest to TP, rejected=farthest from TP within each GT group
                n_pairs = 0
                for gid in torch.unique(gt_idx[:N]):
                    if gid < 0: continue
                    mask = gt_idx[:N] == gid
                    if mask.sum() < 2: continue
                    mask_np = mask.cpu().numpy()
                    group_dists = dists[mask_np]
                    group_logits = fg_max[:N][mask]
                    group_ref = b_fg_max[:N][mask]

                    # closest to TP median = chosen (better), farthest = rejected (worse)
                    best_idx = int(group_dists.argmin())
                    worst_idx = int(group_dists.argmax())
                    if best_idx == worst_idx: continue
                    lc, lr = group_logits[best_idx], group_logits[worst_idx]
                    rc, rr = group_ref[best_idx], group_ref[worst_idx]
                    margin = (lc - rc) - (lr - rr)
                    dpo = dpo - F.logsigmoid(BETA * margin)
                    n_pairs += 1

                if n_pairs > 0:
                    dpo = dpo / n_pairs
                diag["dpo_loss"].append(dpo.item()); diag["n_pairs"].append(n_pairs)
                kl = KL_WEIGHT * (fg_max[:N] - b_fg_max[:N]).pow(2).mean()
                diag["conf_shift"].append((fg_max[:N].mean() - b_fg_max[:N].mean()).item())

            loss = det + DPO_WEIGHT * dpo + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item(); tdpo += dpo.item(); tkl += kl.item()

        model.eval(); ps, ts = [], []
        for img, tgt in vl:
            with torch.no_grad():
                pred = model([img[0].to(DEV)])[0]
            ps.append({k: v.cpu() for k, v in pred.items()})
            ts.append({k: v.cpu() for k, v in tgt[0].items()})
        em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)
        np_m = np.mean(diag["n_pairs"]) if diag["n_pairs"] else 0
        dp_m = np.mean(diag["dpo_loss"]) if diag["dpo_loss"] else 0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "n_pairs": float(np_m), "dpo_loss": float(dp_m),
               "conf_shift": float(cs_m), "det": td, "dpo": tdpo, "kl": tkl}
        h.append(row); print(f"  e{ep}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f} pairs={np_m:.1f} dpo={dp_m:.3f}")
        if em["ap50"] > best_ap50: best_ap50 = em["ap50"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap50"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_ap50, "best_ap75": best_h["val_ap75"], "history": h})
    save_json(em, run_dir / "eval_metrics.json"); return em


if __name__ == "__main__":
    # Build/load global calibration once before any training
    scaler, pca, tp_median = build_or_load_calibration()

    results = []
    for cfg, mode in [("det_only", "det_only"), ("manifold_dpo", "dpo")]:
        for seed in SEEDS:
            results.append(run_one(cfg, mode, seed, scaler, pca, tp_median))
    print("\n## 2.113 NWPU Manifold-DPO (amp_lo distance-based pair selection)")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap50"])
        print(f"  {r['config']:<15s} s{r['seed']} AP50={r['best_ap50']:.4f} AP75={bh['val_ap75']:.4f}")
    det = [r for r in results if r["config"] == "det_only"]
    dpo = [r for r in results if r["config"] == "manifold_dpo"]
    det_ap50 = [r["best_ap50"] for r in det]
    dpo_ap50 = [r["best_ap50"] for r in dpo]
    det_mean = np.mean(det_ap50); det_std = np.std(det_ap50, ddof=1)
    dpo_mean = np.mean(dpo_ap50); dpo_std = np.std(dpo_ap50, ddof=1)
    print(f"\n  det_only:    bestAP50 = {det_mean:.4f} ± {det_std:.4f}")
    print(f"  manifold_dpo: bestAP50 = {dpo_mean:.4f} ± {dpo_std:.4f}")
    print(f"  Delta: AP50 = {dpo_mean - det_mean:+.4f}")
    print(f"\n  Reference: round2109 NWPU DPO best AP50 = 0.654 (baseline)")
