"""Plan 2.113: DPO with percentile-based pair selection.

For each GT with >=2 matching proposals:
  - Extract amp_lo per-channel stats (mean/std/max over 28 freq bins per channel) -> 768 dim
  - StandardScaler + PCA whitening -> 50 dim
  - Measure whitened Euclidean distance to TP cluster median
  - 25th percentile = chosen, 75th percentile = rejected (robust to outliers)

DPO loss: same as round2112 (pairwise best-vs-worst contrastive on logit margins).
"""
import copy, pickle, shutil, sys
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320, decode_boxes, evaluate_model, unfreeze_rlvr,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"; CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SEED, EPOCHS = 42, 8
DPO_WEIGHT, KL_WEIGHT, BETA = 0.1, 0.03, 2.0

CALIB_PKL = Path("runs/round2112_manifold_calib.pkl")


def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})


def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD([{"params": body, "lr": 0.0001}, {"params": head, "lr": 0.001}],
                           lr=0.001, momentum=0.9, weight_decay=0.0005)


def compute_iou_and_gt(sp_raw, bf, box_predictor, tgts_t):
    """Per-proposal max IoU + which GT it matches. Returns (iou, gt_idx)."""
    sp_cat = torch.cat(sp_raw, dim=0); N = sp_cat.shape[0]
    with torch.no_grad():
        reg = box_predictor.bbox_pred(bf[:N]); decoded = decode_boxes(sp_cat, reg[:, 2:6])
    iou = torch.zeros(N, device=DEV); gt_idx = torch.full((N,), -1, dtype=torch.long, device=DEV)
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
    """Build global calibration (StandardScaler + PCA + tp_median) from full training set.

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
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad = False

    tl, _ = build_penn_fudan_loaders_320(batch_size=2)
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
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))

    tl, vl = build_penn_fudan_loaders_320(batch_size=2)
    run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only_unf"; is_dpo = mode == "dpo"
    opt = build_opt(model); baseline_bp = baseline_model.roi_heads.box_predictor
    h, best_ap75 = [], -1.0
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
                person_logit = cls_logits[:, 1]  # (N,)

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_logits = baseline_bp.cls_score(baseline_bf)
                    baseline_person = baseline_logits[:, 1]

                # IoU + GT matching
                iou_p, gt_idx = compute_iou_and_gt(sp_raw, baseline_bf, baseline_bp, tgts_t)
                N = min(cls_logits.shape[0], iou_p.shape[0])

                # ---- Manifold distance-based pair selection ----
                # Extract amp_lo per-channel stats (N, 768)
                amp_lo_feats = extract_amp_lo_perchan_stats(rf[:N])  # numpy (N, 768)

                # Whiten all features and compute distance to TP median
                if scaler is not None and pca is not None:
                    all_w = pca.transform(scaler.transform(amp_lo_feats))  # (N, k)
                    dists = np.linalg.norm(all_w - tp_median, axis=1)  # (N,)
                else:
                    dists = np.zeros(N)

                # DPO: pair chosen=25th percentile, rejected=75th percentile within each GT group
                n_pairs = 0
                for gid in torch.unique(gt_idx[:N]):
                    if gid < 0: continue
                    mask = gt_idx[:N] == gid
                    if mask.sum() < 2: continue
                    mask_np = mask.cpu().numpy()
                    group_dists = dists[mask_np]
                    group_logits = person_logit[:N][mask]
                    group_ref = baseline_person[:N][mask]

                    # Percentile-based selection: 25th = chosen (better), 75th = rejected (worse)
                    sorted_idx = np.argsort(group_dists)
                    n = len(sorted_idx)
                    best_idx = sorted_idx[min(n // 4, n - 1)]                    # 25th percentile
                    worst_idx = sorted_idx[min(3 * n // 4, n - 1)]                 # 75th percentile
                    if best_idx == worst_idx: continue
                    lc, lr = group_logits[best_idx], group_logits[worst_idx]
                    rc, rr = group_ref[best_idx], group_ref[worst_idx]
                    margin = (lc - rc) - (lr - rr)
                    dpo = dpo - F.logsigmoid(BETA * margin)
                    n_pairs += 1

                if n_pairs > 0:
                    dpo = dpo / n_pairs
                diag["dpo_loss"].append(dpo.item()); diag["n_pairs"].append(n_pairs)
                kl = KL_WEIGHT * (person_logit[:N] - baseline_person[:N]).pow(2).mean()
                diag["conf_shift"].append((person_logit[:N].mean() - baseline_person[:N].mean()).item())

            loss = det + DPO_WEIGHT * dpo + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item(); tdpo += dpo.item(); tkl += kl.item()

        em = evaluate_model(model, vl, DEV)
        np_m = np.mean(diag["n_pairs"]) if diag["n_pairs"] else 0
        dp_m = np.mean(diag["dpo_loss"]) if diag["dpo_loss"] else 0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "n_pairs": float(np_m), "dpo_loss": float(dp_m),
               "conf_shift": float(cs_m), "det": td, "dpo": tdpo, "kl": tkl}
        h.append(row); print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} pairs={np_m:.1f} dpo={dp_m:.3f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75, "history": h})
    save_json(em, run_dir / "eval_metrics.json"); return em


if __name__ == "__main__":
    # Build/load global calibration once before any training
    scaler, pca, tp_median = build_or_load_calibration()

    results = []
    for cfg, mode in [("det_only", "det_only_unf"), ("percentile_dpo", "dpo")]:
        results.append(run_one(cfg, mode, SEED, scaler, pca, tp_median))
    print("\n## 2.113 Percentile-DPO (25th/75th percentile pair selection)")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<15s} s{r['seed']} AP75={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    print(f"\n  Reference: round2112 Manifold-DPO best AP75 = 0.732")
    print(f"  Reference: round2108 IoU DPO best AP75 = 0.724")
