"""Plan 2.113: Manifold-Confidence Disagreement DPO.

For each GT with >=2 matching proposals:
  - Compute normalized confidence score (high -> good)
  - Compute normalized manifold score (-isomap_dist, near -> good)
  - disagreement = abs(conf_score - manifold_score)
  - chosen:  manifold better than confidence (hidden good box)
  - rejected: confidence better than manifold (potential FP)

DPO loss: pairwise contrastive on logit margins.
"""
import copy, json, pickle, shutil, sys
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320, decode_boxes, evaluate_model, unfreeze_rlvr,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SEED, EPOCHS = 42, 8
DPO_WEIGHT, KL_WEIGHT, BETA = 0.1, 0.01, 5.0
DISAGREE_THRESH = 0.3  # min normalized disagreement to form a pair

# Precomputed Isomap(6) embeddings and raw features
EMB_NPZ = Path("scripts/manifold_nonlinear_results/embeddings.npz")
RAW_NPZ = Path("scripts/manifold_fft_results/raw_features.npz")
CALIB_PKL = Path("runs/round2113_manifold_conf_disagree_calib.pkl")


def bm():
    return build_detector({
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
        }
    })


def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD(
        [{"params": body, "lr": 0.0001}, {"params": head, "lr": 0.001}],
        lr=0.001, momentum=0.9, weight_decay=0.0005,
    )


def compute_iou_and_gt(sp_raw, bf, box_predictor, tgts_t):
    """Per-proposal max IoU + which GT it matches. Returns (iou, gt_idx)."""
    sp_cat = torch.cat(sp_raw, dim=0)
    N = sp_cat.shape[0]
    with torch.no_grad():
        reg = box_predictor.bbox_pred(bf[:N])
        decoded = decode_boxes(sp_cat, reg[:, 2:6])
    iou = torch.zeros(N, device=DEV)
    gt_idx = torch.full((N,), -1, dtype=torch.long, device=DEV)
    off = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p == 0:
            continue
        gt = tgts_t[i_img]["boxes"]
        if len(gt) > 0:
            i = box_iou(decoded[off:off + n_p], gt)
            iou[off:off + n_p], gt_idx[off:off + n_p] = i.max(dim=1)
        off += n_p
    return iou, gt_idx


def extract_amp_lo_perchan_stats(roi_features):
    """Extract amp_lo per-channel stats from ROI features (before box_head).

    Args:
        roi_features: (N, C, H, W) tensor, e.g. (N, 256, 7, 7)

    Returns:
        (N, C*3) numpy array: per-channel [mean, std, max] over freq bins.
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


def load_precomputed_isomap():
    """Load precomputed Isomap(6) embeddings and compute per-GT distance stats."""
    npz_emb = np.load(EMB_NPZ)
    X_iso = npz_emb["Isomap"].astype(np.float32)  # (3224, 6)
    ious = npz_emb["ious"].astype(np.float32)     # (3224,)
    confs = npz_emb["confs"].astype(np.float32)   # (3224,)
    gt_ids = npz_emb["gt_ids"].astype(np.int32)   # (3224,)

    # Compute TP centroid in Isomap space
    is_tp = ious >= 0.5
    tp_centroid = X_iso[is_tp].mean(axis=0)
    iso_dist = np.linalg.norm(X_iso - tp_centroid, axis=1)

    return X_iso, ious, confs, gt_ids, iso_dist, tp_centroid


def build_or_load_calibration():
    """Build/load calibration: Isomap distance normalization stats."""
    if CALIB_PKL.exists():
        with open(CALIB_PKL, "rb") as f:
            calib = pickle.load(f)
        print(f"[calib] Loaded from {CALIB_PKL}")
        return calib

    print("[calib] Building calibration from precomputed embeddings...")
    X_iso, ious, confs, gt_ids, iso_dist, tp_centroid = load_precomputed_isomap()

    # Global normalization stats for confidence and manifold distance
    conf_min, conf_max = float(confs.min()), float(confs.max())
    dist_min, dist_max = float(iso_dist.min()), float(iso_dist.max())

    calib = {
        "tp_centroid": tp_centroid,
        "conf_min": conf_min,
        "conf_max": conf_max,
        "dist_min": dist_min,
        "dist_max": dist_max,
        "N_total": len(ious),
        "N_tp": int((ious >= 0.5).sum()),
    }
    CALIB_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PKL, "wb") as f:
        pickle.dump(calib, f)
    print(f"[calib] Saved to {CALIB_PKL}")
    return calib


def normalize_confidence(conf, conf_min, conf_max):
    """Normalize confidence to [0, 1]."""
    rng = conf_max - conf_min
    if rng < 1e-6:
        return np.zeros_like(conf)
    return (conf - conf_min) / rng


def normalize_manifold_score(iso_dist, dist_min, dist_max):
    """Normalize -distance so near=1, far=0."""
    rng = dist_max - dist_min
    if rng < 1e-6:
        return np.zeros_like(iso_dist)
    return 1.0 - (iso_dist - dist_min) / rng


@torch.no_grad()
def collect_live_isomap_features(model, loader, device, calib):
    """Run one frozen pass and collect per-proposal Isomap(6) embeddings live.

    Since we don't have a direct mapping from training proposals to precomputed
    embeddings, we recompute Isomap features on the fly from ROI amp_lo stats.
    We use the precomputed PCA + Isomap fitted on the full dataset to transform
    live features.
    """
    # Load precomputed PCA(50) + Isomap(6) fitted on raw features
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from sklearn.preprocessing import StandardScaler

    npz_raw = np.load(RAW_NPZ)
    X_raw_all = npz_raw["X_raw"].astype(np.float32)

    # Fit scaler + PCA on all raw features (same as precomputation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw_all)
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    iso = Isomap(n_neighbors=15, n_components=6)
    iso.fit(X_pca)

    model.eval()
    all_iso, all_ious, all_gtidx, all_confs = [], [], [], []

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    for imgs, tgts in tqdm(loader, desc="live_isomap", leave=False):
        imgs_d = [i.to(device) for i in imgs]
        tgts_t = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        sampled_props.clear()
        box_head_in.clear()
        _ = model(imgs_d, tgts_t)
        rf = box_head_in.get("x")
        sp_raw = sampled_props.get("p")
        if rf is None or sp_raw is None or rf.shape[0] == 0:
            continue

        bf = model.roi_heads.box_head(rf)
        bp = model.roi_heads.box_predictor
        iou_p, gt_idx = compute_iou_and_gt(sp_raw, bf, bp, tgts_t)

        # Extract raw features and transform to Isomap(6)
        raw_feats = extract_amp_lo_perchan_stats(rf)  # (N, 768)
        scaled = scaler.transform(raw_feats)
        pca_proj = pca.transform(scaled)
        iso_proj = iso.transform(pca_proj)  # (N, 6)

        # Compute per-proposal distance to TP centroid
        iso_dist_live = np.linalg.norm(iso_proj - calib["tp_centroid"], axis=1)

        # Get confidence scores
        with torch.no_grad():
            cls_logits = bp.cls_score(bf)
            confs_live = torch.sigmoid(cls_logits[:, 1]).cpu().numpy()

        all_iso.append(iso_dist_live)
        all_ious.append(iou_p.cpu().numpy())
        all_gtidx.append(gt_idx.cpu().numpy())
        all_confs.append(confs_live)

    if not all_iso:
        return None, None, None, None
    return (
        np.concatenate(all_iso),
        np.concatenate(all_ious),
        np.concatenate(all_gtidx),
        np.concatenate(all_confs),
    )


def run_one(cfg_name, mode, seed, calib):
    run_name = f"round2113_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False

    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    tl, vl = build_penn_fudan_loaders_320(batch_size=2)
    run_dir = ensure_run_dir(run_name)
    shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only_unf"
    opt = build_opt(model)
    baseline_bp = baseline_model.roi_heads.box_predictor
    h, best_ap75 = [], -1.0
    diag = {"dpo_loss": [], "conf_shift": [], "n_pairs": [], "disagreement": []}

    # Precompute normalization ranges from calibration
    conf_min = calib["conf_min"]
    conf_max = calib["conf_max"]
    dist_min = calib["dist_min"]
    dist_max = calib["dist_max"]

    for ep in range(1, EPOCHS + 1):
        model.train()
        td, tdpo, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}", leave=False):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear()
            box_head_in.clear()
            ld = model(imgs_d, tgts_t)
            det = sum(ld.values())
            rf = box_head_in.get("x")
            sp_raw = sampled_props.get("p")
            dpo = torch.tensor(0.0, device=DEV)
            kl = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf)
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                person_logit = cls_logits[:, 1]  # (N,)

                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_logits = baseline_bp.cls_score(baseline_bf)
                    baseline_person = baseline_logits[:, 1]

                # IoU + GT matching
                iou_p, gt_idx = compute_iou_and_gt(sp_raw, baseline_bf, baseline_bp, tgts_t)
                N = min(cls_logits.shape[0], iou_p.shape[0])

                # ---- Live Isomap feature extraction for this batch ----
                raw_feats = extract_amp_lo_perchan_stats(rf[:N])  # (N, 768)

                # Transform through pre-fitted scaler+PCA+Isomap
                # We load the fitted transformers once per epoch to avoid overhead
                if not hasattr(run_one, "_scaler"):
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import Isomap
                    from sklearn.preprocessing import StandardScaler
                    npz_raw = np.load(RAW_NPZ)
                    X_raw_all = npz_raw["X_raw"].astype(np.float32)
                    run_one._scaler = StandardScaler()
                    X_scaled = run_one._scaler.fit_transform(X_raw_all)
                    run_one._pca = PCA(n_components=50, random_state=42)
                    X_pca = run_one._pca.fit_transform(X_scaled)
                    run_one._iso = Isomap(n_neighbors=15, n_components=6)
                    run_one._iso.fit(X_pca)

                scaled = run_one._scaler.transform(raw_feats)
                pca_proj = run_one._pca.transform(scaled)
                iso_proj = run_one._iso.transform(pca_proj)  # (N, 6)
                iso_dist = np.linalg.norm(iso_proj - calib["tp_centroid"], axis=1)

                # Get live confidence scores
                with torch.no_grad():
                    confs_live = torch.sigmoid(person_logit[:N]).cpu().numpy()

                # Normalize scores
                conf_score = normalize_confidence(confs_live, conf_min, conf_max)
                manifold_score = normalize_manifold_score(iso_dist, dist_min, dist_max)
                disagreement = np.abs(conf_score - manifold_score)

                # ---- Manifold-Confidence Disagreement DPO pair selection ----
                n_pairs = 0
                for gid in torch.unique(gt_idx[:N]):
                    if gid < 0:
                        continue
                    mask = gt_idx[:N] == gid
                    if mask.sum() < 2:
                        continue

                    mask_np = mask.cpu().numpy()
                    group_disagree = disagreement[mask_np]
                    group_conf = conf_score[mask_np]
                    group_manifold = manifold_score[mask_np]
                    group_logits = person_logit[:N][mask]
                    group_ref = baseline_person[:N][mask]

                    if len(group_disagree) < 2:
                        continue

                    max_disagree = group_disagree.max()
                    if max_disagree < DISAGREE_THRESH:
                        continue  # Not enough disagreement in this group

                    # chosen: manifold better than confidence (hidden good box)
                    # rejected: confidence better than manifold (potential FP)
                    man_minus_conf = group_manifold - group_conf
                    conf_minus_man = group_conf - group_manifold

                    chosen_idx = int(man_minus_conf.argmax())
                    rejected_idx = int(conf_minus_man.argmax())

                    if chosen_idx == rejected_idx:
                        continue

                    lc = group_logits[chosen_idx]
                    lr = group_logits[rejected_idx]
                    rc = group_ref[chosen_idx]
                    rr = group_ref[rejected_idx]
                    margin = (lc - rc) - (lr - rr)
                    dpo = dpo - F.logsigmoid(BETA * margin)
                    n_pairs += 1
                    diag["disagreement"].append(float(group_disagree[chosen_idx]))
                    diag["disagreement"].append(float(group_disagree[rejected_idx]))

                if n_pairs > 0:
                    dpo = dpo / n_pairs
                diag["dpo_loss"].append(dpo.item())
                diag["n_pairs"].append(n_pairs)
                kl = KL_WEIGHT * (person_logit[:N] - baseline_person[:N]).pow(2).mean()
                diag["conf_shift"].append(
                    (person_logit[:N].mean() - baseline_person[:N].mean()).item()
                )

            loss = det + DPO_WEIGHT * dpo + kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item()
            tdpo += dpo.item()
            tkl += kl.item()

        em = evaluate_model(model, vl, DEV)
        np_m = np.mean(diag["n_pairs"]) if diag["n_pairs"] else 0
        dp_m = np.mean(diag["dpo_loss"]) if diag["dpo_loss"] else 0
        cs_m = np.mean(diag["conf_shift"]) if diag["conf_shift"] else 0
        dg_m = np.mean(diag["disagreement"]) if diag["disagreement"] else 0
        row = {
            "epoch": ep,
            "val_ap50": em["ap50"],
            "val_ap75": em["ap75"],
            "ece": em.get("ece", 0),
            "n_pairs": float(np_m),
            "dpo_loss": float(dp_m),
            "conf_shift": float(cs_m),
            "disagreement": float(dg_m),
            "det": td,
            "dpo": tdpo,
            "kl": tkl,
        }
        h.append(row)
        print(
            f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} "
            f"pairs={np_m:.1f} dpo={dp_m:.3f} disagree={dg_m:.3f}"
        )
        if em["ap75"] > best_ap75:
            best_ap75 = em["ap75"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({
        "run_name": run_name,
        "config": cfg_name,
        "mode": mode,
        "seed": seed,
        "epochs": len(h),
        "best_ap50": best_h["val_ap50"],
        "best_ap75": best_ap75,
        "history": h,
    })
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    calib = build_or_load_calibration()
    results = []
    for cfg, mode in [("det_only", "det_only_unf"), ("disagree_dpo", "dpo")]:
        results.append(run_one(cfg, mode, SEED, calib))

    print("\n## 2.113 Manifold-Confidence Disagreement DPO")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(
            f"  {r['config']:<15s} s{r['seed']} AP75={r['best_ap75']:.4f} "
            f"AP50={bh['val_ap50']:.4f}"
        )

    print(f"\n  Reference: round2108 IoU DPO best AP75 = 0.724")
    print(f"  Reference: round2112 manifold DPO best AP75 = 0.732")
    print(f"  Target:    disagreement DPO should beat both")
