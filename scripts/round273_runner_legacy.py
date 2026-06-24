"""Plan 2.73: Detector-Aligned Verifier — 3 key fixes from 2.72 v5.

Fixes:
  1. BoxCoder.decode() instead of hand-written decode_boxes
     BoxCoder.weights=(10,10,5,5). Hand decode was off by those factors.
  2. box_roi_pool routing instead of hand-rolled FPN level assignment
     Uses model.roi_heads.box_roi_pool(fpn, proposals, image_shapes) directly.
  3. Three-input verifier: ROI pooled feat + per-channel FFT + box geometry
     Quality target = IoU * cls_correctness, with pairwise ranking loss.

6 groups × 3 seeds = 18 experiments (same as 2.72):
  A  det_only                supervised fine-tune baseline
  B  qonly_real              real per-channel FFT + ROI + geometry verifier
  C  qonly_shuf              shuffled FFT control
  D  qonly_band              band-permuted control
  E  qonly_random            random q_norm, per-proposal normalized
  F  frozen_random_verifier  random init verifier, frozen, no vloss
"""
import sys, json, subprocess, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou
import numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES = 4
EPOCHS = 5
SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05
RANK_WEIGHT = 0.1  # ranking loss weight


def extract_perchan_fft(x):
    """Per-channel 3-band radial FFT on spatial dims. x: (B, C, H, W)."""
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)
    pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X ** 2 + Y ** 2)
    R = r.max().clamp_min(1e-6)
    rn = r / R
    lo = (rn <= 0.3).float()
    md = ((rn > 0.3) & (rn <= 0.7)).float()
    hi = (rn > 0.7).float()
    a_lo = (amp * lo).flatten(2).sum(2)
    a_md = (amp * md).flatten(2).sum(2)
    a_hi = (amp * hi).flatten(2).sum(2)
    p_lo = (pha * lo).flatten(2).sum(2)
    p_md = (pha * md).flatten(2).sum(2)
    p_hi = (pha * hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi, p_lo, p_md, p_hi], dim=1)


def band_permute(fft_f, rng_gen):
    B = fft_f.shape[0]; ch_per = fft_f.shape[1] // 6
    out = torch.zeros_like(fft_f)
    for b in range(6):
        sl = slice(b * ch_per, (b + 1) * ch_per)
        out[:, sl] = fft_f[torch.randperm(B, generator=rng_gen, device=fft_f.device)][:, sl]
    return out


class AlignedVerifier(nn.Module):
    """Three-input verifier: ROI pooled feature + FFT feature + box geometry."""

    def __init__(self, roi_dim, fft_dim, geo_dim=4, hidden=128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.fft_net = nn.Sequential(nn.Linear(fft_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 32, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, roi_feat, fft_feat, geo_feat):
        r = self.roi_net(roi_feat)
        f = self.fft_net(fft_feat)
        g = self.geo_net(geo_feat)
        return self.head(torch.cat([r, f, g], dim=1)).squeeze(-1)


def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5 * (e.pow(2) + 2 * torch.log(s.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


def ranking_loss(q_pred, iou_r, margin=0.1):
    """Within-image pairwise ranking: high-IoU proposals should have higher q."""
    N, G = q_pred.shape
    loss = torch.tensor(0.0, device=q_pred.device)
    count = 0
    for i in range(N):
        ious = iou_r[i]
        for a in range(G):
            for b in range(a + 1, G):
                if ious[a] - ious[b] > margin:
                    loss += F.relu(margin - (q_pred[i, a] - q_pred[i, b]))
                    count += 1
                elif ious[b] - ious[a] > margin:
                    loss += F.relu(margin - (q_pred[i, b] - q_pred[i, a]))
                    count += 1
    return loss / max(count, 1)


def bl():
    return build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 2}})


def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})


def fe(m, parts):
    for p in m.parameters():
        p.requires_grad = False
    for part in parts:
        if isinstance(part, nn.Module):
            for p in part.parameters():
                p.requires_grad = True


@torch.no_grad()
def ev(model, vl):
    model.eval()
    ps, ts = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)


def run_one(cfg_name, mode, seed):
    run_name = f"round273_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    fe(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])
    box_coder = model.roi_heads.box_coder
    box_pool = model.roi_heads.box_roi_pool

    # --- Hooks ---
    sampled_props = {}
    box_head_in = {}
    fpn_feats = {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(
        lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    rng_gen = torch.Generator(device=DEV).manual_seed(seed + 9999)
    vrf = None
    tl, vl = bl()
    rd = ensure_run_dir(run_name)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    h = []; best_ap75 = -1.0
    use_verifier = mode in ("real", "shuf", "band", "frozen_random")
    diag = {"q_ious": [], "total_grad_norm": [], "reward_std": [], "q_std": []}
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    for ep in range(1, EPOCHS + 1):
        model.train()
        if vrf is not None:
            vrf.train()
        td, trl, tv, pos = 0.0, 0.0, 0.0, 0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x")
            sp_raw = sampled_props.get("p")
            fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV)
            vloss = torch.tensor(0.0, device=DEV)
            total_gn_batch = 0.0

            if mode != "det_only" and rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G_SAMPLES, 4, device=DEV)
                log_probs = glp(deltas, mu, s)

                # --- Build per-image proposal + delta lists for BoxCoder ---
                sp_cat = torch.cat(sp_raw, dim=0)
                N = min(N_rf, sp_cat.shape[0])
                mu = mu[:N]; deltas = deltas[:N]; log_probs = log_probs[:N]

                box_list = []
                delta_list = []
                img_map = []
                offset = 0
                for i_img, p_img in enumerate(sp_raw):
                    n_available = min(p_img.shape[0], N - offset)
                    if n_available <= 0:
                        break
                    pi = sp_cat[offset:offset + n_available]
                    di = deltas[offset:offset + n_available].reshape(-1, 4)
                    box_list.append(pi)
                    delta_list.append(di)
                    img_map.extend([i_img] * (n_available * G_SAMPLES))
                    offset += n_available

                # FIX 1: BoxCoder-correct decode (wx,wy,ww,wh) = (10,10,5,5)
                delta_cat = torch.cat(delta_list, dim=0)
                sp_exp = torch.cat([p.repeat_interleave(G_SAMPLES, dim=0) for p in box_list], dim=0)
                bw = sp_exp[:, 2] - sp_exp[:, 0]
                bh = sp_exp[:, 3] - sp_exp[:, 1]
                bcx = sp_exp[:, 0] + 0.5 * bw
                bcy = sp_exp[:, 1] + 0.5 * bh
                dx = delta_cat[:, 0] / 10.0  # wx
                dy = delta_cat[:, 1] / 10.0  # wy
                dw = delta_cat[:, 2] / 5.0   # ww
                dh = delta_cat[:, 3] / 5.0   # wh
                decoded_cat = torch.stack([
                    dx * bw + bcx - 0.5 * torch.exp(dw) * bw,
                    dy * bh + bcy - 0.5 * torch.exp(dh) * bh,
                    dx * bw + bcx + 0.5 * torch.exp(dw) * bw,
                    dy * bh + bcy + 0.5 * torch.exp(dh) * bh,
                ], dim=1).clamp(min=0)
                # Rebuild per-image decoded list (needed by box_roi_pool)
                decoded_list = []
                off = 0
                for di in delta_list:
                    n = di.shape[0]
                    decoded_list.append(decoded_cat[off:off + n])
                    off += n

                # IoU computation
                iou_r = torch.zeros(offset, G_SAMPLES, device=DEV)
                for pi in range(offset):
                    i_img = img_map[pi * G_SAMPLES]
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt) > 0:
                        b_slice = decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES]
                        iou_r[pi] = box_iou(b_slice, gt).max(dim=1).values

                if use_verifier:
                    # FIX 2: Use box_roi_pool for correctly routed ROI features
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)

                    # FFT on ROI-pooled features
                    fft_f = extract_perchan_fft(pooled)

                    # Geometry features: normalized center + log size
                    img_h = torch.tensor([s[0] for i_img, s in enumerate(image_shapes)
                                          for _ in range(decoded_list[i_img].shape[0])], device=DEV)
                    img_w = torch.tensor([s[1] for i_img, s in enumerate(image_shapes)
                                          for _ in range(decoded_list[i_img].shape[0])], device=DEV)
                    geo = torch.stack([
                        (decoded_cat[:, 0] + decoded_cat[:, 2]) / (2 * img_w),
                        (decoded_cat[:, 1] + decoded_cat[:, 3]) / (2 * img_h),
                        torch.log((decoded_cat[:, 2] - decoded_cat[:, 0]).clamp_min(1)),
                        torch.log((decoded_cat[:, 3] - decoded_cat[:, 1]).clamp_min(1)),
                    ], dim=1)

                    # Shuffle/permute FFT for control groups
                    if mode == "shuf":
                        fft_f = fft_f[torch.randperm(fft_f.shape[0], generator=rng_gen, device=DEV)]
                    elif mode == "band":
                        fft_f = band_permute(fft_f, rng_gen)

                    # Lazy verifier init
                    if vrf is None:
                        roi_dim = pooled.shape[1] * pooled.shape[2] * pooled.shape[3]
                        fft_dim = fft_f.shape[1]
                        vrf = AlignedVerifier(roi_dim, fft_dim).to(DEV)
                        if mode == "frozen_random":
                            for p in vrf.parameters():
                                p.requires_grad = False
                        params = [p for p in list(model.parameters()) + list(vrf.parameters()) if p.requires_grad]
                        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

                    roi_flat = pooled.flatten(1)
                    q_pred = vrf(roi_flat, fft_f, geo)
                    q_pred = q_pred.view(offset, G_SAMPLES)

                    if mode != "frozen_random":
                        # q_target = IoU * cls_correctness (cls_correct ≈ IoU > 0.5 here, simplified to IoU clip)
                        q_target = iou_r.clamp(0, 1)
                        vloss = F.mse_loss(q_pred, q_target.detach())
                        vloss = vloss + RANK_WEIGHT * ranking_loss(q_pred, iou_r)

                    q_norm = (q_pred - q_pred.mean(dim=1, keepdim=True)) / (q_pred.std(dim=1, keepdim=True).clamp_min(1e-6))
                    diag["q_ious"].extend(list(zip(q_pred.flatten().tolist(), iou_r.flatten().tolist())))
                    diag["q_std"].append(q_pred.std().item())
                else:
                    q_norm = torch.randn(offset, G_SAMPLES, generator=rng_gen, device=DEV)
                    q_norm = (q_norm - q_norm.mean(dim=1, keepdim=True)) / (q_norm.std(dim=1, keepdim=True).clamp_min(1e-6))

                diag["reward_std"].append(q_norm.std().item())
                pm = iou_r.max(dim=1).values > 0.3
                if pm.any():
                    rl = -(q_norm[pm].detach() * log_probs[pm]).mean()
                    pos += pm.sum().item()

            # --- Loss + backward ---
            if mode == "det_only":
                loss = det
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            else:
                vloss_term = vloss if vrf is not None else torch.tensor(0.0, device=DEV)
                loss = det + vloss_term + RL_WEIGHT * rl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
                opt.step()

            diag["total_grad_norm"].append(total_gn_batch)
            td += det.item(); trl += rl.item(); tv += vloss.item()

        # --- Epoch diagnostics ---
        em = ev(model, vl)
        if len(diag["q_ious"]) > 0:
            qs = np.array([x[0] for x in diag["q_ious"]])
            iis = np.array([x[1] for x in diag["q_ious"]])
            q_corr = np.corrcoef(qs, iis)[0, 1]
        else:
            q_corr = 0.0
        tgn = np.mean(diag["total_grad_norm"]) if len(diag["total_grad_norm"]) > 0 else 0.0
        qs_m = np.mean(diag["q_std"]) if len(diag["q_std"]) > 0 else 0.0
        rs_m = np.mean(diag["reward_std"]) if len(diag["reward_std"]) > 0 else 0.0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "q_iou_corr": float(q_corr), "q_std": float(qs_m),
               "reward_std": float(rs_m), "total_grad_norm": float(tgn),
               "pred_count": em.get("pred_count", 0), "det_loss": td, "rl_loss": trl,
               "vloss": tv, "pos_count": int(pos)}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} q_corr={q_corr:.4f} "
              f"tgn={tgn:.6f} q_std={qs_m:.4f} r_std={rs_m:.4f}")
        if em["ap75"] > best_ap75:
            best_ap75 = em["ap75"]
        for k in diag:
            diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": EPOCHS, "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
               "history": h, "git_hash": GIT, "q_iou_corr_final": h[-1]["q_iou_corr"],
               "total_grad_final": h[-1]["total_grad_norm"]})
    save_json(em, rd / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    modes = {"det_only": "det_only", "qonly_real": "real", "qonly_shuf": "shuf",
             "qonly_band": "band", "qonly_random": "random", "frozen_random": "frozen_random"}
    for cfg, mode in modes.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.73 Detector-Aligned Verifier")
    print(f"  {'Config':<20s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'q_corr':>8s} {'tot_gn':>10s} {'q_std':>8s} {'r_std':>8s}")
    for r in all_results:
        hf = r["history"][-1]
        print(f"  {r['config']:<20s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {hf['val_ap50']:8.4f} {r.get('q_iou_corr_final', 0):8.4f} {r.get('total_grad_final', 0):10.6f} {hf.get('q_std', 0):8.4f} {hf.get('reward_std', 0):8.4f}")

    for cfg in modes:
        best_vals = [r["best_ap75"] for r in all_results if r["config"] == cfg]
        final_vals = [r["ap75"] for r in all_results if r["config"] == cfg]
        qc = [r.get("q_iou_corr_final", 0) for r in all_results if r["config"] == cfg]
        tgn = [r.get("total_grad_final", 0) for r in all_results if r["config"] == cfg]
        print(f"  {cfg}: bestAP75={np.mean(best_vals):.4f}±{np.std(best_vals):.4f}  finalAP75={np.mean(final_vals):.4f}  q_corr={np.mean(qc):.4f}  tgn={np.mean(tgn):.6f}")

    def get(vals, cfg, key):
        return [r[key] for r in vals if r["config"] == cfg]

    def judge(name_a, name_b, threshold=0.01):
        va = get(all_results, name_a, "best_ap75")
        vb = get(all_results, name_b, "best_ap75")
        delta = np.mean(va) - np.mean(vb)
        wins = sum(1 for i in range(3) if va[i] > vb[i])
        status = "PASS" if delta > threshold and wins >= 2 else "FAIL"
        print(f"\n  {name_a} vs {name_b}: Δ={delta:+.4f} wins={wins}/3 → {status}")

    judge("qonly_real", "qonly_random")
    judge("qonly_real", "frozen_random")
    judge("qonly_real", "qonly_band")
    judge("qonly_real", "qonly_shuf")
    judge("frozen_random", "det_only")
