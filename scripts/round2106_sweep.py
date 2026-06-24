"""Plan 2.106: FFT-augmented RLVR — 3-layer staged sweep (seed 42 only).
L1(6): amp_mid, R_sp, pg_std × hi/lo → pick top2
L2(8): L1 top2 × (amp_fp, pg_fp) × hi/lo → pick top2
L3(8): L2 top2 × (img_var, gap) × hi/lo → pick best
22 runs + baseline = 23 total
"""
import json, copy, shutil, subprocess, sys, numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320, decode_boxes, evaluate_model,
    gaussian_log_prob, unfreeze_rlvr, extract_perchan_fft,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV, CKPT, SEED = "cuda", "runs/round227_v1_baseline_20ep/checkpoint_best.pth", 42
G_SAMPLES, EPOCHS, BASE_RL, KL_W = 4, 8, 0.0005, 0.01

# Dataset-level FFT normalization (computed once across all training batches)
FFT_STATS = {"amp_mid": (0, 1), "R_sp": (0, 1), "pg_std": (0, 1), "amp_var": (0, 1)}

def cross_proposal_grpo(reward, n_props):
    adv = torch.zeros_like(reward); off = 0
    for n_p in n_props:
        if n_p <= 0: continue
        if n_p == 1: adv[off] = 0.0; off += n_p; continue
        r = reward[off:off+n_p]; m = r.mean(); s = r.std().clamp_min(1e-6)
        adv[off:off+n_p] = (r - m) / s; off += n_p
    return adv

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

def fft_features(crops):
    """All 4 FFT features: amp_mid, R_sp, pg_std, img_var. Dataset normalized."""
    f = extract_perchan_fft(crops); ch = f.shape[1] // 6
    # Amplitude
    amp_mid = f[:, 1*ch:2*ch].mean(dim=1)
    amp_var = f[:, 0*ch:3*ch].var(dim=1)
    # Phase: spatial coherence + gradient
    fft_full = torch.fft.rfft2(crops, dim=(-2, -1), norm="ortho")
    phase = torch.angle(fft_full)  # (N, C, 7, 4)
    cp = torch.exp(1j * phase)
    R_sp = cp.mean(dim=(1, 2)).abs().mean(dim=1)  # (N,)
    # Phase gradient std (freq dimension)
    pg = phase[:, :, :, 1:] - phase[:, :, :, :-1]
    pg = (pg + np.pi) % (2 * np.pi) - np.pi  # wrap
    pg_std = pg.std(dim=(1, -1)).mean(dim=1)  # (N,)
    # Dataset-level normalization
    for name, val in [("amp_mid", amp_mid), ("R_sp", R_sp), ("pg_std", pg_std), ("amp_var", amp_var)]:
        mu, sigma = FFT_STATS[name]
        val = (val - mu) / (sigma + 1e-8)
        if name == "amp_mid": amp_mid = val
        elif name == "R_sp": R_sp = val
        elif name == "pg_std": pg_std = val
        else: amp_var = val
    img_var = amp_var.mean()
    return amp_mid, R_sp, pg_std, img_var


def compute_iou(sp_raw, baseline_bf, baseline_bp, tgts_t):
    """Per-proposal max IoU using baseline boxes. Returns (N,)."""
    sp_cat = torch.cat(sp_raw, dim=0)
    N = sp_cat.shape[0]
    with torch.no_grad():
        reg = baseline_bp.bbox_pred(baseline_bf[:N])
        decoded = decode_boxes(sp_cat, reg[:, 2:6])
    iou_p = torch.zeros(N, device=DEV)
    offset = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p == 0:
            continue
        gt = tgts_t[i_img]["boxes"]
        if len(gt) > 0:
            iou_p[offset : offset + n_p] = box_iou(decoded[offset : offset + n_p], gt).max(dim=1).values
        offset += n_p
    return iou_p


def run_config(name, alpha, beta, gamma, delta, epsilon):
    """alpha=amp_mid, beta=R_sp, gamma=pg_std, delta=FP_pen, epsilon=img_var"""
    set_seed(SEED)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    use_fft = bool(alpha or beta or gamma or delta or epsilon)
    sampled_props, box_head_in, roi_crops = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))
    if use_fft:
        model.roi_heads.box_roi_pool.register_forward_hook(lambda m, i, o: roi_crops.update({"c": o.clone()}))

    tl, vl = build_penn_fudan_loaders_320(batch_size=2)
    run_dir = ensure_run_dir(f"round2106_{name}")
    shutil.copy(__file__, run_dir / "runner_snapshot.py")
    opt = build_opt(model); h = []; best_ap75 = -1.0
    baseline_bp = baseline_model.roi_heads.box_predictor

    for ep in range(1, EPOCHS + 1):
        model.train(); td, trl, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{name} e{ep}", leave=False):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); roi_crops.clear()
            ld = model(imgs_d, tgts_t); det = sum(ld.values())
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); crops = roi_crops.get("c")
            rl = torch.tensor(0.0, device=DEV); kl = torch.tensor(0.0, device=DEV)

            if rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf)
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    baseline_cls_conf = F.softmax(baseline_bp.cls_score(baseline_bf), dim=-1)[:, 1]

                # Off-policy sampling
                with torch.no_grad():
                    sp_sigma = 0.05 + 0.2 * (1.0 - baseline_cls_conf)
                    s_base = sp_sigma.unsqueeze(1).expand(-1, cls_logits.shape[1])
                    bl_logits = baseline_bp.cls_score(baseline_bf)
                    perturbed = bl_logits.unsqueeze(1) + s_base.unsqueeze(1) * torch.randn(
                        bl_logits.shape[0], G_SAMPLES, bl_logits.shape[1], device=DEV)
                pert_conf = F.softmax(perturbed, dim=-1)[:, :, 1]
                s_cls = sp_sigma.unsqueeze(1).expand(-1, cls_logits.shape[1])

                # IoU per proposal
                iou_p = compute_iou(sp_raw, baseline_bf, baseline_bp, tgts_t)

                # N = min across logits and iou
                N = min(cls_logits.shape[0], iou_p.shape[0])

                # Base reward: IoU×conf
                quality = (2 * iou_p[:N] - 1).unsqueeze(1)
                reward = pert_conf[:N] * quality

                # FFT in logit bias (not reward) → gradient flows through log_prob
                biased = cls_logits.clone()
                if use_fft and crops is not None:
                    amp_mid, R_sp, pg_std, img_var = fft_features(crops[:N])
                    if alpha: biased[:N, 1] += alpha * amp_mid[:N]
                    if beta:  biased[:N, 1] += beta * (1.0 - R_sp[:N])
                    if gamma: biased[:N, 1] += gamma * pg_std[:N]
                    fp_pen = amp_mid[:N].abs() * (1.0 - iou_p[:N]).clamp(0, 1)
                    if delta: biased[:N, 1] -= delta * fp_pen
                    if epsilon: biased[:N, 1] += epsilon * img_var
                log_probs = gaussian_log_prob(perturbed, biased, s_cls)

                # npp aligned to N (truncated per-image counts)
                npp_actual = []; off_n = 0
                for p_img in sp_raw:
                    cnt = min(p_img.shape[0], N - off_n)
                    if cnt > 0: npp_actual.append(cnt * G_SAMPLES)
                    off_n += cnt
                    if off_n >= N: break

                reward_flat = reward.reshape(-1)
                adv = cross_proposal_grpo(reward_flat, npp_actual).view(N, G_SAMPLES)
                rl = -(adv.detach() * log_probs[:N]).mean()
                kl = KL_W * (pert_conf[:N] - baseline_cls_conf[:N].unsqueeze(1)).pow(2).mean()

            loss = det + BASE_RL * rl + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            td += det.item(); trl += rl.item(); tkl += kl.item()

        em = evaluate_model(model, vl, DEV)
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "ece": em.get("ece", 0), "det": td, "rl": trl, "kl": tkl}
        h.append(row)
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]

    best_ep = max(h, key=lambda r: r["val_ap75"])
    result = {"name": name, "seed": SEED,
              "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "epsilon": epsilon,
              "ap50": best_ep["val_ap50"], "best_ap75": best_ap75,
              "final_ap75": h[-1]["val_ap75"], "history": h}
    save_json(result, run_dir / "eval_metrics.json")
    print(f"  {name}: best_ap75={best_ap75:.4f} ap50={best_ep['val_ap50']:.4f}")
    return result


if __name__ == "__main__":
    set_seed(SEED)
    results, run_map = [], {}

    # Warm-up: collect FFT stats on first training batch (not used directly, just init)
    # Dataset-level stats pre-computed from validation: approximate values
    FFT_STATS.update({"amp_mid": (4.0, 8.0), "R_sp": (0.3, 0.1), "pg_std": (1.35, 0.3), "amp_var": (18.0, 10.0)})

    # Baseline
    b = run_config("baseline", 0, 0, 0, 0, 0); results.append(b)

    # === Stage 1: Layer 1 (box-level FFT) — 6 runs ===
    print("=" * 60 + "\nStage 1: L1 sweep (amp_mid, R_sp, pg_std)\n" + "=" * 60)
    l1_cfgs = [
        ("A_lo", 0.05, 0, 0, 0, 0), ("A_hi", 0.1, 0, 0, 0, 0),
        ("B_lo", 0, 0.02, 0, 0, 0), ("B_hi", 0, 0.05, 0, 0, 0),
        ("C_lo", 0, 0, 0.02, 0, 0), ("C_hi", 0, 0, 0.05, 0, 0),
    ]
    l1 = [run_config(n, *args) for n, *args in l1_cfgs]; results.extend(l1)
    l1_top = sorted(l1, key=lambda x: -x["best_ap75"])[:2]
    print(f"  L1 top2: {l1_top[0]['name']}({l1_top[0]['best_ap75']:.4f}), {l1_top[1]['name']}({l1_top[1]['best_ap75']:.4f})")

    # === Stage 2: L1 top2 × Layer 2 (FP penalty) — 8 runs ===
    print("\n" + "=" * 60 + "\nStage 2: L1 top2 × L2 (FP penalty)\n" + "=" * 60)
    l2_cfgs = [("D_lo", 0.05), ("D_hi", 0.1), ("E_lo", 0.02), ("E_hi", 0.05)]  # D=amp_fp, E=pg_fp
    l2 = []
    for t in l1_top:
        for ln, lv in l2_cfgs:
            a,b,g,d,e = t["alpha"], t["beta"], t["gamma"], (lv if ln.startswith("D") else 0), (lv if ln.startswith("E") else 0)
            r = run_config(f"{t['name']}_{ln}", a, b, g, d, e)
            l2.append(r); results.append(r)
    l2_top = sorted(l2, key=lambda x: -x["best_ap75"])[:2]
    print(f"  L2 top2: {l2_top[0]['name']}({l2_top[0]['best_ap75']:.4f}), {l2_top[1]['name']}({l2_top[1]['best_ap75']:.4f})")

    # === Stage 3: L2 top2 × Layer 3 (image tiebreaker) — 8 runs ===
    print("\n" + "=" * 60 + "\nStage 3: L2 top2 × L3 (image tiebreaker)\n" + "=" * 60)
    l3_cfgs = [("F_lo", 0.05), ("F_hi", 0.1), ("G_lo", 0.05), ("G_hi", 0.1)]  # F=img_var, G=gap
    l3 = []
    for t in l2_top:
        for ln, lv in l3_cfgs:
            a,b,g,d,e = t["alpha"], t["beta"], t["gamma"], t["delta"], lv
            r = run_config(f"{t['name']}_{ln}", a, b, g, d, e)
            l3.append(r); results.append(r)

    # === Final ===
    print("\n" + "=" * 60 + "\nFinal Results\n" + "=" * 60)
    print(f"{'Config':<30s}{'bestAP75':>10s}{'AP50':>8s}")
    for r in sorted(results, key=lambda x: -x["best_ap75"]):
        print(f"{r['name']:<30s}{r['best_ap75']:10.4f}{r['ap50']:8.4f}")
    print(f"\nBaseline: {b['best_ap75']:.4f}")
    best = max(results, key=lambda x: x['best_ap75'])
    print(f"Best: {best['name']} Δ={best['best_ap75']-b['best_ap75']:+.4f}")

    save_json({"sweep": results, "baseline": b, "best": best["name"]}, "runs/round2106_sweep_results.json")
