"""Plan 2.74 v3: Fix C/D/E reward computation bugs.

C (nms_aware): per-image reward for ALL proposals (v2 only processed first proposal)
D (ap75_event): classifier score + NMS → true AP75-event reward
E (fft_residual): image-level normalize (was per-proposal G=4)

All: image-level advantage normalization instead of per-proposal G=4.
Box decode confirmed correct vs TorchVision BoxCoder.decode (IoU=1.0).
"""
import sys, json, subprocess, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou, nms, batched_nms
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
RANK_WEIGHT = 0.1
FFT_RESIDUAL_WEIGHT = 0.1


def extract_perchan_fft(x):
    C = x.shape[1]; H, W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft); pha = torch.angle(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device)
    freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y, X = torch.meshgrid(freq_h, freq_w, indexing='ij')
    r = torch.sqrt(X ** 2 + Y ** 2); R = r.max().clamp_min(1e-6); rn = r / R
    lo = (rn <= 0.3).float(); md = ((rn > 0.3) & (rn <= 0.7)).float(); hi = (rn > 0.7).float()
    a_lo = (amp * lo).flatten(2).sum(2); a_md = (amp * md).flatten(2).sum(2); a_hi = (amp * hi).flatten(2).sum(2)
    p_lo = (pha * lo).flatten(2).sum(2); p_md = (pha * md).flatten(2).sum(2); p_hi = (pha * hi).flatten(2).sum(2)
    return torch.cat([a_lo, a_md, a_hi, p_lo, p_md, p_hi], dim=1)


def compute_nms_aware_reward_per_image(iou_r_img, decoded_img, gt_boxes):
    """Per-image NMS-aware reward. Uses classifier scores + NMS.
    iou_r_img: (n_proposals, G) IoU matrix for one image
    decoded_img: (n_proposals*G, 4) decoded boxes
    Returns: (n_proposals, G) reward, plus diagnostics.
    """
    N, G = iou_r_img.shape
    total = N * G
    reward = torch.zeros(N, G, device=iou_r_img.device)
    diag = {"ap75_tp": 0, "ap50_tp": 0, "high_fp": 0, "duplicate": 0}

    if len(gt_boxes) == 0:
        reward[:] = -1.0
        diag["high_fp"] = total
        return reward, diag

    # Greedy IoU matching: best box per GT
    iou_flat = iou_r_img.flatten()
    best_iou, best_gt = box_iou(decoded_img, gt_boxes).max(dim=1)
    gt_best = {}
    for i in range(len(gt_boxes)):
        mask = best_gt == i
        if mask.any():
            gt_best[i] = mask.nonzero()[best_iou[mask].argmax()].item()

    # Apply NMS on decoded boxes (using IoU as proxy score)
    # NMS keeps best-IoU boxes and removes high-IoU-overlap boxes
    scores = best_iou  # proxy score
    keep = nms(decoded_img, scores, iou_threshold=0.5)
    keep_set = set(keep.tolist())

    for idx in range(total):
        iou_val = best_iou[idx].item()
        gt_i = best_gt[idx].item()

        if idx in keep_set:
            if iou_val >= 0.75:
                if gt_best.get(gt_i) == idx:
                    reward.view(-1)[idx] = 1.0
                    diag["ap75_tp"] += 1
                else:
                    reward.view(-1)[idx] = -0.5
                    diag["duplicate"] += 1
            elif iou_val >= 0.5:
                if gt_best.get(gt_i) == idx:
                    reward.view(-1)[idx] = 0.3
                    diag["ap50_tp"] += 1
                else:
                    reward.view(-1)[idx] = -0.5
                    diag["duplicate"] += 1
            else:
                reward.view(-1)[idx] = -0.5
                diag["high_fp"] += 1
        else:
            # Suppressed by NMS
            if iou_val >= 0.5:
                reward.view(-1)[idx] = -0.5
                diag["duplicate"] += 1
            else:
                reward.view(-1)[idx] = -1.0
                diag["high_fp"] += 1

    return reward, diag


def compute_ap75_reward_with_scores(decoded_img, gt_boxes, cls_scores):
    """True AP75-event reward: classifier scores + NMS + greedy match.
    decoded_img: (N*G, 4), cls_scores: (N*G,) foreground probability.
    Returns: (N*G,) reward, diagnostics.
    """
    total = decoded_img.shape[0]
    reward = torch.zeros(total, device=decoded_img.device)
    diag = {"ap75_tp": 0, "ap50_tp": 0, "high_fp": 0, "duplicate": 0}

    if len(gt_boxes) == 0:
        reward[:] = -1.0
        diag["high_fp"] = total
        return reward, diag

    # NMS
    keep = nms(decoded_img, cls_scores, iou_threshold=0.5)
    keep_set = set(keep.tolist())

    # Match survivors to GT
    if len(keep) > 0:
        iou_surv = box_iou(decoded_img[keep], gt_boxes)
        best_iou_surv, best_gt_surv = iou_surv.max(dim=1)
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=decoded_img.device)

        for si, ki in enumerate(keep.tolist()):
            iou_val = best_iou_surv[si].item()
            gt_i = best_gt_surv[si].item()
            if iou_val >= 0.75 and not gt_matched[gt_i]:
                reward[ki] = 1.0
                gt_matched[gt_i] = True
                diag["ap75_tp"] += 1
            elif iou_val >= 0.5 and not gt_matched[gt_i]:
                reward[ki] = 0.3
                gt_matched[gt_i] = True
                diag["ap50_tp"] += 1
            else:
                reward[ki] = -0.5
                diag["duplicate"] += 1

    # Non-surviving boxes
    for idx in range(total):
        if idx not in keep_set:
            iou_val = box_iou(decoded_img[idx:idx + 1], gt_boxes).max().item()
            if iou_val < 0.3:
                reward[idx] = -1.0
                diag["high_fp"] += 1
            else:
                reward[idx] = -0.5
                diag["duplicate"] += 1

    return reward, diag


class AlignedVerifier(nn.Module):
    def __init__(self, roi_dim, fft_dim, geo_dim=4, hidden=128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.fft_net = nn.Sequential(nn.Linear(fft_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 32, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, roi_feat, fft_feat, geo_feat):
        r = self.roi_net(roi_feat); f = self.fft_net(fft_feat); g = self.geo_net(geo_feat)
        return self.head(torch.cat([r, f, g], dim=1)).squeeze(-1)


class BaseVerifier(nn.Module):
    """ROI + Geo → q. Used as baseline in fft_residual."""
    def __init__(self, roi_dim, geo_dim=4, hidden=128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, roi_feat, geo_feat):
        return self.head(torch.cat([self.roi_net(roi_feat), self.geo_net(geo_feat)], dim=1)).squeeze(-1)


class FFTResidualVerifier(nn.Module):
    """FFT-only residual verifier (no sigmoid, can be negative)."""
    def __init__(self, fft_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fft_dim, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, fft_feat):
        return self.net(fft_feat).squeeze(-1)


def glp(d, m, s):
    e = (d - m.unsqueeze(1)) / s.unsqueeze(1)
    return -0.5 * (e.pow(2) + 2 * torch.log(s.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


def ranking_loss(q_pred, iou_r, margin=0.1):
    N, G = q_pred.shape
    loss = torch.tensor(0.0, device=q_pred.device); count = 0
    for i in range(N):
        ious = iou_r[i]
        for a in range(G):
            for b in range(a + 1, G):
                if ious[a] - ious[b] > margin:
                    loss += F.relu(margin - (q_pred[i, a] - q_pred[i, b])); count += 1
                elif ious[b] - ious[a] > margin:
                    loss += F.relu(margin - (q_pred[i, b] - q_pred[i, a])); count += 1
    return loss / max(count, 1)


def image_level_normalize(x, N, G):
    """Normalize across all boxes in the image: (x - mean) / std."""
    flat = x.view(-1)
    return ((flat - flat.mean()) / (flat.std().clamp_min(1e-6))).view(N, G)


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
    """mode: 'nms_aware' | 'ap75_event' | 'fft_residual'"""
    run_name = f"round274_{cfg_name}_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    fe(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])
    box_pool = model.roi_heads.box_roi_pool

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(
        lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    rng_gen = torch.Generator(device=DEV).manual_seed(seed + 9999)
    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 7777)
    tl, vl = bl()
    run_dir = ensure_run_dir(run_name)

    is_residual = mode == "fft_residual"
    needs_verifier = mode in ("nms_aware", "fft_residual")
    vrf = None; vrf_base = None; vrf_fft = None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    bbox_pred_weight = model.roi_heads.box_predictor.bbox_pred.weight

    h = []; best_ap75 = -1.0
    diag = {"q_ious": [], "total_grad_norm": [], "reward_std": [], "q_std": [],
            "ap75_tp": [], "ap50_tp": [], "high_fp": [], "duplicate": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        for v in [vrf, vrf_base, vrf_fft]:
            if v is not None: v.train()
        td, trl, tv, pos = 0.0, 0.0, 0.0, 0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV); vloss = torch.tensor(0.0, device=DEV)
            total_gn_batch = 0.0

            if rf is not None and sp_raw is not None and rf.shape[0] > 0 and fpn is not None:
                N_rf = rf.shape[0]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
                # Also get classifier scores for AP75-event reward
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                cls_probs = F.softmax(cls_logits, dim=1)[:, 1]  # foreground prob

                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G_SAMPLES, 4, device=DEV)
                log_probs = glp(deltas, mu, s)

                sp_cat = torch.cat(sp_raw, dim=0); N = min(N_rf, sp_cat.shape[0])
                mu = mu[:N]; deltas = deltas[:N]; log_probs = log_probs[:N]

                box_list, delta_list, img_map = [], [], []
                offset = 0
                for i_img, p_img in enumerate(sp_raw):
                    n_a = min(p_img.shape[0], N - offset)
                    if n_a <= 0: break
                    box_list.append(sp_cat[offset:offset + n_a])
                    delta_list.append(deltas[offset:offset + n_a].reshape(-1, 4))
                    img_map.extend([i_img] * (n_a * G_SAMPLES))
                    offset += n_a

                # BoxCoder-correct decode
                sp_exp = torch.cat([p.repeat_interleave(G_SAMPLES, dim=0) for p in box_list], dim=0)
                delta_cat = torch.cat(delta_list, dim=0)
                bw = sp_exp[:, 2] - sp_exp[:, 0]; bh = sp_exp[:, 3] - sp_exp[:, 1]
                bcx = sp_exp[:, 0] + 0.5 * bw; bcy = sp_exp[:, 1] + 0.5 * bh
                dx = delta_cat[:, 0] / 10.0; dy = delta_cat[:, 1] / 10.0
                dw = delta_cat[:, 2] / 5.0;  dh = delta_cat[:, 3] / 5.0
                decoded_cat = torch.stack([
                    dx * bw + bcx - 0.5 * torch.exp(dw) * bw,
                    dy * bh + bcy - 0.5 * torch.exp(dh) * bh,
                    dx * bw + bcx + 0.5 * torch.exp(dw) * bw,
                    dy * bh + bcy + 0.5 * torch.exp(dh) * bh,
                ], dim=1).clamp(min=0)

                decoded_list, off = [], 0
                for di in delta_list:
                    n = di.shape[0]; decoded_list.append(decoded_cat[off:off + n]); off += n

                # IoU
                iou_r = torch.zeros(offset, G_SAMPLES, device=DEV)
                for pi in range(offset):
                    i_img = img_map[pi * G_SAMPLES]
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt) > 0:
                        iou_r[pi] = box_iou(decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES], gt).max(dim=1).values

                # --- Per-image reward computation ---
                reward_img = torch.zeros(offset, G_SAMPLES, device=DEV)
                rd_tot = {"ap75_tp": 0, "ap50_tp": 0, "high_fp": 0, "duplicate": 0}

                for i_img in range(len(tgts_t)):
                    mask = torch.tensor([img_map[pi * G_SAMPLES] == i_img for pi in range(offset)], device=DEV)
                    pi_list = mask.nonzero(as_tuple=True)[0].tolist()
                    if not pi_list: continue

                    # Get all decoded boxes for this image
                    dec_img = torch.cat([decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES] for pi in pi_list], dim=0)
                    iou_img = torch.stack([iou_r[pi] for pi in pi_list], dim=0)  # (n_prop_img, G)

                    if mode == "nms_aware":
                        rwd, rd_img = compute_nms_aware_reward_per_image(iou_img, dec_img, tgts_t[i_img]["boxes"])
                    elif mode in ("ap75_event", "fft_residual"):
                        cls_img = torch.cat([cls_probs[pi].repeat(G_SAMPLES) for pi in pi_list], dim=0)
                        rwd_flat, rd_img = compute_ap75_reward_with_scores(dec_img, tgts_t[i_img]["boxes"], cls_img)
                        rwd = rwd_flat.view(len(pi_list), G_SAMPLES)

                    for k in rd_tot: rd_tot[k] += rd_img[k]
                    for j, pi in enumerate(pi_list):
                        reward_img[pi] = rwd[j]

                for k in rd_tot: diag[k].append(rd_tot[k])

                # --- Verifier (if needed) ---
                if needs_verifier:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    roi_flat = pooled.flatten(1)
                    fft_f = extract_perchan_fft(pooled)
                    geo = torch.cat([
                        torch.stack([
                            (b[:, 0] + b[:, 2]) / (2 * image_shapes[i_img][1]),
                            (b[:, 1] + b[:, 3]) / (2 * image_shapes[i_img][0]),
                            torch.log((b[:, 2] - b[:, 0]).clamp_min(1)),
                            torch.log((b[:, 3] - b[:, 1]).clamp_min(1)),
                        ], dim=1) for i_img, b in enumerate(decoded_list)], dim=0)

                    if mode == "nms_aware":
                        if vrf is None:
                            roi_dim = pooled.shape[1] * pooled.shape[2] * pooled.shape[3]
                            fft_dim = fft_f.shape[1]
                            vrf = AlignedVerifier(roi_dim, fft_dim).to(DEV)
                            params = [p for p in list(model.parameters()) + list(vrf.parameters()) if p.requires_grad]
                            opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

                        q_pred = vrf(roi_flat, fft_f, geo).view(offset, G_SAMPLES)
                        q_target = iou_r.clamp(0, 1)
                        vloss = F.mse_loss(q_pred, q_target.detach()) + RANK_WEIGHT * ranking_loss(q_pred, iou_r)
                        diag["q_ious"].extend(list(zip(q_pred.flatten().tolist(), iou_r.flatten().tolist())))
                        diag["q_std"].append(q_pred.std().item())
                        # Image-level normalize verifier-based reward
                        q_norm = image_level_normalize(q_pred, offset, G_SAMPLES)

                    elif mode == "fft_residual":
                        fft_shuf = fft_f[torch.randperm(fft_f.shape[0], generator=rng_shuf, device=DEV)]

                        if vrf_base is None:
                            roi_dim = pooled.shape[1] * pooled.shape[2] * pooled.shape[3]
                            vrf_base = BaseVerifier(roi_dim).to(DEV)
                            vrf_fft = FFTResidualVerifier(fft_f.shape[1]).to(DEV)
                            params = [p for p in list(model.parameters()) + list(vrf_base.parameters()) + list(vrf_fft.parameters()) if p.requires_grad]
                            opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

                        q_base = vrf_base(roi_flat, geo).view(offset, G_SAMPLES)
                        q_fft_real = vrf_fft(fft_f).view(offset, G_SAMPLES)
                        q_fft_shuf = vrf_fft(fft_shuf).view(offset, G_SAMPLES)

                        base_target = reward_img.clamp(-1, 1)
                        vloss = F.mse_loss(q_base, base_target.detach())
                        residual_target = (base_target - q_base.detach()).clamp(-1, 1)
                        vloss = vloss + F.mse_loss(q_fft_real, residual_target.detach())
                        high_iou_mask = iou_r.max(dim=1).values > 0.5
                        if high_iou_mask.any():
                            vloss = vloss + 0.1 * F.relu(0.1 - (q_fft_real[high_iou_mask].mean() - q_fft_shuf[high_iou_mask].mean()))

                        final_reward = reward_img + FFT_RESIDUAL_WEIGHT * q_fft_real.detach()
                        q_norm = image_level_normalize(final_reward, offset, G_SAMPLES)
                        diag["q_ious"].extend(list(zip(q_fft_real.flatten().tolist(), iou_r.flatten().tolist())))
                        diag["q_std"].append(q_fft_real.std().item())

                else:
                    # ap75_event: no verifier, use reward directly, image-level normalize
                    q_norm = image_level_normalize(reward_img, offset, G_SAMPLES)

                diag["reward_std"].append(q_norm.std().item())
                pm = iou_r.max(dim=1).values > 0.3
                if pm.any():
                    rl = -(q_norm[pm].detach() * log_probs[pm]).mean()
                    pos += pm.sum().item()

            # --- Backward ---
            vloss_term = vloss if (vrf is not None or vrf_base is not None) else torch.tensor(0.0, device=DEV)
            loss = det + vloss_term + RL_WEIGHT * rl
            opt.zero_grad(set_to_none=True); loss.backward()
            total_gn_batch = bbox_pred_weight.grad.norm().item() if bbox_pred_weight.grad is not None else 0.0
            opt.step()

            diag["total_grad_norm"].append(total_gn_batch)
            td += det.item(); trl += rl.item(); tv += vloss.item()

        em = ev(model, vl)
        if len(diag["q_ious"]) > 0:
            qs = np.array([x[0] for x in diag["q_ious"]])
            iis = np.array([x[1] for x in diag["q_ious"]])
            q_corr = np.corrcoef(qs, iis)[0, 1] if len(qs) > 1 else 0.0
        else:
            q_corr = 0.0
        tgn = np.mean(diag["total_grad_norm"]) if len(diag["total_grad_norm"]) > 0 else 0.0
        qs_m = np.mean(diag["q_std"]) if len(diag["q_std"]) > 0 else 0.0
        rs_m = np.mean(diag["reward_std"]) if len(diag["reward_std"]) > 0 else 0.0
        ap75_tp = np.sum(diag["ap75_tp"]); dup = np.sum(diag["duplicate"]); hfp = np.sum(diag["high_fp"])
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "q_iou_corr": float(q_corr), "q_std": float(qs_m),
               "reward_std": float(rs_m), "total_grad_norm": float(tgn),
               "ap75_tp": int(ap75_tp), "duplicate_fp": int(dup), "high_conf_fp": int(hfp),
               "det_loss": td, "rl_loss": trl, "vloss": tv, "pos_count": int(pos)}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} q_corr={q_corr:.4f} "
              f"tgn={tgn:.6f} r_std={rs_m:.4f} AP75tp={ap75_tp} dup={dup} hFP={hfp}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed,
               "epochs": EPOCHS, "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75,
               "history": h, "git_hash": GIT, "q_iou_corr_final": h[-1]["q_iou_corr"],
               "total_grad_final": h[-1]["total_grad_norm"]})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    configs = {"nms_aware_v3": "nms_aware", "ap75_event_v3": "ap75_event", "fft_residual_v3": "fft_residual"}
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.74 v3 — Fixed C/D/E")
    print(f"  {'Config':<18s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'Rec':>8s} {'Prec':>8s} {'q_corr':>8s} {'r_std':>8s} {'AP75tp':>8s} {'dup':>6s} {'hFP':>6s}")
    for r in all_results:
        hf = r["history"][-1]
        print(f"  {r['config']:<18s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {hf['val_ap50']:8.4f} {hf.get('recall', 0):8.4f} {hf.get('precision', 0):8.4f} {r.get('q_iou_corr_final', 0):8.4f} {hf.get('reward_std', 0):8.4f} {hf.get('ap75_tp', 0):8d} {hf.get('duplicate_fp', 0):6d} {hf.get('high_conf_fp', 0):6d}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        bv = [r["best_ap75"] for r in vals]; fv = [r["ap75"] for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f}±{np.std(bv):.4f}  finalAP75={np.mean(fv):.4f}")

    def judge(name_a, name_b, threshold=0.01):
        va = [r["best_ap75"] for r in all_results if r["config"] == name_a]
        vb = [r["best_ap75"] for r in all_results if r["config"] == name_b]
        if not va or not vb: return
        delta = np.mean(va) - np.mean(vb)
        wins = sum(1 for i in range(3) if va[i] > vb[i])
        status = "PASS" if delta > threshold and wins >= 2 else "FAIL"
        print(f"\n  {name_a} vs {name_b}: Δ={delta:+.4f} wins={wins}/3 → {status}")

    print("\n## Key: fft_residual vs ap75_event")
    judge("fft_residual_v3", "ap75_event_v3")
    judge("nms_aware_v3", "ap75_event_v3")
