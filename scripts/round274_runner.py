"""Plan 2.74 v2: Reward-Aligned RLVR — fix reward before adding FFT.

5 groups × 3 seeds = 15 experiments:

  A  local_iou       Local per-proposal normalize(q_iou) — negative control (2.73 ref)
  B  image_adv       Image-level advantage: normalize across all boxes in image
  C  nms_aware       NMS-aware reward: +IoU - duplicate_penalty - FP_penalty
  D  ap75_event      Discrete AP75-event reward: +1/0.3/-0.5/-1 per box
  E  fft_residual    AP75-event + FFT residual verifier with real>shuf constraint

Design principles:
  1. Reward aligns with AP75 events (groups D+E), not IoU regression
  2. FFT predicts residual beyond ROI+Geo baseline (group E)
  3. Only if E > D is FFT proven to have incremental value
"""
import sys, json, subprocess, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou, nms
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
FFT_RESIDUAL_WEIGHT = 0.1  # λ for FFT residual contribution to reward in group E


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


def band_permute(fft_f, rng_gen):
    B = fft_f.shape[0]; ch_per = fft_f.shape[1] // 6
    out = torch.zeros_like(fft_f)
    for b in range(6):
        sl = slice(b * ch_per, (b + 1) * ch_per)
        out[:, sl] = fft_f[torch.randperm(B, generator=rng_gen, device=fft_f.device)][:, sl]
    return out


class AlignedVerifier(nn.Module):
    """ROI + FFT + Geo → q. Same as 2.73."""
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
    """ROI + Geo → q (no FFT). Used as baseline in group E."""
    def __init__(self, roi_dim, geo_dim=4, hidden=128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, roi_feat, geo_feat):
        return self.head(torch.cat([self.roi_net(roi_feat), self.geo_net(geo_feat)], dim=1)).squeeze(-1)


class FFTResidualVerifier(nn.Module):
    """FFT-only verifier for residual signal (no sigmoid, can be negative)."""
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



def compute_ap75_reward(decoded_boxes, gt_boxes, nms_thresh=0.5):
    """Discrete AP75-event reward for each decoded box.
    Returns: reward vector (N*G,), dict with diagnostics.
    """
    total = decoded_boxes.shape[0]
    reward = torch.zeros(total, device=decoded_boxes.device)
    diag = {"ap75_tp": 0, "ap50_tp": 0, "high_fp": 0, "duplicate": 0, "matched": 0}

    if len(gt_boxes) == 0:
        # All boxes are FP (no GT in image)
        for i in range(total):
            reward[i] = -1.0  # high-conf FP (no GT to match)
            diag["high_fp"] += 1
        return reward, diag

    iou_mat = box_iou(decoded_boxes, gt_boxes)  # (N*G, num_gt)
    best_iou, best_gt = iou_mat.max(dim=1)

    # Track which GT is already matched (greedy by IoU)
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=decoded_boxes.device)
    gt_best_box = [-1] * len(gt_boxes)

    # Sort boxes by IoU descending for greedy matching
    sorted_idx = best_iou.argsort(descending=True)

    for idx in sorted_idx.tolist():
        iou = best_iou[idx].item()
        gt_idx = best_gt[idx].item()

        if iou >= 0.75:
            if not gt_matched[gt_idx]:
                reward[idx] = 1.0
                gt_matched[gt_idx] = True
                gt_best_box[gt_idx] = idx
                diag["ap75_tp"] += 1
                diag["matched"] += 1
            else:
                # Duplicate: high IoU but GT already taken
                reward[idx] = -0.5
                diag["duplicate"] += 1
        elif iou >= 0.5:
            if not gt_matched[gt_idx]:
                reward[idx] = 0.3
                gt_matched[gt_idx] = True
                gt_best_box[gt_idx] = idx
                diag["ap50_tp"] += 1
                diag["matched"] += 1
            else:
                reward[idx] = -0.5
                diag["duplicate"] += 1
        else:
            # Low IoU: FP
            reward[idx] = -1.0
            diag["high_fp"] += 1

    return reward, diag


def compute_nms_aware_reward(q_pred, decoded_boxes, gt_boxes, iou_r):
    """NMS-aware reward per image: reward - duplicate_penalty - FP_penalty."""
    N, G = q_pred.shape
    reward = iou_r.clone()
    reward_flat = reward.flatten()
    if len(gt_boxes) > 0:
        iou_mat = box_iou(decoded_boxes, gt_boxes)
        best_iou, best_gt = iou_mat.max(dim=1)
        gt_best = {}
        for i in range(len(gt_boxes)):
            mask = best_gt == i
            if mask.any():
                gt_best[i] = mask.nonzero()[best_iou[mask].argmax()].item()
        for gt_i, best_idx in gt_best.items():
            dup_mask = (best_gt == gt_i) & (torch.arange(len(best_gt), device=DEV) != best_idx)
            reward_flat[dup_mask] = reward_flat[dup_mask] - 0.3 * best_iou[dup_mask]
        fp_mask = best_iou < 0.3
        reward_flat[fp_mask] = reward_flat[fp_mask] - 0.5
    return reward_flat.view(N, G).clamp(min=-2.0)


def compute_rewards_per_image(mode, q_pred, decoded_cat, img_map, iou_r, tgts_t, delta_list):
    """Compute reward per image for modes that need per-image GT alignment."""
    offset = q_pred.shape[0]
    all_rewards = torch.zeros_like(q_pred)
    all_diags = {"ap75_tp": 0, "ap50_tp": 0, "high_fp": 0, "duplicate": 0}

    for i_img in range(len(tgts_t)):
        mask = torch.tensor([img_map[pi * G_SAMPLES] == i_img for pi in range(offset)], device=DEV)
        n_img = mask.sum().item()
        if n_img == 0: continue
        idx = mask.nonzero(as_tuple=True)[0]

        if mode == "ap75_event" or mode == "fft_residual":
            dec_img = torch.cat([decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES] for pi in idx.tolist()], dim=0)
            reward_e, rd = compute_ap75_reward(dec_img, tgts_t[i_img]["boxes"])
            for k in all_diags: all_diags[k] += rd[k]
            for j, pi in enumerate(idx.tolist()):
                all_rewards[pi] = reward_e[j * G_SAMPLES:(j + 1) * G_SAMPLES]

        elif mode == "nms_aware":
            pi = idx[0].item()  # per-image, just one proposal batch
            dec_img = decoded_cat[pi * G_SAMPLES:(pi + 1) * G_SAMPLES]
            all_rewards[pi:pi + 1] = compute_nms_aware_reward(
                q_pred[pi:pi + 1], dec_img, tgts_t[i_img]["boxes"], iou_r[pi:pi + 1])

    return all_rewards, all_diags


def run_one(cfg_name, mode, seed):
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
    rd = ensure_run_dir(run_name)

    # Group-specific setup
    needs_verifier = mode in ("local_iou", "image_adv", "nms_aware")
    is_residual = mode == "fft_residual"
    is_ap75_event = mode == "ap75_event"

    vrf = None  # main verifier (local_iou/image_adv/nms_aware)
    vrf_base = None  # ROI+Geo baseline (fft_residual)
    vrf_fft = None  # FFT residual (fft_residual)

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
                s = torch.full_like(mu, 0.1, requires_grad=False)
                deltas = mu.detach().unsqueeze(1) + s.unsqueeze(1) * torch.randn(N_rf, G_SAMPLES, 4, device=DEV)
                log_probs = glp(deltas, mu, s)

                sp_cat = torch.cat(sp_raw, dim=0); N = min(N_rf, sp_cat.shape[0])
                mu = mu[:N]; deltas = deltas[:N]; log_probs = log_probs[:N]

                # Build per-image structures
                box_list, delta_list, img_map = [], [], []
                offset = 0
                for i_img, p_img in enumerate(sp_raw):
                    n_a = min(p_img.shape[0], N - offset)
                    if n_a <= 0: break
                    box_list.append(sp_cat[offset:offset + n_a])
                    delta_list.append(deltas[offset:offset + n_a].reshape(-1, 4))
                    img_map.extend([i_img] * (n_a * G_SAMPLES))
                    offset += n_a

                # Decode
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

                # --- Compute reward per group ---
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

                    if mode == "local_iou":
                        q_norm = (q_pred - q_pred.mean(dim=1, keepdim=True)) / (q_pred.std(dim=1, keepdim=True).clamp_min(1e-6))
                    elif mode == "image_adv":
                        q_flat = q_pred.flatten()
                        q_norm = ((q_flat - q_flat.mean()) / (q_flat.std().clamp_min(1e-6))).view(offset, G_SAMPLES)
                    elif mode == "nms_aware":
                        reward_nms, _ = compute_rewards_per_image("nms_aware", q_pred, decoded_cat, img_map, iou_r, tgts_t, delta_list)
                        q_norm = (reward_nms - reward_nms.mean(dim=1, keepdim=True)) / (reward_nms.std(dim=1, keepdim=True).clamp_min(1e-6))

                elif is_ap75_event:
                    # D: AP75-event discrete reward (no verifier, per-image)
                    reward_e, rd_ap = compute_rewards_per_image("ap75_event", iou_r, decoded_cat, img_map, iou_r, tgts_t, delta_list)
                    for k in ["ap75_tp", "ap50_tp", "high_fp", "duplicate"]:
                        diag[k].append(rd_ap[k])
                    q_norm = (reward_e - reward_e.mean(dim=1, keepdim=True)) / (reward_e.std(dim=1, keepdim=True).clamp_min(1e-6))

                elif is_residual:
                    # E: AP75-event base reward + FFT residual
                    reward_e, rd_ap = compute_rewards_per_image("fft_residual", iou_r, decoded_cat, img_map, iou_r, tgts_t, delta_list)
                    for k in ["ap75_tp", "ap50_tp", "high_fp", "duplicate"]:
                        diag[k].append(rd_ap[k])

                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    with torch.no_grad():
                        pooled = box_pool(fpn, decoded_list, image_shapes)
                    roi_flat = pooled.flatten(1)
                    fft_f = extract_perchan_fft(pooled)
                    fft_shuf = fft_f[torch.randperm(fft_f.shape[0], generator=rng_shuf, device=DEV)]
                    geo = torch.cat([
                        torch.stack([
                            (b[:, 0] + b[:, 2]) / (2 * image_shapes[i_img][1]),
                            (b[:, 1] + b[:, 3]) / (2 * image_shapes[i_img][0]),
                            torch.log((b[:, 2] - b[:, 0]).clamp_min(1)),
                            torch.log((b[:, 3] - b[:, 1]).clamp_min(1)),
                        ], dim=1) for i_img, b in enumerate(decoded_list)], dim=0)

                    if vrf_base is None:
                        roi_dim = pooled.shape[1] * pooled.shape[2] * pooled.shape[3]
                        vrf_base = BaseVerifier(roi_dim).to(DEV)
                        vrf_fft = FFTResidualVerifier(fft_f.shape[1]).to(DEV)
                        params = [p for p in list(model.parameters()) + list(vrf_base.parameters()) + list(vrf_fft.parameters()) if p.requires_grad]
                        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

                    q_base = vrf_base(roi_flat, geo).view(offset, G_SAMPLES)
                    q_fft_real = vrf_fft(fft_f).view(offset, G_SAMPLES)
                    q_fft_shuf = vrf_fft(fft_shuf).view(offset, G_SAMPLES)

                    # Base loss: predict AP75 reward
                    base_target = reward_e.clamp(-1, 1)
                    vloss = F.mse_loss(q_base, base_target.detach())
                    # Residual target
                    residual_target = (base_target - q_base.detach()).clamp(-1, 1)
                    vloss = vloss + F.mse_loss(q_fft_real, residual_target.detach())
                    # FFT contrastive constraint (high IoU proposals only)
                    high_iou_mask = iou_r.max(dim=1).values > 0.5
                    if high_iou_mask.any():
                        vloss = vloss + 0.1 * F.relu(0.1 - (q_fft_real[high_iou_mask].mean() - q_fft_shuf[high_iou_mask].mean()))

                    # Final reward = AP75_event + λ * FFT residual
                    final_reward = reward_e + FFT_RESIDUAL_WEIGHT * q_fft_real.detach()
                    q_norm = (final_reward - final_reward.mean(dim=1, keepdim=True)) / (final_reward.std(dim=1, keepdim=True).clamp_min(1e-6))

                    diag["q_ious"].extend(list(zip(q_fft_real.flatten().tolist(), iou_r.flatten().tolist())))
                    diag["q_std"].append(q_fft_real.std().item())

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

        # --- Epoch diagnostics ---
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
    save_json(em, rd / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    configs = {
        "local_iou": "local_iou", "image_adv": "image_adv", "nms_aware": "nms_aware",
        "ap75_event": "ap75_event", "fft_residual": "fft_residual",
    }
    for cfg, mode in configs.items():
        for s in SEEDS:
            r = run_one(cfg, mode, s)
            all_results.append(r)

    print("\n## Plan 2.74 v2 Reward-Aligned RLVR")
    print(f"  {'Config':<15s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'AP50':>8s} {'Recall':>8s} {'Precision':>10s} {'q_corr':>8s} {'AP75tp':>8s} {'dup':>6s} {'hFP':>6s}")
    for r in all_results:
        hf = r["history"][-1]
        print(f"  {r['config']:<15s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {hf['val_ap50']:8.4f} {hf.get('recall', 0):8.4f} {hf.get('precision', 0):10.4f} {r.get('q_iou_corr_final', 0):8.4f} {hf.get('ap75_tp', 0):8d} {hf.get('duplicate_fp', 0):6d} {hf.get('high_conf_fp', 0):6d}")

    for cfg in configs:
        vals = [r for r in all_results if r["config"] == cfg]
        bv = [r["best_ap75"] for r in vals]
        fv = [r["ap75"] for r in vals]
        rl_gn = [np.mean([x["total_grad_norm"] for x in r["history"]]) for r in vals]
        print(f"  {cfg}: bestAP75={np.mean(bv):.4f}±{np.std(bv):.4f}  finalAP75={np.mean(fv):.4f}  avg_tgn={np.mean(rl_gn):.6f}")

    def judge(name_a, name_b, threshold=0.01):
        va = [r["best_ap75"] for r in all_results if r["config"] == name_a]
        vb = [r["best_ap75"] for r in all_results if r["config"] == name_b]
        if not va or not vb: return
        delta = np.mean(va) - np.mean(vb)
        wins = sum(1 for i in range(3) if va[i] > vb[i])
        status = "PASS" if delta > threshold and wins >= 2 else "FAIL"
        print(f"\n  {name_a} vs {name_b}: Δ={delta:+.4f} wins={wins}/3 → {status}")

    print("\n## Key judgments")
    judge("image_adv", "local_iou")       # Does image-level advantage beat local?
    judge("nms_aware", "local_iou")       # Does NMS-aware beat local?
    judge("ap75_event", "local_iou")      # Does AP75-event beat IoU?
    judge("fft_residual", "ap75_event")   # Does FFT residual add beyond AP75?
    judge("ap75_event", "image_adv")      # AP75-event vs image-level advantage
