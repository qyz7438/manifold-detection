"""Plan 2.88: Discrete Action GRPO — discretize bbox refinement into an action set.

Key insight: continuous delta perturbations (sigma=0.1) produce near-zero reward variance.
All successful RL+detector papers use DISCRETE actions. This bridges the gap.

Action set: 8 discrete box transforms at 3 scales each = 24 actions
  Shift: left/right/up/down × {0.02, 0.05, 0.10} fraction of box size
  Scale: expand/shrink width × {0.05, 0.10, 0.20} fraction
  Aspect: widen/narrow × {0.05, 0.10, 0.20} fraction

Action head: box_head → Linear(256, 24) → Categorical distribution
GRPO: sample G actions, compute reward, update via PG.
"""
import sys, json, subprocess, math, copy, shutil
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
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
DEV = "cuda"; CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G_SAMPLES = 8; EPOCHS = 8; SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05; KL_WEIGHT = 0.1; ENERGY_BETA = 0.02
HEAD_LR = 0.001; BODY_LR = 0.0001; IOU_LO = 0.3; IOU_HI = 0.55

# Define discrete action set: (dx, dy, dw, dh) in fraction of box size
def build_actions():
    scales = [0.02, 0.05, 0.10]
    acts = []
    # Shift actions
    for s in scales:
        acts.extend([(s, 0, 0, 0), (-s, 0, 0, 0), (0, s, 0, 0), (0, -s, 0, 0)])
    # Scale width
    for s in scales:
        acts.extend([(0, 0, s, 0), (0, 0, -s, 0)])
    # Scale height
    for s in scales:
        acts.extend([(0, 0, 0, s), (0, 0, 0, -s)])
    return torch.tensor(acts, dtype=torch.float32)  # (24, 4)

ACTIONS = build_actions()  # (24, 4)
N_ACTIONS = len(ACTIONS)

class DiscreteActionHead(nn.Module):
    """Predict action logits from box_head features (1024-dim)."""
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        return self.net(x)  # (N, n_actions)

def apply_actions_batch(base_boxes, action_indices):
    """Vectorized: apply discrete actions to base boxes.
    base_boxes: (M, 4) in [x1,y1,x2,y2]
    action_indices: (M,) int tensor
    Returns: (M, 4) transformed boxes.
    """
    a = ACTIONS.to(base_boxes.device)[action_indices]  # (M, 4)
    w = base_boxes[:, 2] - base_boxes[:, 0]; h = base_boxes[:, 3] - base_boxes[:, 1]
    cx = base_boxes[:, 0] + 0.5*w; cy = base_boxes[:, 1] + 0.5*h

    new_cx = cx + a[:, 0] * w; new_cy = cy + a[:, 1] * h
    new_w = torch.clamp(w * (1.0 + a[:, 2]), min=1)
    new_h = torch.clamp(h * (1.0 + a[:, 3]), min=1)

    x1 = new_cx - 0.5*new_w; y1 = new_cy - 0.5*new_h
    x2 = new_cx + 0.5*new_w; y2 = new_cy + 0.5*new_h
    return torch.stack([x1, y1, x2, y2], dim=1).clamp(min=0)

def compute_loc_reward(iou):
    r = torch.zeros_like(iou); r[iou >= 0.75] = 1.0; r[(iou>=0.5)&(iou<0.75)] = 0.3; r[iou<0.5] = -0.5
    return r

def extract_perchan_fft(x):
    C = x.shape[1]; H,W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2,-1), norm="ortho"); amp = torch.abs(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device); freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y,X = torch.meshgrid(freq_h, freq_w, indexing='ij'); r = torch.sqrt(X**2+Y**2)
    R = r.max().clamp_min(1e-6); rn = r/R
    lo = (rn<=0.3).float(); md = ((rn>0.3)&(rn<=0.7)).float(); hi = (rn>0.7).float()
    a_lo = (amp*lo).flatten(2).sum(2); a_md = (amp*md).flatten(2).sum(2); a_hi = (amp*hi).flatten(2).sum(2)
    return a_lo/(a_lo+a_md+a_hi+1e-8)

def compute_energy(fft_f): return fft_f.mean(dim=1)

def grpo_advantage(reward):
    r_mean = reward.mean(dim=1, keepdim=True); r_std = reward.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (reward - r_mean) / r_std

def unfreeze_rlvr(model):
    for p in model.backbone.body.parameters(): p.requires_grad = False
    if hasattr(model.backbone, 'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad = True
    for p in model.rpn.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

def build_opt(model, action_head=None):
    body_params = []; head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head_params.append(p)
        else: body_params.append(p)
    extra = list(action_head.parameters()) if action_head else []
    return torch.optim.SGD([{'params': body_params, 'lr': BODY_LR}, {'params': head_params, 'lr': HEAD_LR}, {'params': extra, 'lr': HEAD_LR}], lr=HEAD_LR, momentum=0.9, weight_decay=0.0005)

def bl():
    return build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 4}})

def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})

@torch.no_grad()
def ev(model, vl):
    model.eval(); ps, ts = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)

def run_one(cfg_name, mode, seed):
    run_name = f"round288_{cfg_name}_s{seed}"; set_seed(seed)
    model = bm().to(DEV); ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"]); unfreeze_rlvr(model)
    box_pool = model.roi_heads.box_roi_pool

    # Discrete action head
    action_head = DiscreteActionHead().to(DEV) if mode != "det_only_unf" else None
    if action_head: action_head.train()

    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    tl, vl = bl(); run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only_unf"; use_energy = mode == "discrete_energy"; is_shuffle = mode == "discrete_shuffle"
    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 9999)
    opt = build_opt(model, action_head)
    bw_base = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    bb_base = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    h = []; best_ap75 = -1.0
    diag = {"reward_std": [], "action_entropy": [], "energy_gap": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        if action_head: action_head.train()
        td, trl, tkl = 0.0, 0.0, 0.0

        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV); kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and action_head is not None and rf is not None and sp_raw is not None and fpn is not None and rf.shape[0] > 0:
                N = rf.shape[0]
                kl_loss = KL_WEIGHT * ((model.roi_heads.box_predictor.bbox_pred.weight - bw_base).pow(2).sum() + (model.roi_heads.box_predictor.bbox_pred.bias - bb_base).pow(2).sum())

                # Get base boxes from detector
                sp_cat = torch.cat(sp_raw, dim=0)[:N]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]

                # Decode base boxes
                bw = sp_cat[:,2]-sp_cat[:,0]; bh = sp_cat[:,3]-sp_cat[:,1]; bcx = sp_cat[:,0]+0.5*bw; bcy = sp_cat[:,1]+0.5*bh
                dx_b = mu[:,0]/10.0; dy_b = mu[:,1]/10.0; dw_b = mu[:,2]/5.0; dh_b = mu[:,3]/5.0
                base_boxes = torch.stack([dx_b*bw+bcx-0.5*torch.exp(dw_b)*bw, dy_b*bh+bcy-0.5*torch.exp(dh_b)*bh, dx_b*bw+bcx+0.5*torch.exp(dw_b)*bw, dy_b*bh+bcy+0.5*torch.exp(dh_b)*bh], dim=1).clamp(min=0)  # (N, 4)

                # Action head: predict action distribution from box_head output
                action_logits = action_head(bf)  # (N, 24), bf is box_head output
                action_probs = F.softmax(action_logits, dim=-1)  # (N, 24)

                # Sample G actions per proposal
                action_dist = torch.distributions.Categorical(probs=action_probs)
                sampled_actions = action_dist.sample((G_SAMPLES,))  # (G, N)
                # log_prob per sample
                action_log_probs = torch.stack([action_dist.log_prob(sampled_actions[g, :]) for g in range(G_SAMPLES)], dim=1)  # (N, G)

                # Vectorized: apply actions to all proposals at once
                base_boxes_exp = base_boxes.repeat_interleave(G_SAMPLES, dim=0)  # (N*G, 4)
                action_indices = sampled_actions.transpose(0, 1).reshape(-1)  # (N*G,)
                all_boxes_t = apply_actions_batch(base_boxes_exp, action_indices)  # (N*G, 4)

                # Compute IoU per image (vectorized per image)
                iou_r = torch.zeros(N, G_SAMPLES, device=DEV)
                boxes_per_image = []
                off_p = 0
                for i_img, p_img in enumerate(sp_raw):
                    np_i = min(p_img.shape[0], N - off_p)
                    if np_i <= 0: break
                    idx_start = off_p * G_SAMPLES
                    idx_end = (off_p + np_i) * G_SAMPLES
                    boxes_i = all_boxes_t[idx_start:idx_end]
                    boxes_per_image.append(boxes_i)
                    gt_i = tgts_t[i_img]["boxes"]
                    if len(gt_i) > 0:
                        ious = box_iou(boxes_i, gt_i)  # (np_i*G, #GT)
                        iou_r[off_p:off_p+np_i] = ious.max(dim=1).values.view(np_i, G_SAMPLES)
                    off_p += np_i

                reward_img = 2 * iou_r - 1  # (N, G), continuous IoU -> [-1, 1]

                # Energy penalty — per-image pooling (reuse boxes_per_image)
                if use_energy or is_shuffle:
                    image_shapes = [(im.shape[-2], im.shape[-1]) for im in imgs_d]
                    energy_chunks = []
                    for i_img, boxes_i in enumerate(boxes_per_image):
                        if len(boxes_i) == 0: continue
                        pooled_i = box_pool(fpn, [boxes_i], [image_shapes[i_img]])
                        fft_i = extract_perchan_fft(pooled_i)
                        en_i = compute_energy(fft_i)
                        energy_chunks.append(en_i)
                    energy_flat = torch.cat(energy_chunks)  # (N*G,)

                    if is_shuffle:
                        perm = torch.randperm(energy_flat.shape[0], generator=rng_shuf, device=DEV)
                        energy_flat = energy_flat[perm]

                    energy = energy_flat.view(N, G_SAMPLES)  # (N, G)
                    energy_pen = -torch.sigmoid(15*(energy - 0.5)) * ENERGY_BETA
                    group_max_iou = iou_r.max(dim=1).values
                    border_mask = ((group_max_iou >= IOU_LO) & (group_max_iou < IOU_HI)).unsqueeze(1).float()
                    reward_img = reward_img + energy_pen * border_mask

                    tp_mask = group_max_iou >= 0.5; fn_mask = group_max_iou < 0.5
                    if tp_mask.any() and fn_mask.any():
                        diag["energy_gap"].append((energy[tp_mask].mean() - energy[fn_mask].mean()).item())

                # GRPO advantage
                adv = grpo_advantage(reward_img)  # (N, G)
                diag["reward_std"].append(adv.std().item())
                diag["action_entropy"].append(action_dist.entropy().mean().item())

                # PG loss with advantage
                soft_w = iou_r.max(dim=1).values.clamp(0, 1).unsqueeze(1)
                rl = -(adv.detach() * action_log_probs * soft_w).mean()

            loss = det + RL_WEIGHT * rl + kl_loss; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            td += det.item(); trl += rl.item(); tkl += kl_loss.item()

        em = ev(model, vl)
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0; ae_m = np.mean(diag["action_entropy"]) if diag["action_entropy"] else 0; eg_m = np.mean(diag["energy_gap"]) if diag["energy_gap"] else 0
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"], "pre": em.get("precision",0), "rec": em.get("recall",0), "ece": em.get("ece",0), "reward_std": float(rs_m), "action_entropy": float(ae_m), "energy_gap": float(eg_m), "det_loss": td, "rl_loss": trl, "kl_loss": tkl}
        h.append(row); print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} r_std={rs_m:.4f} entropy={ae_m:.4f} en_gap={eg_m:.4f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed, "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75, "history": h, "git_hash": GIT})
    save_json(em, run_dir / "eval_metrics.json"); return em

if __name__ == "__main__":
    all_results = []
    for cfg, mode in [("det_only_unf","det_only_unf"), ("discrete_energy","discrete_energy"), ("discrete_shuffle","discrete_shuffle")]:
        for s in SEEDS:
            r = run_one(cfg, mode, s); all_results.append(r)

    print("\n## Plan 2.88 Discrete Action GRPO")
    for r in all_results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<18s} s{r['seed']} AP75={r['ap75']:.4f} best={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only_unf", "discrete_energy", "discrete_shuffle"]:
        vals = [r for r in all_results if r["config"] == cfg]
        if vals: print(f"  {cfg}: {np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
