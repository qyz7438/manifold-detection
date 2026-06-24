"""Plan 2.89: Multi-step Feature-Driven Bbox Refinement.

Key insight: single-step PG has no "dynamics" to amplify small action differences.
Solution: iterate refinement — each step pools ROI features at the current box,
uses them to predict the next action, creating a chain of T=3 refinements.

The intermediate ROI features CHANGE with each box shift → provide actionable
gradient signal that single-step reward-to-delta chain cannot.
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
G_SAMPLES = 4; T_STEPS = 3; EPOCHS = 8; SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05; KL_WEIGHT = 0.1; ENERGY_BETA = 0.02
HEAD_LR = 0.001; BODY_LR = 0.0001; IOU_LO = 0.3; IOU_HI = 0.55

# Action set: shifts + scales
def build_actions():
    scales = [0.02, 0.05, 0.10]
    acts = []
    for s in scales:
        acts.extend([(s,0,0,0),(-s,0,0,0),(0,s,0,0),(0,-s,0,0)])
    for s in scales:
        acts.extend([(0,0,s,0),(0,0,-s,0)])
    for s in scales:
        acts.extend([(0,0,0,s),(0,0,0,-s)])
    return torch.tensor(acts, dtype=torch.float32)

ACTIONS = build_actions()  # (24, 4)
N_ACTIONS = len(ACTIONS)

class RefinementStep(nn.Module):
    """Takes pooled ROI features (N,256,7,7), outputs action logits (N,24)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, N_ACTIONS)
        )
    def forward(self, roi): return self.net(roi)

def apply_actions_batch(boxes, action_indices):
    a = ACTIONS.to(boxes.device)[action_indices]  # (M, 4)
    w = boxes[:,2]-boxes[:,0]; h = boxes[:,3]-boxes[:,1]
    cx = boxes[:,0]+0.5*w; cy = boxes[:,1]+0.5*h
    new_cx = cx + a[:,0]*w; new_cy = cy + a[:,1]*h
    new_w = torch.clamp(w*(1.0+a[:,2]), min=1); new_h = torch.clamp(h*(1.0+a[:,3]), min=1)
    x1 = new_cx-0.5*new_w; y1 = new_cy-0.5*new_h
    x2 = new_cx+0.5*new_w; y2 = new_cy+0.5*new_h
    return torch.stack([x1,y1,x2,y2], dim=1).clamp(min=0)

def compute_energy(fft_f):
    return fft_f.mean(dim=1)

def grpo_advantage(reward):
    r_mean = reward.mean(dim=1, keepdim=True); r_std = reward.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (reward - r_mean) / r_std

def ext_fft(x):
    C=x.shape[1]; H,W=x.shape[-2],x.shape[-1]
    fft = torch.fft.rfft2(x,dim=(-2,-1),norm="ortho"); amp=torch.abs(fft)
    freq_h=torch.fft.fftfreq(H,device=x.device); freq_w=torch.fft.rfftfreq(W,device=x.device)
    Y,X=torch.meshgrid(freq_h,freq_w,indexing='ij'); r=torch.sqrt(X**2+Y**2); R=r.max().clamp_min(1e-6); rn=r/R
    lo=(rn<=0.3).float(); md=((rn>0.3)&(rn<=0.7)).float(); hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2); am=(amp*md).flatten(2).sum(2); ah=(amp*hi).flatten(2).sum(2)
    return al/(al+am+ah+1e-8)

def unfreeze_rlvr(model):
    for p in model.backbone.body.parameters(): p.requires_grad = False
    if hasattr(model.backbone, 'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad = True
    for p in model.rpn.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

def build_opt(model, extra_modules=None):
    body_params=[]; head_params=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head_params.append(p)
        else: body_params.append(p)
    extra = []
    if extra_modules:
        for m in extra_modules:
            extra.extend(m.parameters())
    return torch.optim.SGD([{'params':body_params,'lr':BODY_LR},{'params':head_params,'lr':HEAD_LR},{'params':extra,'lr':HEAD_LR}],lr=HEAD_LR,momentum=0.9,weight_decay=0.0005)

def bl():
    return build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":4}})
def bm():
    return build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}})

@torch.no_grad()
def refine_inference(box_pool, fpn, boxes, ref_steps, img_shape):
    """Apply T-step refinement to predicted boxes at inference time."""
    current = boxes  # (M, 4)
    for t in range(T_STEPS):
        roi = box_pool(fpn, [current], [img_shape])  # (M, 256, 7, 7)
        logits = ref_steps[t](roi)  # (M, 24)
        actions = logits.argmax(dim=-1)  # deterministic at inference
        current = apply_actions_batch(current, actions)
    return current

@torch.no_grad()
def ev(model, vl, ref_steps=None, fpn_dict=None, box_pool=None):
    model.eval()
    if ref_steps: ref_steps.eval()
    ps, ts = [], []
    for img, tgt in vl:
        im = [i.to(DEV) for i in img]
        out = model(im)
        if ref_steps is not None:
            if fpn_dict is not None: fpn_dict.clear()
            with torch.no_grad():
                _ = model(im, [{k: v.to(DEV) for k, v in t.items()} for t in tgt])
            fpn = fpn_dict.get("f") if fpn_dict else None
            if fpn is not None:
                for i_img in range(len(out)):
                    boxes = out[i_img]["boxes"]
                    if len(boxes) > 0:
                        refined = refine_inference(box_pool, fpn, boxes.to(DEV), ref_steps, (im[i_img].shape[-2], im[i_img].shape[-1]))
                        out[i_img] = {k: v for k, v in out[i_img].items()}
                        out[i_img]["boxes"] = refined.cpu()
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)

def run_one(cfg_name, mode, seed):
    run_name = f"round289_{cfg_name}_s{seed}"; set_seed(seed)
    model = bm().to(DEV); ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"]); unfreeze_rlvr(model)
    box_pool = model.roi_heads.box_roi_pool

    # Create refinement chain
    is_det = mode == "det_only_unf"
    ref_steps = nn.ModuleList([RefinementStep().to(DEV) for _ in range(T_STEPS)]) if not is_det else None
    if ref_steps: ref_steps.train()

    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    tl, vl = bl(); run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_energy = mode == "multi_step_energy"; is_shuffle = mode == "multi_step_shuffle"
    rng_shuf = torch.Generator(device=DEV).manual_seed(seed+9999)
    opt = build_opt(model, ref_steps)
    bw_base = baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    bb_base = baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    h = []; best_ap75 = -1.0
    diag = {"reward_std":[], "en_gap":[]}

    for ep in range(1, EPOCHS+1):
        model.train()
        if ref_steps: ref_steps.train()
        td, trl, tkl = 0.0, 0.0, 0.0

        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k:v.to(DEV) for k,v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()
            image_shapes = [(im.shape[-2],im.shape[-1]) for im in imgs_d]

            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
            rl = torch.tensor(0.0, device=DEV); kl_loss = torch.tensor(0.0, device=DEV)

            if not is_det and ref_steps is not None and rf is not None and sp_raw is not None and fpn is not None and rf.shape[0] > 0:
                N = rf.shape[0]
                kl_loss = KL_WEIGHT*((model.roi_heads.box_predictor.bbox_pred.weight-bw_base).pow(2).sum()+(model.roi_heads.box_predictor.bbox_pred.bias-bb_base).pow(2).sum())

                # Base boxes from detector
                sp_cat = torch.cat(sp_raw, dim=0)[:N]
                bf = model.roi_heads.box_head(rf)
                mu = model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
                bw=sp_cat[:,2]-sp_cat[:,0]; bh=sp_cat[:,3]-sp_cat[:,1]; bcx=sp_cat[:,0]+0.5*bw; bcy=sp_cat[:,1]+0.5*bh
                dx_b=mu[:,0]/10.0; dy_b=mu[:,1]/10.0; dw_b=mu[:,2]/5.0; dh_b=mu[:,3]/5.0
                base_boxes = torch.stack([dx_b*bw+bcx-0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy-0.5*torch.exp(dh_b)*bh,dx_b*bw+bcx+0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy+0.5*torch.exp(dh_b)*bh],dim=1).clamp(min=0)

                # Expand per-proposal: replicate each proposal G times
                base_exp = base_boxes.repeat_interleave(G_SAMPLES, dim=0)  # (N*G, 4)
                rg = G_SAMPLES
                total_log_prob = torch.zeros(N, G_SAMPLES, device=DEV)

                # Baseline IoU (before refinement) for delta reward
                iou_base = torch.zeros(N, G_SAMPLES, device=DEV)
                off_p = 0
                for i_img, p_img in enumerate(sp_raw):
                    np_i = min(p_img.shape[0], N-off_p)
                    if np_i <= 0: break
                    idx_s = off_p*G_SAMPLES; idx_e = (off_p+np_i)*G_SAMPLES
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt)>0:
                        ious = box_iou(base_exp[idx_s:idx_e], gt)
                        iou_base[off_p:off_p+np_i] = ious.max(dim=1).values.view(np_i, G_SAMPLES)
                    off_p += np_i

                # T-step refinement chain
                current_boxes = base_exp  # (N*G, 4)
                for t in range(T_STEPS):
                    # Pool ROI features at current box positions (per-image)
                    roi_chunks = []
                    off_p = 0
                    for i_img, p_img in enumerate(sp_raw):
                        np_i = min(p_img.shape[0], N-off_p)
                        if np_i <= 0: break
                        idx_s = off_p*G_SAMPLES; idx_e = (off_p+np_i)*G_SAMPLES
                        roi_chunks.append(current_boxes[idx_s:idx_e])
                        off_p += np_i

                    all_roi = []
                    for i_img, boxes_i in enumerate(roi_chunks):
                        if len(boxes_i)==0: continue
                        all_roi.append(box_pool(fpn, [boxes_i], [image_shapes[i_img]]))
                    roi_cat = torch.cat(all_roi, dim=0)  # (N*G, 256, 7, 7)

                    # Predict actions
                    action_logits = ref_steps[t](roi_cat)  # (N*G, 24)
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_dist = torch.distributions.Categorical(probs=action_probs)
                    actions = action_dist.sample()  # (N*G,)
                    lp = action_dist.log_prob(actions).view(N, G_SAMPLES)  # (N, G)
                    total_log_prob = total_log_prob + lp

                    # Apply actions
                    current_boxes = apply_actions_batch(current_boxes, actions)

                # Final reward: IoU-based
                iou_r = torch.zeros(N, G_SAMPLES, device=DEV)
                off_p = 0
                for i_img, p_img in enumerate(sp_raw):
                    np_i = min(p_img.shape[0], N-off_p)
                    if np_i <= 0: break
                    idx_s = off_p*G_SAMPLES; idx_e = (off_p+np_i)*G_SAMPLES
                    boxes_i = current_boxes[idx_s:idx_e]
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt)>0:
                        ious = box_iou(boxes_i, gt)
                        iou_r[off_p:off_p+np_i] = ious.max(dim=1).values.view(np_i, G_SAMPLES)
                    off_p += np_i

                reward_img = iou_r - iou_base  # delta IoU: improvement over baseline

                # Energy penalty on final boxes
                if is_energy or is_shuffle:
                    en_chunks = []
                    off_p = 0
                    for i_img, p_img in enumerate(sp_raw):
                        np_i = min(p_img.shape[0], N-off_p)
                        if np_i <= 0: break
                        idx_s = off_p*G_SAMPLES; idx_e = (off_p+np_i)*G_SAMPLES
                        boxes_i = current_boxes[idx_s:idx_e]
                        with torch.no_grad():
                            pooled_i = box_pool(fpn, [boxes_i], [image_shapes[i_img]])
                        fft_i = ext_fft(pooled_i)
                        en_i = fft_i.mean(dim=1)
                        en_chunks.append(en_i)
                        off_p += np_i
                    energy = torch.cat(en_chunks).view(N, G_SAMPLES)

                    if is_shuffle:
                        energy = energy.reshape(-1)[torch.randperm(N*G_SAMPLES, generator=rng_shuf, device=DEV)].view(N, G_SAMPLES)

                    en_pen = -torch.sigmoid(15*(energy-0.5)) * ENERGY_BETA
                    gmax = iou_r.max(dim=1).values
                    bmask = ((gmax>=IOU_LO)&(gmax<IOU_HI)).unsqueeze(1).float()
                    tp_mask = gmax>=0.5; fn_mask = gmax<0.5
                    if tp_mask.any() and fn_mask.any():
                        diag["en_gap"].append((energy[tp_mask].mean()-energy[fn_mask].mean()).item())

                # GRPO on R_loc only, then add energy bias AFTER (round283 fix)
                adv = grpo_advantage(reward_img)
                if is_energy or is_shuffle:
                    adv = adv + en_pen * bmask
                diag["reward_std"].append(adv.std().item())
                soft_w = iou_r.max(dim=1).values.clamp(0,1).unsqueeze(1)
                rl = -(adv.detach() * total_log_prob * soft_w).mean()

            loss = det + RL_WEIGHT*rl + kl_loss; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            td += det.item(); trl += rl.item(); tkl += kl_loss.item()

        em = ev(model, vl, ref_steps if not is_det else None, fpn_feats if not is_det else None, box_pool if not is_det else None)
        rs_m = np.mean(diag["reward_std"]) if diag["reward_std"] else 0
        eg_m = np.mean(diag["en_gap"]) if diag["en_gap"] else 0
        row = {"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"],"pre":em.get("precision",0),"rec":em.get("recall",0),"ece":em.get("ece",0),"reward_std":float(rs_m),"en_gap":float(eg_m),"det_loss":td,"rl_loss":trl,"kl_loss":tkl}
        h.append(row); print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} r_std={rs_m:.4f} en_gap={eg_m:.4f}")
        if em["ap75"]>best_ap75: best_ap75=em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name":run_name,"config":cfg_name,"mode":mode,"seed":seed,"epochs":len(h),"best_ap50":best_h["val_ap50"],"best_ap75":best_ap75,"history":h,"git_hash":GIT})
    save_json(em, run_dir/"eval_metrics.json"); return em

if __name__ == "__main__":
    all_results=[]
    for cfg, mode in [("det_only_unf","det_only_unf"),("multi_step_energy","multi_step_energy"),("multi_step_shuffle","multi_step_shuffle")]:
        for s in SEEDS: r=run_one(cfg,mode,s); all_results.append(r)
    print("\n## Plan 2.89 Multi-step Feature-Driven Refinement")
    for r in all_results:
        bh=max(r["history"],key=lambda x:x["val_ap75"])
        print(f"  {r['config']:<20s} s{r['seed']} best={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only_unf","multi_step_energy","multi_step_shuffle"]:
        vals=[r for r in all_results if r["config"]==cfg]
        if vals: print(f"  {cfg}: {np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
