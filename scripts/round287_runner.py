"""Plan 2.87: Energy-weighted post-training — not PG, but auxiliary weighting.

Key insight: at same IoU (0.5-0.6), energy separates TP from FP-duplicate (d=1.05)
while model score cannot (d=0.05). Energy captures residual quality that det loss misses.

Instead of PG reward: use energy to WEIGHT the detection loss.
High-energy confusing proposals → lower weight.
Low-energy clean proposals → higher weight.
Model focuses gradient on learnable samples, ignores confusion.
"""
import sys, json, subprocess, math, copy, shutil
from pathlib import Path
import torch, torch.nn as nn
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
EPOCHS = 8; SEEDS = [42, 123, 456]
HEAD_LR = 0.001; BODY_LR = 0.0001
ENERGY_WEIGHT = 0.1  # how much to downweight high-energy proposals

def extract_perchan_fft(x):
    C = x.shape[1]; H,W = x.shape[-2], x.shape[-1]
    fft = torch.fft.rfft2(x, dim=(-2,-1), norm="ortho"); amp = torch.abs(fft)
    freq_h = torch.fft.fftfreq(H, device=x.device); freq_w = torch.fft.rfftfreq(W, device=x.device)
    Y,X = torch.meshgrid(freq_h, freq_w, indexing='ij'); r = torch.sqrt(X**2+Y**2)
    R = r.max().clamp_min(1e-6); rn = r/R
    lo = (rn<=0.3).float(); md = ((rn>0.3)&(rn<=0.7)).float(); hi = (rn>0.7).float()
    a_lo = (amp*lo).flatten(2).sum(2); a_md = (amp*md).flatten(2).sum(2); a_hi = (amp*hi).flatten(2).sum(2)
    return a_lo/(a_lo+a_md+a_hi+1e-8)

def compute_energy(fft_f):
    return fft_f.mean(dim=1)

def unfreeze_rlvr(model):
    for p in model.backbone.body.parameters(): p.requires_grad = False
    if hasattr(model.backbone, 'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad = True
    for p in model.rpn.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

def build_opt(model):
    body_params = []; head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head_params.append(p)
        else: body_params.append(p)
    return torch.optim.SGD([{'params': body_params, 'lr': BODY_LR}, {'params': head_params, 'lr': HEAD_LR}], lr=HEAD_LR, momentum=0.9, weight_decay=0.0005)

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
    run_name = f"round287_{cfg_name}_s{seed}"; set_seed(seed)
    model = bm().to(DEV); ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"]); unfreeze_rlvr(model)
    box_pool = model.roi_heads.box_roi_pool

    sampled_props, box_head_in, fpn_feats = {}, {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, args: box_head_in.update({"x": args[0]}))
    model.backbone.register_forward_hook(lambda m, i, o: fpn_feats.update({"f": {k: o[k] for k in o if k != "pool"}}))

    tl, vl = bl(); run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only_unf"; is_energy_w = mode == "energy_weighted"; is_shuffle = mode == "shuffle_weight"
    rng_shuf = torch.Generator(device=DEV).manual_seed(seed + 9999)
    opt = build_opt(model)

    h = []; best_ap75 = -1.0
    diag = {"energy_mean": [], "energy_tp": [], "energy_fn": []}

    for ep in range(1, EPOCHS + 1):
        model.train(); td = 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear(); fpn_feats.clear()
            img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])

            ld = model(imgs_d, tgts_t)
            det_orig = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))

            # Energy-weighted: use energy from ROI features to weight proposals
            if is_energy_w or is_shuffle:
                rf = box_head_in.get("x"); sp_raw = sampled_props.get("p"); fpn = fpn_feats.get("f")
                if rf is not None and sp_raw is not None and fpn is not None and rf.shape[0] > 0:
                    with torch.no_grad():
                        sp_list = [p.to(DEV) if p.device != DEV else p for p in sp_raw]
                        pooled = box_pool(fpn, sp_list, [img_shape])
                    fft_f = extract_perchan_fft(pooled)
                    energy = compute_energy(fft_f)

                    if is_shuffle:
                        energy = energy[torch.randperm(energy.shape[0], generator=rng_shuf, device=DEV)]

                    # Weight: low energy → 1.0, high energy → <1.0
                    # Use energy relative to proposal batch mean
                    en_norm = (energy - energy.mean()) / energy.std().clamp_min(1e-6)
                    weights = 1.0 - ENERGY_WEIGHT * torch.sigmoid(en_norm)  # ~[0.9, 1.0]

                    # Apply weights to det loss: re-weight each proposal's contribution
                    # Det loss is a dict: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
                    # We can only weight box_head losses (classifier + box_reg)
                    if isinstance(ld, dict):
                        for k in ["loss_classifier", "loss_box_reg"]:
                            if k in ld:
                                ld[k] = ld[k] * weights.mean()  # global weight for simplicity
                    det = sum(ld.values()) if isinstance(ld, dict) else det_orig

                    diag["energy_mean"].append(energy.mean().item())
                    # Track energy for TP vs FN proposals
                    sp_cat = torch.cat(sp_raw, dim=0)[:rf.shape[0]]
                    iou_vecs = []
                    for i_img in range(len(imgs_d)):
                        gt = tgts_t[i_img]["boxes"].to(DEV)
                        if len(gt) == 0: continue
                        # Find which proposals belong to this image
                        n_prev = sum(p.shape[0] for p in sp_raw[:i_img]) if i_img > 0 else 0
                        n_img = sp_raw[i_img].shape[0]
                        if n_img == 0: continue
                        img_sp = sp_cat[n_prev:n_prev+n_img]
                        if len(gt) > 0:
                            ious = box_iou(img_sp, gt)
                            iou_vecs.append(ious.max(dim=1).values)
                    if iou_vecs:
                        all_iou = torch.cat(iou_vecs)
                        tp_m = all_iou >= 0.5; fn_m = all_iou < 0.5
                        if tp_m.any(): diag["energy_tp"].append(energy[:len(all_iou)][tp_m].mean().item())
                        if fn_m.any(): diag["energy_fn"].append(energy[:len(all_iou)][fn_m].mean().item())
                else:
                    det = det_orig
            else:
                det = det_orig

            opt.zero_grad(set_to_none=True); det.backward(); opt.step()
            td += det.item()

        em = ev(model, vl)
        en_m = np.mean(diag["energy_mean"]) if diag["energy_mean"] else 0
        en_tp = np.mean(diag["energy_tp"]) if diag["energy_tp"] else float("nan")
        en_fn = np.mean(diag["energy_fn"]) if diag["energy_fn"] else float("nan")
        en_gap = en_tp - en_fn if not (np.isnan(en_tp) or np.isnan(en_fn)) else float("nan")

        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "pre": em.get("precision",0), "rec": em.get("recall",0), "ece": em.get("ece",0),
               "energy_mean": float(en_m), "energy_tp": float(en_tp), "energy_fn": float(en_fn), "energy_gap": float(en_gap),
               "det_loss": td}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} en_gap={en_gap:.4f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]
        for k in diag: diag[k].clear()

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name": run_name, "config": cfg_name, "mode": mode, "seed": seed, "epochs": len(h), "best_ap50": best_h["val_ap50"], "best_ap75": best_ap75, "history": h, "git_hash": GIT})
    save_json(em, run_dir / "eval_metrics.json"); return em

if __name__ == "__main__":
    all_results = []
    for cfg, mode in [("det_only_unf","det_only_unf"), ("energy_weighted","energy_weighted"), ("shuffle_weight","shuffle_weight")]:
        for s in SEEDS:
            r = run_one(cfg, mode, s); all_results.append(r)

    print("\n## Plan 2.87 Energy-weighted Post-training")
    for r in all_results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<18s} s{r['seed']} AP75={r['ap75']:.4f} best={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only_unf", "energy_weighted", "shuffle_weight"]:
        vals = [r for r in all_results if r["config"] == cfg]
        print(f"  {cfg}: {np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
