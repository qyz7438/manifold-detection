"""Round 2.115: Phase-only edge strength as DPO pair selector."""
import copy, shutil, subprocess, sys, numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320, decode_boxes, evaluate_model, unfreeze_rlvr
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV, SEED = "cuda", 42; EPOCHS = 8
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
DPO_WEIGHT, KL_WEIGHT, BETA = 0.1, 0.01, 2.0

def bm():
    return build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}})

def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD([{"params":body,"lr":0.0001},{"params":head,"lr":0.001}], lr=0.001, momentum=0.9, weight_decay=0.0005)

def phase_edge_score(full_imgs, sp_raw_list):
    """Phase-only reconstruction edge strength per proposal (no crops needed)."""
    scores = []
    off = 0
    for i_img, p_img in enumerate(sp_raw_list):
        n_p = p_img.shape[0]; full = full_imgs[i_img]
        sp_cat = p_img
        for i in range(n_p):
            x1, y1, x2, y2 = sp_cat[i].long()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(full.shape[2], x2), min(full.shape[1], y2)
            if x2 <= x1 or y2 <= y1: off += 1; continue
            crop = full[:, y1:y2, x1:x2].unsqueeze(0)
            crop = F.interpolate(crop, (64, 64), mode='bilinear', align_corners=False).squeeze(0)
            fft = torch.fft.rfft2(crop, dim=(-2, -1))
            recon = torch.fft.irfft2(torch.exp(1j * torch.angle(fft)), s=(64, 64))
            gray = 0.299*recon[0] + 0.587*recon[1] + 0.114*recon[2]
            sx = torch.zeros_like(gray); sy = torch.zeros_like(gray)
            sx[1:-1,1:-1] = (gray[2:,1:-1] - gray[:-2,1:-1]) / 8
            sy[1:-1,1:-1] = (gray[1:-1,2:] - gray[1:-1,:-2]) / 8
            scores.append((sx**2 + sy**2).sqrt().mean().item())
            off += 1
    return torch.tensor(scores, device=DEV)

def compute_stable_iou(sp_raw, bf, box_predictor, tgts_t):
    sp_cat = torch.cat(sp_raw, dim=0); N = sp_cat.shape[0]
    with torch.no_grad():
        reg = box_predictor.bbox_pred(bf[:N]); decoded = decode_boxes(sp_cat, reg[:, 2:6])
    iou, gt_idx = torch.zeros(N, device=DEV), torch.full((N,), -1, dtype=torch.long, device=DEV)
    off = 0
    for i_img, p_img in enumerate(sp_raw):
        n_p = p_img.shape[0]
        if n_p > 0 and len(tgts_t[i_img]["boxes"]) > 0:
            i = box_iou(decoded[off:off+n_p], tgts_t[i_img]["boxes"])
            iou[off:off+n_p], gt_idx[off:off+n_p] = i.max(dim=1)
        off += n_p
    return iou, gt_idx

def run_one(cfg_name, mode, seed):
    run_name = f"round2115_{cfg_name}_s{seed}"; set_seed(seed)
    model = bm().to(DEV); ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"]); unfreeze_rlvr(model)
    baseline_model = copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad = False
    sampled_props, box_head_in = {}, {}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m, a: sampled_props.update({"p":[x.clone() for x in a[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m, a: box_head_in.update({"x":a[0]}))
    tl, vl = build_penn_fudan_loaders_320(batch_size=2)
    opt = build_opt(model); bp = baseline_model.roi_heads.box_predictor
    run_dir = ensure_run_dir(run_name); shutil.copy(__file__, run_dir / "runner_snapshot.py")
    is_det = mode == "det_only"; h, best_ap75 = [], -1.0

    for ep in range(1, EPOCHS + 1):
        model.train(); td, tdpo, tkl = 0.0, 0.0, 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}", leave=False):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k:v.to(DEV) for k,v in t.items()} for t in tgts]
            sampled_props.clear(); box_head_in.clear()
            ld = model(imgs_d, tgts_t); det = sum(ld.values())
            rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
            dpo = torch.tensor(0.0, device=DEV); kl = torch.tensor(0.0, device=DEV)

            if not is_det and rf is not None and sp_raw is not None and rf.shape[0] > 0:
                bf = model.roi_heads.box_head(rf)
                cls_logits = model.roi_heads.box_predictor.cls_score(bf)
                person_logit = cls_logits[:, 1]
                with torch.no_grad():
                    baseline_bf = baseline_model.roi_heads.box_head(rf)
                    b_logits = bp.cls_score(baseline_bf)
                    b_person = b_logits[:, 1]

                iou_p, gt_idx = compute_stable_iou(sp_raw, baseline_bf, bp, tgts_t)
                N = min(cls_logits.shape[0], iou_p.shape[0])

                # Phase-edge scores
                edge_s = phase_edge_score(imgs_d, sp_raw)[:N]

                n_pairs = 0
                for gid in torch.unique(gt_idx[:N]):
                    if gid < 0: continue
                    mask = gt_idx[:N] == gid
                    if mask.sum() < 2: continue
                    scores = edge_s[mask].cpu().numpy()
                    logits = person_logit[mask]; ref = b_person[mask]
                    best_idx = int(np.argmax(scores))
                    worst_idx = int(np.argmin(scores))
                    if best_idx == worst_idx: continue
                    lc, lr = logits[best_idx], logits[worst_idx]
                    rc, rr = ref[best_idx], ref[worst_idx]
                    dpo = dpo - F.logsigmoid(BETA * ((lc - rc) - (lr - rr)))
                    n_pairs += 1
                if n_pairs > 0: dpo = dpo / n_pairs
                kl = KL_WEIGHT * (person_logit[:N] - b_person[:N]).pow(2).mean()

            loss = det + DPO_WEIGHT * dpo + kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0); opt.step()
            td += det.item(); tdpo += dpo.item(); tkl += kl.item()

        em = evaluate_model(model, vl, DEV)
        row = {"epoch":ep, "val_ap50":em["ap50"], "val_ap75":em["ap75"], "ece":em.get("ece",0), "det":td, "dpo":tdpo, "kl":tkl}
        h.append(row); print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]

    best_h = max(h, key=lambda r: r["val_ap75"])
    em.update({"run_name":run_name,"config":cfg_name,"mode":mode,"seed":seed,"epochs":len(h),"best_ap50":best_h["val_ap50"],"best_ap75":best_ap75,"history":h})
    save_json(em, run_dir / "eval_metrics.json"); return em

if __name__ == "__main__":
    results = []
    for cfg, mode in [("det_only","det_only"),("edge_dpo","edge_dpo")]:
        r = run_one(cfg, mode, 42); results.append(r)
    print("\n## 2.115 Phase-Only Edge DPO")
    for r in results:
        bh = max(r["history"], key=lambda x: x["val_ap75"])
        print(f"  {r['config']:<12s} s{r['seed']} AP75={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
