"""For each proposal, enumerate 24 actions. Can PHASE DIFF predict best IoU action?"""
import sys, math
import torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; SEED = 42
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(SEED)

def build_actions():
    scales = [0.02, 0.05, 0.10, 0.20]
    acts = []
    for s in scales: acts.extend([(s,0,0,0),(-s,0,0,0),(0,s,0,0),(0,-s,0,0)])
    for s in scales: acts.extend([(0,0,s,0),(0,0,-s,0)])
    return torch.tensor(acts, dtype=torch.float32)

ACTIONS = build_actions().to(DEV); N_ACTIONS = 24

model = build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"]); model.eval()

sampled_props, box_head_in = {}, {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,args: sampled_props.update({"p":[a.clone() for a in args[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,args: box_head_in.update({"x":args[0]}))

tl, vl = build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})

def crop_img(raw, boxes, out=7):
    if raw.dim()==4: raw=raw[0]
    M=boxes.shape[0]; _,H,W=raw.shape
    crops=[]
    for i in range(M):
        x1,y1,x2,y2=boxes[i].long().clamp(min=0)
        x1=x1.clamp(max=W-2);x2=x2.clamp(max=W-1);y1=y1.clamp(max=H-2);y2=y2.clamp(max=H-1)
        if x2<=x1+1:x2=x1+2
        if y2<=y1+1:y2=y1+2
        c=raw[:,y1:y2,x1:x2].float()/255.0
        crops.append(F.interpolate(c.unsqueeze(0),size=(out,out),mode='bilinear',align_corners=False).squeeze(0))
    return torch.stack(crops)

def energy_img(crops):
    M,C,H,W=crops.shape
    fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho");amp=torch.abs(fft)
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing='ij');r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.3).float();md=((rn>0.3)&(rn<=0.7)).float();hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2);am=(amp*md).flatten(2).sum(2);ah=(amp*hi).flatten(2).sum(2)
    return (al/(al+am+ah+1e-8)).mean(dim=1)

def phase_diff(crops, ref_crop):
    """Phase diff relative to unshifted reference. Lower = closer to reference position."""
    fft = torch.fft.rfft2(crops, dim=(-2,-1), norm="ortho")
    fft_ref = torch.fft.rfft2(ref_crop.expand(crops.shape[0],-1,-1,-1), dim=(-2,-1), norm="ortho")
    pha = torch.angle(fft); pha_ref = torch.angle(fft_ref); amp = torch.abs(fft)
    d = torch.atan2(torch.sin(pha-pha_ref), torch.cos(pha-pha_ref))
    w = (torch.abs(d)*amp).flatten(2).sum(2)
    return (w/amp.flatten(2).sum(2).clamp_min(1e-8)).mean(dim=1)

def apply_actions_batch(boxes, action_indices):
    a=ACTIONS[action_indices]
    w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    new_cx=cx+a[:,0]*w;new_cy=cy+a[:,1]*h
    new_w=torch.clamp(w*(1.0+a[:,2]),min=1);new_h=torch.clamp(h*(1.0+a[:,3]),min=1)
    x1=new_cx-0.5*new_w;y1=new_cy-0.5*new_h;x2=new_cx+0.5*new_w;y2=new_cy+0.5*new_h
    return torch.stack([x1,y1,x2,y2],dim=1).clamp(min=0)

all_results = []

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]; img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    raw = imgs_d[0]
    sampled_props.clear(); box_head_in.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k:v.to(DEV) for k,v in t.items()} for t in tgts])
    rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
    if rf is None or sp_raw is None or rf.shape[0]==0: continue

    N = rf.shape[0]
    sp_cat = torch.cat(sp_raw, dim=0)[:N]
    bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
    bw=sp_cat[:,2]-sp_cat[:,0];bh=sp_cat[:,3]-sp_cat[:,1];bcx=sp_cat[:,0]+0.5*bw;bcy=sp_cat[:,1]+0.5*bh
    dx_b=mu[:,0]/10.0;dy_b=mu[:,1]/10.0;dw_b=mu[:,2]/5.0;dh_b=mu[:,3]/5.0
    base_boxes=torch.stack([dx_b*bw+bcx-0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy-0.5*torch.exp(dh_b)*bh,dx_b*bw+bcx+0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy+0.5*torch.exp(dh_b)*bh],dim=1).clamp(min=0)

    prop_to_img = []
    for i_img, p_img in enumerate(sp_raw): prop_to_img.extend([i_img]*p_img.shape[0])

    for k in range(N):
        base = base_boxes[k]; i_img = prop_to_img[k]
        bases = base.unsqueeze(0).repeat(N_ACTIONS, 1)
        aidx = torch.arange(N_ACTIONS, device=DEV)
        refined = apply_actions_batch(bases, aidx)
        refined[:,2] = refined[:,2].clamp(max=img_shape[1]-1)
        refined[:,3] = refined[:,3].clamp(max=img_shape[0]-1)

        gt = tgts[i_img]["boxes"].to(DEV)
        ious = box_iou(refined, gt).max(dim=1).values if len(gt)>0 else torch.zeros(N_ACTIONS,device=DEV)
        iou_base_val = box_iou(base.unsqueeze(0).cpu(), gt.cpu()).max().item() if len(gt)>0 else 0

        # Energy + Phase diff
        with torch.no_grad():
            crops = crop_img(raw, refined)
            ref_crop = crop_img(raw, base.unsqueeze(0))  # reference at original box
            gt_ref = crop_img(raw, gt[0:1]) if len(gt)>0 else ref_crop  # GT reference
            en = energy_img(crops)
            ph = phase_diff(crops, ref_crop)        # vs original box
            ph_gt = phase_diff(crops, gt_ref)       # vs GT box ← THE FAIR TEST

        best_iou_idx = ious.argmax().item()
        best_en_idx = en.argmin().item()  # lower energy = better
        best_ph_idx = ph.argmin().item()  # lower phase diff = closer to original
        best_phgt_idx = ph_gt.argmin().item()  # lower phase diff = closer to GT

        all_results.append({
            "iou_base": iou_base_val,
            "best_iou_idx": best_iou_idx,
            "best_en_idx": best_en_idx,
            "best_ph_idx": best_ph_idx,
            "best_phgt_idx": best_phgt_idx,
            "en_vals": en.cpu().numpy(),
            "ph_vals": ph.cpu().numpy(),
            "phgt_vals": ph_gt.cpu().numpy(),
            "iou_vals": ious.detach().cpu().numpy(),
        })

print(f"Total proposals: {len(all_results)}\n")

print(f"{'Metric':<20s} {'Best%':>8s} {'Top3%':>8s} {'Top5%':>8s} {'Spearman':>10s}")
print("-" * 55)
for name, key in [("Energy (vs orig)", "en_vals"), ("Phase (vs orig)", "ph_vals"), ("Phase (vs GT)", "phgt_vals")]:
    bests = [1 if r[f"best_{'en' if 'en' in key else 'phgt' if 'phgt' in key else 'ph'}_idx"]==r["best_iou_idx"] else 0 for r in all_results]
    top3 = [1 if r["best_iou_idx"] in set(np.argsort(r[key])[:3]) else 0 for r in all_results]
    top5 = [1 if r["best_iou_idx"] in set(np.argsort(r[key])[:5]) else 0 for r in all_results]
    corrs = []
    for r in all_results:
        if np.std(r[key])>1e-8 and np.std(r["iou_vals"])>1e-8:
            corrs.append(np.corrcoef(r[key], r["iou_vals"])[0,1])
    print(f"{name:<20s} {100*np.mean(bests):7.1f}% {100*np.mean(top3):7.1f}% {100*np.mean(top5):7.1f}% {np.mean(corrs):+9.4f}")
print(f"{'(Random)':<20s} {'4.2%':>8s} {'12.5%':>8s} {'20.8%':>8s}")

print(f"\n--- By base IoU ---")
for lo,hi,label in [(0,0.5,"FN"),(0.5,0.75,"TP-border"),(0.75,1.0,"TP-good")]:
    sub=[r for r in all_results if lo<=r["iou_base"]<hi]
    if not sub: continue
    print(f"\n{label}(IoU{lo}-{hi}) n={len(sub)}:")
    for name, key, best_key in [("Energy", "en_vals", "best_en_idx"), ("Phase vs GT", "phgt_vals", "best_phgt_idx")]:
        pct = 100*sum(1 for r in sub if r[best_key]==r["best_iou_idx"])/len(sub)
        print(f"  {name}: {pct:.1f}%")
