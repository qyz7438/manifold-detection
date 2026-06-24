"""For each proposal, enumerate all 24 actions. Can energy predict which action gives best IoU?"""
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

# Action set (same as round290)
def build_actions():
    scales = [0.02, 0.05, 0.10, 0.20]
    acts = []
    for s in scales:
        acts.extend([(s,0,0,0),(-s,0,0,0),(0,s,0,0),(0,-s,0,0)])
    for s in scales:
        acts.extend([(0,0,s,0),(0,0,-s,0)])
    return torch.tensor(acts, dtype=torch.float32)

ACTIONS = build_actions().to(DEV)  # (24, 4)
N_ACTIONS = 24

model = build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"]); model.eval()

sampled_props, box_head_in = {}, {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,args: sampled_props.update({"p":[a.clone() for a in args[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,args: box_head_in.update({"x":args[0]}))

tl, vl = build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})

def crop_image(raw, boxes, out=7):
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
    lo=(rn<=0.3).float()
    al=(amp*lo).flatten(2).sum(2);at=al+(amp*((rn>0.3)&(rn<=0.7)).float()).flatten(2).sum(2)+(amp*(rn>0.7).float()).flatten(2).sum(2)+1e-8
    return (al/at).mean(dim=1)  # (M,)

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

    # Map proposal index -> image index
    prop_to_img = []
    for i_img, p_img in enumerate(sp_raw):
        prop_to_img.extend([i_img] * p_img.shape[0])

    # For each proposal, enumerate all 24 actions
    for k in range(N):
        base = base_boxes[k]
        i_img = prop_to_img[k] if k < len(prop_to_img) else 0
        # Repeat base box 24 times
        bases = base.unsqueeze(0).repeat(N_ACTIONS, 1)  # (24, 4)
        action_indices = torch.arange(N_ACTIONS, device=DEV)  # (24,)
        refined = apply_actions_batch(bases, action_indices)
        refined[:,2] = refined[:,2].clamp(max=img_shape[1]-1)
        refined[:,3] = refined[:,3].clamp(max=img_shape[0]-1)

        # IoU
        gt = tgts[i_img]["boxes"].to(DEV) if i_img < len(tgts) else tgts[0]["boxes"].to(DEV)
        iou_base_val = box_iou(base.unsqueeze(0).cpu(), gt.cpu()).max().item() if len(gt)>0 else 0
        ious = box_iou(refined, gt).max(dim=1).values if len(gt)>0 else torch.zeros(N_ACTIONS,device=DEV)

        # Compute energy on raw image crops
        with torch.no_grad():
            crops = crop_image(raw, refined)  # (24, 3, 7, 7)
            energy = energy_img(crops)  # (24,)

        # Record
        best_iou_idx = ious.argmax().item()
        best_energy_idx = energy.argmin().item()  # lower energy = better
        best_iou = ious.max().item()

        all_results.append({
            "iou_base": iou_base_val,
            "iou_vals": ious.detach().cpu().numpy(),
            "energy_vals": energy.cpu().numpy(),
            "best_iou_idx": best_iou_idx,
            "best_energy_idx": best_energy_idx,
        })

print(f"Total proposals analyzed: {len(all_results)}")

# Key metric: does energy pick the same action as IoU?
energy_correct = sum(1 for r in all_results if r["best_energy_idx"] == r["best_iou_idx"])
print(f"\nEnergy picks same action as IoU: {energy_correct}/{len(all_results)} = {100*energy_correct/len(all_results):.1f}%")

# Top-k agreement
for k in [3, 5, 8]:
    hits = 0
    for r in all_results:
        topk_energy = set(np.argsort(r["energy_vals"])[:k])  # lowest k energy
        if r["best_iou_idx"] in topk_energy:
            hits += 1
    print(f"Best IoU in top-{k} energy: {hits}/{len(all_results)} = {100*hits/len(all_results):.1f}%")

# Spearman correlation per proposal
corrs = []
for r in all_results:
    if np.std(r["energy_vals"]) > 1e-8 and np.std(r["iou_vals"]) > 1e-8:
        c = np.corrcoef(r["energy_vals"], r["iou_vals"])[0,1]
        corrs.append(c)
print(f"\nEnergy-IoU Spearman (per proposal): mean={np.mean(corrs):.3f} median={np.median(corrs):.3f}")
print(f"  % negative corr: {100*sum(1 for c in corrs if c<0)/len(corrs):.1f}%")

# Split by base IoU
for lo, hi, label in [(0, 0.5, "FN"), (0.5, 0.75, "TP-borderline"), (0.75, 1.0, "TP-good")]:
    sub = [r for r in all_results if lo <= r["iou_base"] < hi]
    if not sub: continue
    ec = sum(1 for r in sub if r["best_energy_idx"]==r["best_iou_idx"])
    sub_corrs = []
    for r in sub:
        if np.std(r["energy_vals"])>1e-8 and np.std(r["iou_vals"])>1e-8:
            sub_corrs.append(np.corrcoef(r["energy_vals"], r["iou_vals"])[0,1])
    print(f"\n{label} (IoU {lo}-{hi}): n={len(sub)}  energy_picks_best={100*ec/len(sub):.1f}%  mean_corr={np.mean(sub_corrs):.3f}")

# Energy variance per proposal
en_vars = []
for r in all_results:
    en_vars.append(np.var(r["energy_vals"]))
print(f"\nEnergy std within proposal: mean={np.sqrt(np.mean(en_vars)):.4f}")
