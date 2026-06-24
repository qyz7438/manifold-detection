"""Proper phase analysis — circular statistics, phase gradient, phase coherence."""
import sys, torch, numpy as np
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320, decode_boxes
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed
from tqdm import tqdm
set_seed(42)
DEV="cuda"; CKPT="runs/round227_v1_baseline_20ep/checkpoint_best.pth"
model=bm().to(DEV)
ckpt_=torch.load(CKPT,map_location=DEV); model.load_state_dict(ckpt_["model"]); model.eval()

sampled_props,box_head_in,roi_crops={},{},{}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,args: sampled_props.update({"p":[a.clone() for a in args[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,args: box_head_in.update({"x":args[0]}))
model.roi_heads.box_roi_pool.register_forward_hook(lambda m,i,o: roi_crops.update({"c":o.clone()}))

_,vl=build_penn_fudan_loaders_320(batch_size=1)

all_phase=[]
for img,tgt in tqdm(vl,desc="Phase analysis"):
    img_d=[img[0].to(DEV)]; tgt_d=[{k:v.to(DEV) for k,v in tgt[0].items()}]
    sampled_props.clear();box_head_in.clear();roi_crops.clear()
    with torch.no_grad(): model(img_d,tgt_d)
    rf=box_head_in.get("x"); sp_raw=sampled_props.get("p"); crops=roi_crops.get("c")
    if rf is None or sp_raw is None or crops is None or rf.shape[0]==0: continue
    sp_cat=torch.cat(sp_raw,dim=0)
    bf=model.roi_heads.box_head(rf)
    reg=model.roi_heads.box_predictor.bbox_pred(bf)
    decoded=decode_boxes(sp_cat,reg[:,2:6])
    conf=F.softmax(model.roi_heads.box_predictor.cls_score(bf),dim=-1)[:,1]
    gt=tgt_d[0]["boxes"]
    if len(gt)==0: continue
    keep=nms(decoded,conf,0.5)

    for i in range(crops.shape[0]):
        crop=crops[i]  # (C, 7, 7)
        fft=torch.fft.rfft2(crop,dim=(-2,-1),norm="ortho")  # (C, 7, 4)
        amp=torch.abs(fft)
        phase=torch.angle(fft)  # (C, 7, 4)

        # 1. Phase gradient (horizontal freq dim): sharp edges = consistent gradient
        phase_grad=phase[:,:,1:]-phase[:,:,:-1]
        phase_grad=(phase_grad+np.pi)%(2*np.pi)-np.pi  # wrap to [-pi,pi]
        pg_std=phase_grad.std().item()  # high = disorganized

        # 2. Circular coherence R = |mean(exp(j*phase))|
        cp=torch.exp(1j*phase)
        R_ch=cp.mean(dim=0).abs().mean().item()   # coherence across channels
        R_sp=cp.mean(dim=(1,2)).abs().mean().item()  # coherence across spatial pos
        circ_var=1.0-cp.mean(dim=(0,1,2)).abs().item()  # 1-R_total

        # 3. Phase-weighted amplitude: amp * cos(phase) = real part
        real_part=fft.real.mean().item()  # structure-preserving component

        # IoU matching
        if i < decoded.shape[0]:
            iou_match=box_iou(decoded[i:i+1],gt).max().item()
            # Best for GT: is this the highest-IoU match for its assigned GT?
            best_gt_idx=box_iou(decoded[i:i+1],gt).argmax().item()
            is_best=True
            for j in range(decoded.shape[0]):
                if j!=i:
                    iou_j=box_iou(decoded[j:j+1],gt).max().item()
                    gt_j=box_iou(decoded[j:j+1],gt).argmax().item()
                    if gt_j==best_gt_idx and iou_j>iou_match:
                        is_best=False; break
            if decoded.shape[0]==1:
                is_best=(iou_match>0.5)
        else:
            iou_match=0.0; is_best=False

        all_phase.append({
            "iou":iou_match,"is_best":is_best,
            "nms_survives": i in keep.tolist(),
            "pg_std":pg_std,"R_ch":R_ch,"R_sp":R_sp,"circ_var":circ_var,
            "real_part":real_part,
        })

print(f"Total proposals: {len(all_phase)}")

iou=np.array([d["iou"] for d in all_phase])
is_best=np.array([d["is_best"] for d in all_phase])
nms_s=np.array([d["nms_survives"] for d in all_phase])

# 1. Global
print("\n=== Phase: best vs non-best ===")
for name in ["pg_std","R_ch","R_sp","circ_var","real_part"]:
    vals=np.array([d[name] for d in all_phase])
    b=vals[is_best]; nb=vals[~is_best]
    gap=b.mean()-nb.mean()
    d=gap/(np.sqrt(b.var()+nb.var())/2+1e-8)
    m=" <<<" if abs(d)>0.5 else ""
    print(f"  {name:<14s} best={b.mean():.4f} non={nb.mean():.4f} d={d:+.3f}{m}")

# 2. Within IoU bins
print("\n=== Phase within IoU bins ===")
for lo,hi in [(0.3,0.5),(0.5,0.7),(0.7,0.9)]:
    mask=(iou>=lo)&(iou<hi)
    bm=mask&is_best; nm=mask&~is_best
    if bm.sum()<3 or nm.sum()<3: continue
    strong=[]
    for name in ["pg_std","R_ch","R_sp","circ_var","real_part"]:
        vals=np.array([d[name] for d in all_phase])
        d=(vals[bm].mean()-vals[nm].mean())/(np.sqrt(vals[bm].var()+vals[nm].var())/2+1e-8)
        if abs(d)>0.3: strong.append((name,d))
    if strong:
        items=", ".join([f"{n}:{d:+.2f}" for n,d in strong])
        print(f"  IoU[{lo:.1f},{hi:.1f}) n={mask.sum()}: {items}")

# 3. NMS survival
print("\n=== Phase: NMS survival ===")
for name in ["pg_std","R_ch","R_sp","circ_var","real_part"]:
    vals=np.array([d[name] for d in all_phase])
    s=vals[nms_s]; ns=vals[~nms_s]
    gap=s.mean()-ns.mean()
    d=gap/(np.sqrt(s.var()+ns.var())/2+1e-8)
    m=" <<<" if abs(d)>0.5 else ""
    print(f"  {name:<14s} surv={s.mean():.4f} nonsurv={ns.mean():.4f} d={d:+.3f}{m}")

# 4. Within NMS survivors: high vs low IoU
print("\n=== NMS survivors: high-IoU vs low-IoU phase ===")
sv_iou=iou[nms_s]
if len(sv_iou)>=2:
    med=np.median(sv_iou)
    hi=sv_iou>=med; lo=sv_iou<med
    for name in ["pg_std","R_ch","R_sp","circ_var","real_part"]:
        vals=np.array([d[name] for d in all_phase])[nms_s]
        gap=vals[hi].mean()-vals[lo].mean()
        d=gap/(np.sqrt(vals[hi].var()+vals[lo].var())/2+1e-8)
        print(f"  {name:<14s} highIoU={vals[hi].mean():.4f} lowIoU={vals[lo].mean():.4f} d={d:+.3f}")
