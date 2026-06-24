"""Plan 2.90: Raw Image Energy + sigma=1.0 — the sweet spot.

Key finding: at sigma=1.0 with raw image crops (7x7), energy variance is 0.023
vs IoU 0.015 — energy FINALLY dominates IoU with meaningful signal.
This is 50x more sensitive than FPN ROI pool at sigma=0.1 (0.0004).

Architecture: raw image crop -> small CNN -> action logits -> discrete action PG.
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
G_SAMPLES = 6; EPOCHS = 8; SEEDS = [42, 123, 456]
RL_WEIGHT = 0.05; KL_WEIGHT = 0.1; ENERGY_BETA = 0.05
HEAD_LR = 0.001; BODY_LR = 0.0001; SIGMA = 1.0
IOU_LO = 0.3; IOU_HI = 0.55

# Action set
def build_actions():
    scales = [0.02, 0.05, 0.10, 0.20]
    acts = []
    for s in scales:
        acts.extend([(s,0,0,0),(-s,0,0,0),(0,s,0,0),(0,-s,0,0)])
    for s in scales:
        acts.extend([(0,0,s,0),(0,0,-s,0)])
    return torch.tensor(acts, dtype=torch.float32)

ACTIONS = build_actions()  # (24, 4)
N_ACTIONS = len(ACTIONS)

class CropActionNet(nn.Module):
    """Take raw image crop (3,7,7), output action logits (24)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
            nn.Flatten(),
            nn.Linear(32*9, 64), nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )
    def forward(self, crop): return self.net(crop)  # (N, 3, 7, 7) -> (N, 24)

def crop_image(raw, boxes):
    """Crop raw image at box locations, resize to 7x7. raw: (3, H, W) or (B, 3, H, W)."""
    if raw.dim() == 4: raw = raw[0]
    M = boxes.shape[0]; _, H, W = raw.shape
    crops = []
    for i in range(M):
        x1,y1,x2,y2 = boxes[i].long().clamp(min=0)
        x1=x1.clamp(max=W-2);x2=x2.clamp(max=W-1);y1=y1.clamp(max=H-2);y2=y2.clamp(max=H-1)
        if x2<=x1+1:x2=x1+2
        if y2<=y1+1:y2=y1+2
        c = raw[:, y1:y2, x1:x2].float() / 255.0  # normalize to [0,1]
        crops.append(F.interpolate(c.unsqueeze(0), size=(7,7), mode='bilinear', align_corners=False).squeeze(0))
    return torch.stack(crops)

def apply_actions_batch(boxes, action_indices):
    a = ACTIONS.to(boxes.device)[action_indices]
    w=boxes[:,2]-boxes[:,0]; h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w; cy=boxes[:,1]+0.5*h
    new_cx=cx+a[:,0]*w; new_cy=cy+a[:,1]*h
    new_w=torch.clamp(w*(1.0+a[:,2]),min=1); new_h=torch.clamp(h*(1.0+a[:,3]),min=1)
    x1=new_cx-0.5*new_w;y1=new_cy-0.5*new_h;x2=new_cx+0.5*new_w;y2=new_cy+0.5*new_h
    return torch.stack([x1,y1,x2,y2],dim=1).clamp(min=0)

def energy_img(crops):
    M,C,H,W=crops.shape
    fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho");amp=torch.abs(fft)
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing='ij');r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.3).float();md=((rn>0.3)&(rn<=0.7)).float();hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2);am=(amp*md).flatten(2).sum(2);ah=(amp*hi).flatten(2).sum(2)
    return (al/(al+am+ah+1e-8)).mean(dim=1)

def grpo_advantage(reward):
    r_mean=reward.mean(dim=1,keepdim=True); r_std=reward.std(dim=1,keepdim=True).clamp_min(1e-6)
    return (reward-r_mean)/r_std

def unfreeze_rlvr(model):
    for p in model.backbone.body.parameters(): p.requires_grad=False
    if hasattr(model.backbone,'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad=True
    for p in model.rpn.parameters(): p.requires_grad=True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad=True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad=True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

def build_opt(model, extra=None):
    body=[];head=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head.append(p)
        else: body.append(p)
    ex=list(extra.parameters()) if extra else []
    return torch.optim.SGD([{'params':body,'lr':BODY_LR},{'params':head,'lr':HEAD_LR},{'params':ex,'lr':HEAD_LR}],lr=HEAD_LR,momentum=0.9,weight_decay=0.0005)

def bl():
    return build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":4}})
def bm():
    return build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}})

@torch.no_grad()
def ev(model, vl):
    model.eval();ps,ts=[],[]
    for img,tgt in vl:
        out=model([i.to(DEV) for i in img])
        ps.extend([{k:v.cpu() for k,v in o.items()} for o in out])
        ts.extend([{k:v.cpu() for k,v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps,ts,iou_threshold=0.5,score_threshold=0.05)

def run_one(cfg_name, mode, seed):
    run_name=f"round290_{cfg_name}_s{seed}";set_seed(seed)
    model=bm().to(DEV);ckpt=torch.load(CKPT,map_location=DEV)
    model.load_state_dict(ckpt["model"]);unfreeze_rlvr(model)

    is_det=mode=="det_only_unf"
    action_net=CropActionNet().to(DEV) if not is_det else None
    if action_net: action_net.train()

    baseline_model=copy.deepcopy(model);baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad=False

    sampled_props,box_head_in={},{}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,args: sampled_props.update({"p":[a.clone() for a in args[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m,args: box_head_in.update({"x":args[0]}))

    tl,vl=bl();run_dir=ensure_run_dir(run_name);shutil.copy(__file__,run_dir/"runner_snapshot.py")
    is_energy=mode=="raw_energy";is_shuffle=mode=="raw_shuffle"
    rng_shuf=torch.Generator(device=DEV).manual_seed(seed+9999)
    opt=build_opt(model,action_net)
    bw_base=baseline_model.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    bb_base=baseline_model.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    h=[];best_ap75=-1.0;diag={"reward_std":[],"en_gap":[],"energy_var":[]}

    for ep in range(1,EPOCHS+1):
        model.train()
        if action_net: action_net.train()
        td,trl,tkl=0.0,0.0,0.0

        for imgs,tgts in tqdm(tl,desc=f"{run_name} e{ep}"):
            imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
            sampled_props.clear();box_head_in.clear()
            img_shape=(imgs_d[0].shape[-2],imgs_d[0].shape[-1]);raw=imgs_d[0]

            ld=model(imgs_d,tgts_t)
            det=sum(ld.values()) if isinstance(ld,dict) else sum(sum(d.values()) for d in ld if isinstance(d,dict))
            rf=box_head_in.get("x");sp_raw=sampled_props.get("p")
            rl=torch.tensor(0.0,device=DEV);kl_loss=torch.tensor(0.0,device=DEV)

            if not is_det and action_net is not None and rf is not None and sp_raw is not None and rf.shape[0]>0:
                N=rf.shape[0]
                kl_loss=KL_WEIGHT*((model.roi_heads.box_predictor.bbox_pred.weight-bw_base).pow(2).sum()+(model.roi_heads.box_predictor.bbox_pred.bias-bb_base).pow(2).sum())

                # Base boxes
                sp_cat=torch.cat(sp_raw,dim=0)[:N]
                bf=model.roi_heads.box_head(rf)
                mu=model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
                bw=sp_cat[:,2]-sp_cat[:,0];bh=sp_cat[:,3]-sp_cat[:,1];bcx=sp_cat[:,0]+0.5*bw;bcy=sp_cat[:,1]+0.5*bh
                dx_b=mu[:,0]/10.0;dy_b=mu[:,1]/10.0;dw_b=mu[:,2]/5.0;dh_b=mu[:,3]/5.0
                base_boxes=torch.stack([dx_b*bw+bcx-0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy-0.5*torch.exp(dh_b)*bh,dx_b*bw+bcx+0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy+0.5*torch.exp(dh_b)*bh],dim=1).clamp(min=0)

                # G random perturbations (sigma=1.0, the sweet spot)
                s=torch.full_like(mu,SIGMA)
                deltas=mu.detach().unsqueeze(1)+s.unsqueeze(1)*torch.randn(N,G_SAMPLES,4,device=DEV)
                sp_exp=sp_cat.repeat_interleave(G_SAMPLES,dim=0)
                delta_flat=deltas.reshape(-1,4)
                bw2=sp_exp[:,2]-sp_exp[:,0];bh2=sp_exp[:,3]-sp_exp[:,1];bcx2=sp_exp[:,0]+0.5*bw2;bcy2=sp_exp[:,1]+0.5*bh2
                dx=delta_flat[:,0]/10.0;dy=delta_flat[:,1]/10.0;dw=delta_flat[:,2]/5.0;dh=delta_flat[:,3]/5.0
                perturbed_boxes=torch.stack([dx*bw2+bcx2-0.5*torch.exp(dw)*bw2,dy*bh2+bcy2-0.5*torch.exp(dh)*bh2,dx*bw2+bcx2+0.5*torch.exp(dw)*bw2,dy*bh2+bcy2+0.5*torch.exp(dh)*bh2],dim=1).clamp(min=0)
                perturbed_boxes[:,2]=perturbed_boxes[:,2].clamp(max=img_shape[1]-1)
                perturbed_boxes[:,3]=perturbed_boxes[:,3].clamp(max=img_shape[0]-1)

                # Raw image crop -> action net -> discrete action
                crops=crop_image(raw,perturbed_boxes)  # (N*G, 3, 7, 7)
                action_logits=action_net(crops)  # (N*G, 24)
                action_dist=torch.distributions.Categorical(logits=action_logits)
                actions=action_dist.sample()  # (N*G,)
                log_probs=action_dist.log_prob(actions).view(N,G_SAMPLES)

                # Apply actions to perturbed boxes
                refined_boxes=apply_actions_batch(perturbed_boxes,actions)
                refined_boxes[:,2]=refined_boxes[:,2].clamp(max=img_shape[1]-1)
                refined_boxes[:,3]=refined_boxes[:,3].clamp(max=img_shape[0]-1)

                # IoU reward
                iou_r=torch.zeros(N,G_SAMPLES,device=DEV)
                off_p=0
                for i_img,p_img in enumerate(sp_raw):
                    np_i=min(p_img.shape[0],N-off_p)
                    if np_i<=0: break
                    idx_s=off_p*G_SAMPLES;idx_e=(off_p+np_i)*G_SAMPLES
                    gt=tgts_t[i_img]["boxes"]
                    if len(gt)>0:
                        ious=box_iou(refined_boxes[idx_s:idx_e],gt)
                        iou_r[off_p:off_p+np_i]=ious.max(dim=1).values.view(np_i,G_SAMPLES)
                    off_p+=np_i
                reward_img=2*iou_r-1

                # Energy on refined box crops
                gated_bias=torch.zeros(N,G_SAMPLES,device=DEV)
                if is_energy or is_shuffle:
                    refined_crops=crop_image(raw,refined_boxes)
                    energy=energy_img(refined_crops).view(N,G_SAMPLES)
                    diag["energy_var"].append(energy.var(dim=1).mean().item())
                    if is_shuffle:
                        energy=energy.reshape(-1)[torch.randperm(N*G_SAMPLES,generator=rng_shuf,device=DEV)].view(N,G_SAMPLES)
                    en_pen=-torch.sigmoid(15*(energy-0.5))*ENERGY_BETA
                    gmax=iou_r.max(dim=1).values
                    bmask=((gmax>=IOU_LO)&(gmax<IOU_HI)).unsqueeze(1).float()
                    gated_bias=en_pen*bmask
                    tp_mask=gmax>=0.5;fn_mask=gmax<0.5
                    if tp_mask.any() and fn_mask.any():
                        diag["en_gap"].append((energy[tp_mask].mean()-energy[fn_mask].mean()).item())

                # GRPO + energy after
                adv=grpo_advantage(reward_img)
                if is_energy or is_shuffle:
                    adv=adv+gated_bias
                diag["reward_std"].append(adv.std().item())
                soft_w=iou_r.max(dim=1).values.clamp(0,1).unsqueeze(1)
                rl=-(adv.detach()*log_probs*soft_w).mean()

            loss=det+RL_WEIGHT*rl+kl_loss;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            td+=det.item();trl+=rl.item();tkl+=kl_loss.item()

        em=ev(model,vl)
        rs_m=np.mean(diag["reward_std"]) if diag["reward_std"] else 0
        eg_m=np.mean(diag["en_gap"]) if diag["en_gap"] else 0
        ev_m=np.mean(diag["energy_var"]) if diag["energy_var"] else 0
        row={"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"],"reward_std":float(rs_m),"en_gap":float(eg_m),"energy_var":float(ev_m),"det_loss":td,"rl_loss":trl,"kl_loss":tkl}
        h.append(row);print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f} r_std={rs_m:.4f} en_var={ev_m:.6f} en_gap={eg_m:.4f}")
        if em["ap75"]>best_ap75:best_ap75=em["ap75"]
        for k in diag:diag[k].clear()

    best_h=max(h,key=lambda r:r["val_ap75"])
    em.update({"run_name":run_name,"config":cfg_name,"mode":mode,"seed":seed,"epochs":len(h),"best_ap50":best_h["val_ap50"],"best_ap75":best_ap75,"history":h,"git_hash":GIT})
    save_json(em,run_dir/"eval_metrics.json");return em

if __name__=="__main__":
    all_results=[]
    for cfg,mode in [("det_only_unf","det_only_unf"),("raw_energy","raw_energy"),("raw_shuffle","raw_shuffle")]:
        for s in SEEDS: r=run_one(cfg,mode,s);all_results.append(r)
    print("\n## Plan 2.90 Raw Image Energy + sigma=1.0")
    for r in all_results:
        bh=max(r["history"],key=lambda x:x["val_ap75"])
        print(f"  {r['config']:<15s} s{r['seed']} best={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
    for cfg in ["det_only_unf","raw_energy","raw_shuffle"]:
        vals=[r for r in all_results if r["config"]==cfg]
        if vals: print(f"  {cfg}: {np.mean([v['best_ap75'] for v in vals]):.4f} +/- {np.std([v['best_ap75'] for v in vals]):.4f}")
