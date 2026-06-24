"""Test: image change when box region is removed. Good box = more change."""
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

def apply_actions_batch(boxes, action_indices):
    a=ACTIONS[action_indices]; w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    new_cx=cx+a[:,0]*w;new_cy=cy+a[:,1]*h;new_w=torch.clamp(w*(1.0+a[:,2]),min=1);new_h=torch.clamp(h*(1.0+a[:,3]),min=1)
    x1=new_cx-0.5*new_w;y1=new_cy-0.5*new_h;x2=new_cx+0.5*new_w;y2=new_cy+0.5*new_h
    return torch.stack([x1,y1,x2,y2],dim=1).clamp(min=0)

def removal_delta(raw, boxes):
    """MSE between original image and image with box region filled with mean color.
    Returns (M,) scalar per box. Higher = more content was in the box."""
    if raw.dim()==4: raw=raw[0]
    M=boxes.shape[0]; C,H,W=raw.shape
    raw_f = raw.float()/255.0
    fill_val = raw_f.flatten(1).mean(dim=1, keepdim=True)  # (C, 1) mean per channel

    deltas = []
    for i in range(M):
        x1,y1,x2,y2 = boxes[i].long().clamp(min=0)
        x1=x1.clamp(max=W-2);x2=x2.clamp(max=W-1);y1=y1.clamp(max=H-2);y2=y2.clamp(max=H-1)
        if x2<=x1+1: x2=x1+2
        if y2<=y1+1: y2=y1+2

        filled = raw_f.clone()
        filled[:, y1:y2, x1:x2] = fill_val.unsqueeze(1)  # fill box with mean color

        delta = ((filled - raw_f)**2).mean()  # MSE over entire image
        deltas.append(delta)
    return torch.tensor(deltas, device=DEV)

# Test 1: sensitivity
G = 4
sigmas = [0.1, 0.2, 0.5, 1.0]
results = {s: [] for s in sigmas}

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]; img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    raw = imgs_d[0]
    for tgt in tgts:
        for box in tgt["boxes"].to(DEV):
            w=box[2]-box[0];h=box[3]-box[1];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2

            for sigma in sigmas:
                dx=sigma*torch.randn(G,device=DEV);dy=sigma*torch.randn(G,device=DEV)
                dw=sigma*torch.randn(G,device=DEV);dh=sigma*torch.randn(G,device=DEV)
                g_cx=dx*w+cx;g_cy=dy*h+cy;g_w=torch.exp(dw)*w;g_h=torch.exp(dh)*h
                boxes_g=torch.stack([g_cx-0.5*g_w,g_cy-0.5*g_h,g_cx+0.5*g_w,g_cy+0.5*g_h],dim=1).clamp(min=0)
                boxes_g[:,2]=boxes_g[:,2].clamp(max=img_shape[1]-1)
                boxes_g[:,3]=boxes_g[:,3].clamp(max=img_shape[0]-1)

                deltas = removal_delta(raw, boxes_g)
                ious = box_iou(boxes_g, box.unsqueeze(0)).squeeze()
                results[sigma].append({"var": deltas.var().item(), "iou_var": ious.var().item()})

print(f"{'sigma':>8s} {'delta_var':>12s} {'iou_var':>12s} {'delta/iou':>10s}")
print("-" * 45)
for sigma in sigmas:
    dv = np.mean([r["var"] for r in results[sigma]])
    iv = np.mean([r["iou_var"] for r in results[sigma]])
    print(f"{sigma:8.2f} {dv:12.6f} {iv:12.6f} {dv/max(iv,1e-8):9.1f}x")

# Test 2: ranking — can removal_delta pick the best IoU action?
print(f"\n=== Remove-crop ranking on 24 actions (single image) ===")
for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]; img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    raw = imgs_d[0]
    sampled_props.clear(); box_head_in.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k:v.to(DEV) for k,v in t.items()} for t in tgts])
    rf = box_head_in.get("x"); sp_raw = sampled_props.get("p")
    if rf is None or rf.shape[0]==0: continue

    N = min(rf.shape[0], 50); sp_cat = torch.cat(sp_raw, dim=0)[:N]
    bf = model.roi_heads.box_head(rf)[:N]; mu = model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
    bw=sp_cat[:,2]-sp_cat[:,0];bh=sp_cat[:,3]-sp_cat[:,1];bcx=sp_cat[:,0]+0.5*bw;bcy=sp_cat[:,1]+0.5*bh
    dx_b=mu[:,0]/10.0;dy_b=mu[:,1]/10.0;dw_b=mu[:,2]/5.0;dh_b=mu[:,3]/5.0
    base_boxes=torch.stack([dx_b*bw+bcx-0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy-0.5*torch.exp(dh_b)*bh,dx_b*bw+bcx+0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy+0.5*torch.exp(dh_b)*bh],dim=1).clamp(min=0)
    prop_to_img = [i_img for i_img, p_img in enumerate(sp_raw) for _ in range(p_img.shape[0])]

    all_best = []
    for k in range(N):
        base = base_boxes[k]; i_img = prop_to_img[k]
        bases = base.unsqueeze(0).repeat(N_ACTIONS, 1)
        aidx = torch.arange(N_ACTIONS, device=DEV)
        refined = apply_actions_batch(bases, aidx)
        refined[:,2] = refined[:,2].clamp(max=img_shape[1]-1); refined[:,3] = refined[:,3].clamp(max=img_shape[0]-1)

        gt = tgts[i_img]["boxes"].to(DEV)
        ious = box_iou(refined, gt).max(dim=1).values if len(gt)>0 else torch.zeros(N_ACTIONS,device=DEV)
        deltas = removal_delta(raw, refined)

        best_iou = ious.argmax().item()
        best_delta = deltas.argmax().item()  # higher removal delta = more content in box
        all_best.append(1 if best_delta==best_iou else 0)

    print(f"Remove-crop picks best IoU: {sum(all_best)}/{len(all_best)} = {100*sum(all_best)/len(all_best):.1f}%")
    break
