"""HOG + edge distribution as content-aware quality metric. No GT needed."""
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

def edge_quality(crops):
    """Edge-based content quality. Higher = more structured object in center.
    Returns (M,) scalar per crop."""
    M,C,H,W = crops.shape
    crops_g = crops.mean(dim=1, keepdim=True)  # (M, 1, H, W) grayscale

    # Sobel edges
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=DEV).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=DEV).view(1,1,3,3)

    gx = F.conv2d(crops_g, sobel_x, padding=1)  # (M, 1, H, W)
    gy = F.conv2d(crops_g, sobel_y, padding=1)

    edge_mag = torch.sqrt(gx**2 + gy**2).squeeze(1)  # (M, H, W)
    edge_dir = torch.atan2(gy, gx).squeeze(1)  # (M, H, W)

    # Metrics:
    # 1. Total edge energy (more edges = more structure)
    total_edge = edge_mag.flatten(1).mean(dim=1)  # (M,)

    # 2. Edge concentration in center (center weight map)
    cy, cx = torch.meshgrid(torch.linspace(-1,1,H,device=DEV), torch.linspace(-1,1,W,device=DEV), indexing='ij')
    center_w = 1.0 - torch.sqrt(cx**2 + cy**2) / 1.414  # 1 at center, 0 at corners
    center_w = center_w.clamp(min=0)
    center_edge = (edge_mag * center_w).flatten(1).mean(dim=1) / total_edge.clamp_min(1e-8)  # (M,)

    # 3. Vertical edge dominance (pedestrians have vertical edges)
    vert_w = torch.cos(edge_dir).abs()  # 1 when vertical, 0 when horizontal
    vert_ratio = (edge_mag * vert_w).flatten(1).mean(dim=1) / total_edge.clamp_min(1e-8)  # (M,)

    # Composite quality score
    quality = total_edge * center_edge * vert_ratio  # (M,)
    return quality

def energy_img(crops):
    M,C,H,W = crops.shape
    fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho"); amp=torch.abs(fft)
    fh=torch.fft.fftfreq(H,device=DEV); fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing='ij'); r=torch.sqrt(X**2+Y**2); R=r.max().clamp_min(1e-6); rn=r/R
    lo=(rn<=0.3).float(); md=((rn>0.3)&(rn<=0.7)).float(); hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2); am=(amp*md).flatten(2).sum(2); ah=(amp*hi).flatten(2).sum(2)
    return (al/(al+am+ah+1e-8)).mean(dim=1)

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

    N = rf.shape[0]; sp_cat = torch.cat(sp_raw, dim=0)[:N]
    bf = model.roi_heads.box_head(rf); mu = model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
    bw=sp_cat[:,2]-sp_cat[:,0];bh=sp_cat[:,3]-sp_cat[:,1];bcx=sp_cat[:,0]+0.5*bw;bcy=sp_cat[:,1]+0.5*bh
    dx_b=mu[:,0]/10.0;dy_b=mu[:,1]/10.0;dw_b=mu[:,2]/5.0;dh_b=mu[:,3]/5.0
    base_boxes=torch.stack([dx_b*bw+bcx-0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy-0.5*torch.exp(dh_b)*bh,dx_b*bw+bcx+0.5*torch.exp(dw_b)*bw,dy_b*bh+bcy+0.5*torch.exp(dh_b)*bh],dim=1).clamp(min=0)

    prop_to_img = [i_img for i_img, p_img in enumerate(sp_raw) for _ in range(p_img.shape[0])]

    for k in range(N):
        base = base_boxes[k]; i_img = prop_to_img[k]
        bases = base.unsqueeze(0).repeat(N_ACTIONS, 1)
        aidx = torch.arange(N_ACTIONS, device=DEV)
        refined = apply_actions_batch(bases, aidx)
        refined[:,2] = refined[:,2].clamp(max=img_shape[1]-1)
        refined[:,3] = refined[:,3].clamp(max=img_shape[0]-1)

        gt = tgts[i_img]["boxes"].to(DEV)
        ious = box_iou(refined, gt).max(dim=1).values if len(gt)>0 else torch.zeros(N_ACTIONS,device=DEV)

        with torch.no_grad():
            crops = crop_img(raw, refined)
            eq = edge_quality(crops)
            en = energy_img(crops)

        best_iou = ious.argmax().item()
        best_eq = eq.argmax().item()  # higher edge quality = better
        best_en = en.argmin().item()

        all_results.append({
            "iou_vals": ious.detach().cpu().numpy(),
            "eq_vals": eq.cpu().numpy(),
            "en_vals": en.cpu().numpy(),
            "best_iou": best_iou,
            "best_eq": best_eq,
            "best_en": best_en,
        })

print(f"Proposals: {len(all_results)}\n")

for name, key in [("Edge Quality", "best_eq"), ("Energy", "best_en")]:
    correct = sum(1 for r in all_results if r[key]==r["best_iou"])
    top3 = sum(1 for r in all_results if r["best_iou"] in set(np.argsort(r["eq_vals" if "eq" in key else "en_vals"])[::-1 if "eq" in key else 0][:3]))
    vals = np.array([r["eq_vals" if "eq" in key else "en_vals"] for r in all_results])
    ious = np.array([r["iou_vals"] for r in all_results])
    corrs = []
    for i in range(len(vals)):
        if np.std(vals[i])>1e-8 and np.std(ious[i])>1e-8:
            corrs.append(np.corrcoef(vals[i], ious[i])[0,1])
    print(f"{name}: best={100*correct/len(all_results):.1f}% top-3={100*top3/len(all_results):.1f}% Spearman={np.mean(corrs):.4f}")
