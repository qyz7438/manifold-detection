"""Single-image full parameter scan: which metrics correlate with optimal action?"""
import sys, math
import torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(42)

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
        if x2<=x1+1: x2=x1+2
        if y2<=y1+1: y2=y1+2
        c=raw[:,y1:y2,x1:x2].float()/255.0
        crops.append(F.interpolate(c.unsqueeze(0),size=(out,out),mode='bilinear',align_corners=False).squeeze(0))
    return torch.stack(crops)

def apply_actions_batch(boxes, action_indices):
    a=ACTIONS[action_indices]; w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    new_cx=cx+a[:,0]*w;new_cy=cy+a[:,1]*h;new_w=torch.clamp(w*(1.0+a[:,2]),min=1);new_h=torch.clamp(h*(1.0+a[:,3]),min=1)
    x1=new_cx-0.5*new_w;y1=new_cy-0.5*new_h;x2=new_cx+0.5*new_w;y2=new_cy+0.5*new_h
    return torch.stack([x1,y1,x2,y2],dim=1).clamp(min=0)

# ===== METRIC DEFINITIONS =====
def metric_energy(crops):
    """FFT amplitude low-freq concentration."""
    M,C,H,W=crops.shape; fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho"); amp=torch.abs(fft)
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing='ij');r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.3).float();md=((rn>0.3)&(rn<=0.7)).float();hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2);am=(amp*md).flatten(2).sum(2);ah=(amp*hi).flatten(2).sum(2)
    return (al/(al+am+ah+1e-8)).mean(dim=1)

def metric_edge_quality(crops):
    """Edge energy * center weight * vertical dominance."""
    M,C,H,W=crops.shape; crops_g=crops.mean(dim=1,keepdim=True)
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    sy=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    gx=F.conv2d(crops_g,sx,padding=1);gy=F.conv2d(crops_g,sy,padding=1)
    em=torch.sqrt(gx**2+gy**2).squeeze(1);ed=torch.atan2(gy,gx).squeeze(1)
    te=em.flatten(1).mean(dim=1)
    cy,cx=torch.meshgrid(torch.linspace(-1,1,H,device=DEV),torch.linspace(-1,1,W,device=DEV),indexing='ij')
    cw=(1.0-torch.sqrt(cx**2+cy**2)/1.414).clamp(min=0)
    ce=(em*cw).flatten(1).mean(dim=1)/te.clamp_min(1e-8)
    vw=torch.cos(ed).abs();vr=(em*vw).flatten(1).mean(dim=1)/te.clamp_min(1e-8)
    return te*ce*vr

def metric_entropy(crops):
    """Pixel-wise entropy per crop — higher = more structure."""
    M,C,H,W=crops.shape
    flat=crops.flatten(2)  # (M, C, H*W)
    eps=1e-8; return -(flat*torch.log(flat+eps)).sum(dim=(1,2))/C

def metric_grad_std(crops):
    """Std of gradient magnitude — structured objects have high variance."""
    M,C,H,W=crops.shape; crops_g=crops.mean(dim=1,keepdim=True)
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    sy=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    gx=F.conv2d(crops_g,sx,padding=1);gy=F.conv2d(crops_g,sy,padding=1)
    return torch.sqrt(gx**2+gy**2).flatten(2).std(dim=2).squeeze(1)

def metric_center_surround(crops):
    """Center mean - border mean. Object in center = large positive difference."""
    M,C,H,W=crops.shape; c=crops[:,:,H//3:2*H//3,W//3:2*W//3]; cmean=c.flatten(2).mean(dim=2)
    smean=crops.flatten(2).mean(dim=2); return (cmean-smean).mean(dim=1)

def metric_laplacian_var(crops):
    """Variance of Laplacian — sharp edges = high variance."""
    M,C,H,W=crops.shape; crops_g=crops.mean(dim=1,keepdim=True)
    lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    return F.conv2d(crops_g,lap,padding=1).flatten(2).var(dim=2).squeeze(1)

def metric_dct_concentration(crops):
    """DCT low-mid freq concentration via padding + FFT (Type-I DCT approximation)."""
    M,C,H,W=crops.shape; crops_g=crops.mean(dim=1)  # (M, H, W)
    padded = F.pad(crops_g, (0, W, 0, H), mode='reflect')  # (M, 2H, 2W)
    fft = torch.fft.rfft2(padded, dim=(-2,-1), norm="ortho"); amp = torch.abs(fft)
    Hf, Wf = fft.shape[-2], fft.shape[-1]
    total = amp.flatten(2).sum(2).clamp_min(1e-8)  # (M, 1)
    lo = amp[:, :Hf//2, :Wf//2].flatten(2).sum(2)  # (M, 1)
    return (lo/total).squeeze(1)

def metric_autocorr_peak(crops):
    """Autocorrelation peak sharpness via FFT. power_spectrum = |FFT|^2, autocorr = iFFT(power_spectrum).
    Sharp peak at center = strong structure."""
    M,C,H,W=crops.shape; crops_g=crops.mean(dim=1,keepdim=True)  # (M, 1, H, W)
    fft=torch.fft.rfft2(crops_g,dim=(-2,-1),norm="ortho"); ps=torch.abs(fft)**2
    ac=torch.fft.irfft2(ps,dim=(-2,-1),norm="ortho",s=(H,W)).squeeze(1)  # (M, H, W)
    peak=ac[:,H//2,W//2]; surround=ac.flatten(1).sum(dim=1)/(H*W)
    return peak/surround.clamp_min(1e-8)

def metric_vert_edge_ratio(crops):
    """Ratio of vertical to horizontal edge energy. Pedestrians = more vertical."""
    M,C,H,W=crops.shape; crops_g=crops.mean(dim=1,keepdim=True)
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    sy=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    gx=F.conv2d(crops_g,sx,padding=1);gy=F.conv2d(crops_g,sy,padding=1)
    v=(gx**2).flatten(2).sum(2); h=(gy**2).flatten(2).sum(2)
    return (v/(v+h+1e-8)).squeeze(1)

def metric_phase_std(crops):
    """Std of FFT phase spectrum — structured objects have coherent phase."""
    M,C,H,W=crops.shape; fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho"); pha=torch.angle(fft)
    return pha.flatten(2).std(dim=2).mean(dim=1)

METRICS = {
    "energy": metric_energy,
    "edge_quality": metric_edge_quality,
    "entropy": metric_entropy,
    "grad_std": metric_grad_std,
    "center_surround": metric_center_surround,
    "laplacian_var": metric_laplacian_var,
    # "dct_concentration": metric_dct_concentration,  # padding bug on 7x7
    "autocorr_peak": metric_autocorr_peak,
    "vert_edge_ratio": metric_vert_edge_ratio,
    "phase_std": metric_phase_std,
}

# ===== SINGLE IMAGE ANALYSIS =====
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

    # Analyze ALL proposals on the first image
    all_proposal_results = []
    for k in range(N):
        base = base_boxes[k]; i_img = prop_to_img[k]
        bases = base.unsqueeze(0).repeat(N_ACTIONS, 1)
        aidx = torch.arange(N_ACTIONS, device=DEV)
        refined = apply_actions_batch(bases, aidx)
        refined[:,2] = refined[:,2].clamp(max=img_shape[1]-1); refined[:,3] = refined[:,3].clamp(max=img_shape[0]-1)

        gt = tgts[i_img]["boxes"].to(DEV)
        ious = box_iou(refined, gt).max(dim=1).values if len(gt)>0 else torch.zeros(N_ACTIONS,device=DEV)
        base_iou = box_iou(base.unsqueeze(0).cpu(), gt.cpu()).max().item() if len(gt)>0 else 0

        with torch.no_grad(): crops = crop_img(raw, refined)

        metric_vals = {}
        for name, fn in METRICS.items():
            metric_vals[name] = fn(crops).cpu().numpy()  # (24,)

        iou_vals = ious.detach().cpu().numpy()
        best_iou_idx = ious.argmax()

        row = {"base_iou": base_iou, "best_iou": best_iou_idx, "iou_vals": iou_vals}
        for name in METRICS:
            vals = metric_vals[name]
            row[name] = {
                "vals": vals,
                "picks_best": int(np.argmax(vals) == best_iou_idx),
                "corr": np.corrcoef(vals, iou_vals)[0,1] if np.std(vals)>1e-8 and np.std(iou_vals)>1e-8 else 0,
            }
        all_proposal_results.append(row)

    # Aggregate across all proposals on this image
    print(f"\n=== Single Image Analysis: {len(all_proposal_results)} proposals on image 1 ===")
    print(f"{'Metric':<20s} {'Best%':>7s} {'Top3%':>7s} {'Top5%':>7s} {'Spearman':>9s} {'FN':>6s} {'BL':>6s} {'TP':>6s}")
    print("-" * 80)

    for name in METRICS:
        bests = [r[name]["picks_best"] for r in all_proposal_results]
        corrs = [r[name]["corr"] for r in all_proposal_results]

        # By base IoU
        fn_best = [r[name]["picks_best"] for r in all_proposal_results if r["base_iou"] < 0.5]
        bl_best = [r[name]["picks_best"] for r in all_proposal_results if 0.5 <= r["base_iou"] < 0.75]
        tp_best = [r[name]["picks_best"] for r in all_proposal_results if r["base_iou"] >= 0.75]

        # Top-3/5
        top3 = [1 if r["best_iou"] in set(np.argsort(r[name]["vals"])[::-1][:3]) else 0 for r in all_proposal_results]
        top5 = [1 if r["best_iou"] in set(np.argsort(r[name]["vals"])[::-1][:5]) else 0 for r in all_proposal_results]

        fn_s = f"{100*np.mean(fn_best):.0f}%" if fn_best else "--"
        bl_s = f"{100*np.mean(bl_best):.0f}%" if bl_best else "--"
        tp_s = f"{100*np.mean(tp_best):.0f}%" if tp_best else "--"

        print(f"{name:<20s} {100*np.mean(bests):6.1f}% {100*np.mean(top3):6.1f}% {100*np.mean(top5):6.1f}% {np.mean(corrs):+8.4f} {fn_s:>6s} {bl_s:>6s} {tp_s:>6s}")

    # Best individual proposal
    best_prop = max(all_proposal_results, key=lambda r: max(r["iou_vals"]) - min(r["iou_vals"]))
    print(f"\nBest proposal: base_IoU={best_prop['base_iou']:.3f}, IoU_range=[{best_prop['iou_vals'].min():.3f},{best_prop['iou_vals'].max():.3f}]")
    print(f"Action ranking (best -> worst by IoU):")
    ranks = np.argsort(-best_prop["iou_vals"])[:5]
    for rank, idx in enumerate(ranks):
        print(f"  #{rank+1}: action {idx:2d}, IoU={best_prop['iou_vals'][idx]:.4f}")

    # Only analyze first image
    break
