"""Multi-frequency phase: recover shift DIRECTION from frequency-varying phase changes.
Phase shift theorem: phase_diff(w) = -w * spatial_shift.
Different frequencies encode same spatial shift with different phase changes.
The linear fit of phase_diff vs frequency recovers shift magnitude AND direction."""
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

def multifreq_phase_direction(crops, ref_crop):
    """Extract spatial shift direction from multi-frequency phase analysis.
    Returns (M,) scalar: signed shift along dominant direction.
    Positive = rightward, Negative = leftward (relative to reference)."""
    M = crops.shape[0]; _, H, W = crops.shape[1], crops.shape[-2], crops.shape[-1]
    fft = torch.fft.rfft2(crops, dim=(-2,-1), norm="ortho")
    fft_ref = torch.fft.rfft2(ref_crop.expand(M,-1,-1,-1), dim=(-2,-1), norm="ortho")

    pha = torch.angle(fft); pha_ref = torch.angle(fft_ref)
    amp = torch.abs(fft)

    # Phase difference (circular), unwrapped
    pha_diff = torch.atan2(torch.sin(pha-pha_ref), torch.cos(pha-pha_ref))  # (M, C, H, W_rfft)

    # Frequencies along x-axis (the rfft dimension)
    freq_x = torch.fft.rfftfreq(W, device=DEV)  # (W_rfft,)

    # For each frequency, compute the phase diff / freq ratio
    # Weight by amplitude
    freq_x_2d = freq_x.view(1, 1, 1, -1).expand(M, crops.shape[1], H, -1)  # (M, C, H, W_rfft)

    # Avoid division by zero at DC (freq=0)
    freq_safe = freq_x_2d.clone(); freq_safe[:,:,:,0] = 1e-8  # DC component

    # phase/freq should be ~constant = -spatial_shift_x
    shift_estimate = -pha_diff / freq_safe  # (M, C, H, W_rfft)

    # Weighted average across frequencies (excluding DC), weighted by amplitude
    weights = amp[:,:,:,1:]  # exclude DC
    shifts = shift_estimate[:,:,:,1:]  # exclude DC

    # Mean shift per channel, then per sample
    w_sum = weights.flatten(2).sum(2).clamp_min(1e-8)  # (M, C)
    shift_avg = (shifts * weights).flatten(2).sum(2) / w_sum  # (M, C)

    return shift_avg.mean(dim=1)  # (M,) mean across channels

def apply_actions_batch(boxes, action_indices):
    a=ACTIONS[action_indices]
    w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    new_cx=cx+a[:,0]*w;new_cy=cy+a[:,1]*h
    new_w=torch.clamp(w*(1.0+a[:,2]),min=1);new_h=torch.clamp(h*(1.0+a[:,3]),min=1)
    x1=new_cx-0.5*new_w;y1=new_cy-0.5*new_h;x2=new_cx+0.5*new_w;y2=new_cy+0.5*new_h
    return torch.stack([x1,y1,x2,y2],dim=1).clamp(min=0)

# Test: for each proposal, compute multifreq phase shift direction vs actual box shift
shift_preds = []  # (estimated_shift_x, actual_dx)
action_results = []

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

    for k in range(min(N, 20)):
        base = base_boxes[k]; i_img = prop_to_img[k]
        bases = base.unsqueeze(0).repeat(N_ACTIONS, 1)
        aidx = torch.arange(N_ACTIONS, device=DEV)
        refined = apply_actions_batch(bases, aidx)
        refined[:,2] = refined[:,2].clamp(max=img_shape[1]-1)
        refined[:,3] = refined[:,3].clamp(max=img_shape[0]-1)

        with torch.no_grad():
            crops = crop_img(raw, refined)
            ref_crop = crop_img(raw, base.unsqueeze(0))
            mf = multifreq_phase_direction(crops, ref_crop)

        # Actual shift: cx of each refined box minus cx of base
        base_cx = (base[0]+base[2])/2
        refined_cx = (refined[:,0]+refined[:,2])/2
        actual_dx = refined_cx - base_cx.item()  # (24,)

        # Correlation: does estimated shift correlate with actual shift?
        mfv = mf.detach().cpu().numpy(); adv = actual_dx.detach().cpu().numpy()
        if np.std(mfv) > 1e-8 and np.std(adv) > 1e-8:
            c = np.corrcoef(mfv, adv)[0,1]
            shift_preds.append(c)
            c_val = c
        else:
            c_val = 0

        # Action ranking: does multifreq phase pick the best IoU action?
        gt = tgts[i_img]["boxes"].to(DEV)
        ious = box_iou(refined, gt).max(dim=1).values if len(gt)>0 else torch.zeros(N_ACTIONS, device=DEV)
        best_idx = ious.argmax().item()

        # "Better" here means: shift toward GT center. Use phase as proxy.
        mf_best = mf.argmin().item() if mf.abs().mean() > 0 else 0

        action_results.append({
            "mf_best_idx": mf_best,
            "iou_best_idx": best_idx,
            "corr": c if 'c' in dir() else 0,
        })

print(f"Proposals analyzed: {len(action_results)}")
print(f"Actions analyzed: {len(shift_preds)}")

if shift_preds:
    print(f"\nMulti-freq phase vs actual shift correlation:")
    print(f"  mean r: {np.mean(shift_preds):.4f}")
    print(f"  median r: {np.median(shift_preds):.4f}")
    print(f"  % positive: {100*sum(1 for c in shift_preds if c>0)/len(shift_preds):.1f}%")
    print(f"  % |r|>0.5: {100*sum(1 for c in shift_preds if abs(c)>0.5)/len(shift_preds):.1f}%")

if action_results:
    mf_correct = sum(1 for r in action_results if r["mf_best_idx"]==r["iou_best_idx"])
    print(f"\nMulti-freq phase picks best IoU: {mf_correct}/{len(action_results)} = {100*mf_correct/len(action_results):.1f}%")
