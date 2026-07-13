"""Test: 32x32 crop energy sensitivity vs 7x7."""
import sys, math
import torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV = "cuda"; SEED = 42; G = 4
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
set_seed(SEED)

model = build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"]); model.eval()

fpn_feats = {}
model.backbone.register_forward_hook(lambda m,i,o: fpn_feats.update({"f":{k:o[k] for k in o if k!="pool"}}))

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

def energy_fft(crops):
    M,C,H,W=crops.shape
    fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho");amp=torch.abs(fft)
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing='ij');r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.3).float()
    al=(amp*lo).flatten(2).sum(2);at=al+(amp*((rn>0.3)&(rn<=0.7)).float()).flatten(2).sum(2)+(amp*(rn>0.7).float()).flatten(2).sum(2)+1e-8
    return (al/at).mean(dim=1)

def phase_var(crops):
    """Variance of phase spectrum — structured objects have lower variance."""
    M,C,H,W=crops.shape
    fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho")
    pha=torch.angle(fft); amp=torch.abs(fft)
    wvar=(pha**2 * amp).flatten(2).sum(2)/amp.flatten(2).sum(2).clamp_min(1e-8)
    return wvar.mean(dim=1)

def energy_raw_pixel(crops):
    """Simple raw pixel energy: mean squared pixel value. Structured objects have more variation."""
    return (crops**2).flatten(2).mean(dim=2).mean(dim=1)

results = {}

for size in [7, 16, 32]:
    results[size] = {"energy": [], "phase": [], "raw": [], "iou": []}
    for sigma in [0.1, 0.2, 0.5, 1.0]:
        results[size][f"e@{sigma}"] = []

for imgs, tgts in vl:
    imgs_d = [i.to(DEV) for i in imgs]; img_shape = (imgs_d[0].shape[-2], imgs_d[0].shape[-1])
    raw = imgs_d[0]
    fpn_feats.clear()
    with torch.no_grad(): _ = model(imgs_d, [{k:v.to(DEV) for k,v in t.items()} for t in tgts])

    for tgt in tgts:
        for box in tgt["boxes"].to(DEV):
            w=box[2]-box[0];h=box[3]-box[1];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2

            for sigma in [0.1, 0.2, 0.5, 1.0]:
                dx=sigma*torch.randn(G,device=DEV);dy=sigma*torch.randn(G,device=DEV)
                dw=sigma*torch.randn(G,device=DEV);dh=sigma*torch.randn(G,device=DEV)
                g_cx=dx*w+cx;g_cy=dy*h+cy;g_w=torch.exp(dw)*w;g_h=torch.exp(dh)*h
                boxes_g=torch.stack([g_cx-0.5*g_w,g_cy-0.5*g_h,g_cx+0.5*g_w,g_cy+0.5*g_h],dim=1).clamp(min=0)
                boxes_g[:,2]=boxes_g[:,2].clamp(max=img_shape[1]-1)
                boxes_g[:,3]=boxes_g[:,3].clamp(max=img_shape[0]-1)

                ious=box_iou(boxes_g,box.unsqueeze(0)).squeeze()

                for size in [7, 16, 32]:
                    crops=crop_img(raw,boxes_g,out=size)
                    en=energy_fft(crops); pv=phase_var(crops); rw=energy_raw_pixel(crops)
                    results[size]["energy"].append(en.var().item())
                    results[size]["phase"].append(pv.var().item())
                    results[size]["raw"].append(rw.var().item())
                    results[size]["iou"].append(ious.var().item())
                    results[size][f"e@{sigma}"].append(en.var().item())

print(f"{'Size':>5s} {'energy':>10s} {'phase_var':>10s} {'raw_pixel':>10s} {'iou':>10s}")
print("-" * 48)
for size in [7, 16, 32]:
    ev=np.mean(results[size]["energy"]); pv=np.mean(results[size]["phase"])
    rv=np.mean(results[size]["raw"]); iv=np.mean(results[size]["iou"])
    print(f"{size:5d} {ev:10.6f} {pv:10.6f} {rv:10.6f} {iv:10.6f}")

# Target: which crop size gives energy >> iou?
print(f"\n=== Energy/IoU ratio by crop size ===")
print(f"{'Size':>5s}", end="")
for sigma in [0.1, 0.2, 0.5, 1.0]:
    print(f" {'σ='+str(sigma):>10s}", end="")
print()
for size in [7, 16, 32]:
    print(f"{size:5d}", end="")
    for sigma in [0.1, 0.2, 0.5, 1.0]:
        ev=np.mean(results[size][f"e@{sigma}"])
        iv=np.mean(results[size]["iou"])
        print(f" {ev/max(iv,1e-8):9.1f}x", end="")
    print()
