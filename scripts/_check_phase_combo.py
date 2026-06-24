"""Test energy + |phase| combo at multiple crop sizes."""
import sys, torch, numpy as np,torch.nn.functional as F
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV="cuda";set_seed(42)
def build_actions():
    s=[0.02,0.05,0.10,0.20];a=[]
    for x in s:a.extend([(x,0,0,0),(-x,0,0,0),(0,x,0,0),(0,-x,0,0)])
    for x in s:a.extend([(0,0,x,0),(0,0,-x,0)])
    return torch.tensor(a,dtype=torch.float32)
ACT=build_actions().to(DEV);NA=24

model=build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}).to(DEV)
ckpt=torch.load("runs/round227_v1_baseline_20ep/checkpoint_best.pth",map_location=DEV);model.load_state_dict(ckpt["model"]);model.eval()
sp,bi={},{}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bi.update({"x":a[0]}))
tl,vl=build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})

def crop(raw,b,out=7):
    if raw.dim()==4:raw=raw[0]
    M=b.shape[0];_,H,W=raw.shape;cs=[]
    for i in range(M):
        x1,y1,x2,y2=b[i].long().clamp(0);x1=x1.clamp(max=W-2);x2=x2.clamp(max=W-1);y1=y1.clamp(max=H-2);y2=y2.clamp(max=H-1)
        if x2<=x1+1:x2=x1+2
        if y2<=y1+1:y2=y1+2
        c=raw[:,y1:y2,x1:x2].float()/255.0
        cs.append(F.interpolate(c.unsqueeze(0),size=(out,out),mode="bilinear",align_corners=False).squeeze(0))
    return torch.stack(cs)

def all_feat(crops):
    M,Ch,H,W=crops.shape
    f=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho");amp=torch.abs(f);pha=torch.angle(f)
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing="ij");r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.3).float();md=((rn>0.3)&(rn<=0.7)).float();hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2);am=(amp*md).flatten(2).sum(2);ah=(amp*hi).flatten(2).sum(2)
    en=(al/(al+am+ah+1e-8)).mean(dim=1)
    def aph(mask):return -((torch.abs(pha)*amp*mask).flatten(2).sum(2)/(amp*mask).flatten(2).sum(2).clamp_min(1e-8)).mean(dim=1)
    return en, aph(lo), aph(hi)

def apply_actions(boxes,idx):
    a=ACT[idx];w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    nc=cx+a[:,0]*w;ny=cy+a[:,1]*h;nw=torch.clamp(w*(1.0+a[:,2]),min=1);nh=torch.clamp(h*(1.0+a[:,3]),min=1)
    return torch.stack([nc-0.5*nw,ny-0.5*nh,nc+0.5*nw,ny+0.5*nh],dim=1).clamp(min=0)

print(f"{'Size':>6s} {'energy':>8s} {'|ph|_hi':>8s} {'en*|ph|':>8s}")
print("-"*35)

for crop_size in [7, 11, 15, 21]:
    res=[]
    for imgs,tgts in vl:
        imd=[i.to(DEV) for i in imgs];sh=(imd[0].shape[-2],imd[0].shape[-1]);raw=imd[0]
        sp.clear();bi.clear()
        with torch.no_grad():_=model(imd,[{k:v.to(DEV) for k,v in t.items()} for t in tgts])
        rf=bi.get("x");sr=sp.get("p")
        if rf is None or sr is None or rf.shape[0]==0:continue
        nn=rf.shape[0];sc=torch.cat(sr,dim=0)[:nn]
        bf=model.roi_heads.box_head(rf);mu=model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
        bw=sc[:,2]-sc[:,0];bh=sc[:,3]-sc[:,1];bc=sc[:,0]+0.5*bw;by=sc[:,1]+0.5*bh
        dx_b=mu[:,0]/10.0;dy_b=mu[:,1]/10.0;dw_b=mu[:,2]/5.0;dh_b=mu[:,3]/5.0
        base=torch.stack([dx_b*bw+bc-0.5*torch.exp(dw_b)*bw,dy_b*bh+by-0.5*torch.exp(dh_b)*bh,dx_b*bw+bc+0.5*torch.exp(dw_b)*bw,dy_b*bh+by+0.5*torch.exp(dh_b)*bh],dim=1).clamp(min=0)
        p2i=[i for i,p in enumerate(sr) for _ in range(p.shape[0])]
        bs=base.unsqueeze(1).repeat(1,NA,1);bs=bs.reshape(-1,4)
        aidx=torch.arange(NA,device=DEV).repeat(nn)
        ref=apply_actions(bs,aidx);ref[:,2]=ref[:,2].clamp(max=sh[1]-1);ref[:,3]=ref[:,3].clamp(max=sh[0]-1)

        with torch.no_grad(): cs=crop(raw,ref,out=crop_size);en,pl,ph=all_feat(cs)
        en_v=en.view(nn,NA).cpu().numpy();ph_v=ph.view(nn,NA).cpu().numpy();ep_v=(en*ph).view(nn,NA).cpu().numpy()

        for k in range(nn):
            gt=tgts[p2i[k]]["boxes"].to(DEV);ious=box_iou(ref[k*NA:(k+1)*NA],gt).max(dim=1).values if len(gt)>0 else torch.zeros(NA,device=DEV)
            bi_idx=ious.argmax().item()
            res.append({"bi":bi_idx,"en":en_v[k],"ph":ph_v[k],"ep":ep_v[k]})

    e_best=np.mean([1 if np.argmax(r["en"])==r["bi"] else 0 for r in res])*100
    p_best=np.mean([1 if np.argmax(r["ph"])==r["bi"] else 0 for r in res])*100
    ep_best=np.mean([1 if np.argmax(r["ep"])==r["bi"] else 0 for r in res])*100
    print(f"{crop_size:6d} {e_best:7.1f}% {p_best:7.1f}% {ep_best:7.1f}%")
print(f"{'Random':>6s}   4.2%")
