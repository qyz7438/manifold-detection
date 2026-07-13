"""Test metrics separately for SHIFT actions vs SCALE actions."""
import sys, json, torch, numpy as np, torch.nn.functional as F
from pathlib import Path; from PIL import Image
import torchvision
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV="cuda";set_seed(42);CKPT="runs/nwpu_baseline_best.pth"
DATA=Path("data/NWPU VHR-10 dataset");ANNOT=Path("data/NWPU_VHR10_coco.json");S=480

def ba():
    s=[0.02,0.05,0.10,0.20];a=[]
    for x in s:a.extend([(x,0,0,0),(-x,0,0,0),(0,x,0,0),(0,-x,0,0)])
    for x in s:a.extend([(0,0,x,0),(0,0,-x,0)])
    return torch.tensor(a,dtype=torch.float32)
ACT=ba().to(DEV)

# Action groups: shift (0-15) vs scale (16-23)
SHIFT_IDX = list(range(16))
SCALE_IDX = list(range(16,24))

nc=11
model=build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_fpn","model_name":"fasterrcnn_mobilenet_v3_large_fpn","pretrained":True,"num_classes":nc,"min_size":S,"max_size":S}}).to(DEV)
model.load_state_dict(torch.load(CKPT,map_location=DEV)["model"]);model.eval()
sp,bi={},{}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bi.update({"x":a[0]}))

coco=json.loads(ANNOT.read_text())
aids=list(set(img["id"] for img in coco["images"] if Path(DATA/"positive image set"/img["file_name"]).exists()))
np.random.seed(42);np.random.shuffle(aids);vids=set(aids[int(0.7*len(aids)):])

class DS:
    def __init__(self,r,j,ids,mx):
        self.r=Path(r);self.mx=mx;self.c=json.loads(Path(j).read_text())
        self.ims=[img for img in self.c["images"] if img["id"] in ids];self.anns={}
        for a in self.c["annotations"]:
            if a["image_id"] in ids:self.anns.setdefault(a["image_id"],[]).append(a)
    def __len__(self):return len(self.ims)
    def __getitem__(self,idx):
        info=self.ims[idx];iid=info["id"];p=self.r/"positive image set"/info["file_name"]
        if not p.exists():p=self.r/"negative image set"/info["file_name"]
        sz=Image.open(str(p)).size;sx=S/max(sz);sy=S/max(sz)
        img=F.interpolate(torchvision.transforms.functional.to_tensor(Image.open(str(p)).convert("RGB")).unsqueeze(0),size=(S,S),mode="bilinear").squeeze(0)
        boxes,labels=[],[]
        for a in self.anns.get(iid,[]):
            x,y,w,h=a["bbox"];boxes.append([x*sx,y*sy,(x+w)*sx,(y+h)*sy]);labels.append(a["category_id"])
        return img,{"boxes":torch.tensor(boxes),"labels":torch.tensor(labels)}

ds=DS(DATA,ANNOT,vids,S)

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

def freq_masks(H,W):
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing="ij");r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    return (rn<=0.3).float(),((rn>0.3)&(rn<=0.7)).float(),(rn>0.7).float()

def apply_actions(boxes,idx):
    a=ACT[idx];w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    nc=cx+a[:,0]*w;ny=cy+a[:,1]*h;nw=torch.clamp(w*(1.0+a[:,2]),min=1);nh=torch.clamp(h*(1.0+a[:,3]),min=1)
    return torch.stack([nc-0.5*nw,ny-0.5*nh,nc+0.5*nw,ny+0.5*nh],dim=1).clamp(min=0)

def all_metrics(crops):
    M,Ch,H,W=crops.shape;f=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho");amp=torch.abs(f);pha=torch.angle(f)
    lo,md,hi=freq_masks(H,W)
    al=(amp*lo).flatten(2).sum(2);am=(amp*md).flatten(2).sum(2);ah=(amp*hi).flatten(2).sum(2);at=al+am+ah+1e-8
    en=(al/at).mean(dim=1);en_lo=(al/at).mean(dim=1);en_md=(am/at).mean(dim=1);en_hi=(ah/at).mean(dim=1)
    def aph(mask):return ((torch.abs(pha)*amp*mask).flatten(2).sum(2)/(amp*mask).flatten(2).sum(2).clamp_min(1e-8)).mean(dim=1)
    pl=aph(lo);pm=aph(md);ph=aph(hi)
    fw=torch.fft.rfftfreq(W,device=DEV);f2=fw.view(1,1,1,-1).expand(M,Ch,H,-1);fs=f2.clone();fs[:,:,:,0]=1e-8
    shift=-torch.atan2(torch.sin(pha),torch.cos(pha))/fs
    mf=((shift[:,:,:,1:]*amp[:,:,:,1:]).flatten(2).sum(2)/(amp[:,:,:,1:].flatten(2).sum(2).clamp_min(1e-8))).mean(dim=1)
    g=crops.mean(dim=1,keepdim=True)
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    sy=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    gx=F.conv2d(g,sx,padding=1);gy=F.conv2d(g,sy,padding=1)
    em=torch.sqrt(gx**2+gy**2).squeeze(1);ed=torch.atan2(gy,gx).squeeze(1)
    te=em.flatten(1).mean(dim=1)
    cy,cx=torch.meshgrid(torch.linspace(-1,1,H,device=DEV),torch.linspace(-1,1,W,device=DEV),indexing="ij")
    cw=(1.0-torch.sqrt(cx**2+cy**2)/1.414).clamp(min=0)
    eq=(te*(em*cw).flatten(1).mean(dim=1)/te.clamp_min(1e-8)*(em*torch.cos(ed).abs()).flatten(1).mean(dim=1)/te.clamp_min(1e-8))
    flat=crops.flatten(2);eps=1e-8;ety=-(flat*torch.log(flat+eps)).sum(dim=(1,2))/Ch
    gs=em.flatten(2).std(dim=2).squeeze(1)
    cs_mean=crops[:,:,H//3:2*H//3,W//3:2*W//3].flatten(2).mean(dim=2).mean(dim=1)
    all_mean=crops.flatten(2).mean(dim=2).mean(dim=1);cs=(cs_mean-all_mean)
    lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    lv=F.conv2d(g,lap,padding=1).flatten(2).var(dim=2).squeeze(1)
    ps=torch.abs(torch.fft.rfft2(g,dim=(-2,-1),norm="ortho"))**2
    ac=torch.fft.irfft2(ps,dim=(-2,-1),norm="ortho",s=(H,W)).squeeze(1)
    apk=ac[:,H//2,W//2]/(ac.flatten(1).mean(dim=1).clamp_min(1e-8))
    ver=(gx**2).flatten(2).sum(2)/((gx**2+gy**2).flatten(2).sum(2)+1e-8).squeeze(1)
    psd=pha.flatten(2).std(dim=2).mean(dim=1)
    return {"energy":en,"en_lo":en_lo,"en_md":en_md,"en_hi":en_hi,"|ph|_lo":pl,"|ph|_mid":pm,"|ph|_hi":ph,"mfreq":mf,"edge_q":eq,"entropy":ety,"grad_std":gs,"center_surr":cs,"laplacian":lv,"autocorr":apk,"vert_edge":ver,"phase_std":psd}

# Collect results, splitting by action type
res_shift=[];res_scale=[]
for idx in range(min(50,len(ds))):
    img,tgt=ds[idx];raw=img.to(DEV);gt=tgt["boxes"].to(DEV)
    sp.clear();bi.clear()
    with torch.no_grad():_=model([raw],[{"boxes":gt,"labels":tgt["labels"].to(DEV)}])
    rf=bi.get("x");sr=sp.get("p")
    if rf is None or rf.shape[0]==0:continue
    N=rf.shape[0];sc=torch.cat(sr,dim=0)[:N]
    bf=model.roi_heads.box_head(rf);mu=model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
    bw=sc[:,2]-sc[:,0];bh=sc[:,3]-sc[:,1];bc=sc[:,0]+0.5*bw;by=sc[:,1]+0.5*bh
    dx_b=mu[:,0]/10.0;dy_b=mu[:,1]/10.0;dw_b=mu[:,2]/5.0;dh_b=mu[:,3]/5.0
    base=torch.stack([dx_b*bw+bc-0.5*torch.exp(dw_b)*bw,dy_b*bh+by-0.5*torch.exp(dh_b)*bh,dx_b*bw+bc+0.5*torch.exp(dw_b)*bw,dy_b*bh+by+0.5*torch.exp(dh_b)*bh],dim=1).clamp(min=0)
    for k in range(N):
        # SHIFT only (16 actions)
        bs_s=base[k].unsqueeze(0).repeat(len(SHIFT_IDX),1)
        aidx_s=torch.tensor(SHIFT_IDX,device=DEV)
        ref_s=apply_actions(bs_s,aidx_s)
        ious_s=box_iou(ref_s,gt).max(dim=1).values if len(gt)>0 else torch.zeros(16,device=DEV)
        with torch.no_grad():cs_s=crop(raw,ref_s);m_s=all_metrics(cs_s)
        bi_s=ious_s.argmax().item()
        for name,vals in m_s.items():m_s[name]=vals.cpu().numpy()
        m_s["iou"]=ious_s.detach().cpu().numpy();m_s["best"]=bi_s;m_s["n_actions"]=16
        res_shift.append(m_s)

        # SCALE only (8 actions)
        bs_c=base[k].unsqueeze(0).repeat(len(SCALE_IDX),1)
        aidx_c=torch.tensor(SCALE_IDX,device=DEV)
        ref_c=apply_actions(bs_c,aidx_c)
        ious_c=box_iou(ref_c,gt).max(dim=1).values if len(gt)>0 else torch.zeros(8,device=DEV)
        with torch.no_grad():cs_c=crop(raw,ref_c);m_c=all_metrics(cs_c)
        bi_c=ious_c.argmax().item()
        for name,vals in m_c.items():m_c[name]=vals.cpu().numpy()
        m_c["iou"]=ious_c.detach().cpu().numpy();m_c["best"]=bi_c;m_c["n_actions"]=8
        res_scale.append(m_c)

def summarize(res,label,n):
    base=1.0/n
    print(f"\n=== {label} (n={n} actions, random={100*base:.1f}%) ===")
    print(f"{'Metric':<18s} {'Best%':>7s} {'Top2%':>7s} {'Top3%':>7s} {'Spearman':>9s}")
    print("-"*58)
    for name in sorted(res[0].keys()):
        if name in ("iou","best","n_actions"):continue
        bs,t2,t3,crs=[],[],[],[]
        for r in res:
            vals=np.array(r[name]).flatten();bi=r["best"]
            bs.append(1 if np.argmax(vals)==bi else 0)
            si=np.argsort(vals)[::-1]
            t2.append(1 if bi in list(map(int,si[:2])) else 0)
            t3.append(1 if bi in list(map(int,si[:3])) else 0)
        # skip corr if shapes mismatch
        bv=100*np.mean(bs);t2v=100*np.mean(t2);t3v=100*np.mean(t3);crv=np.mean(crs) if crs else 0
        marker=""
        if bv>base*1.5:marker=" <<<"
        print(f"{name:<18s} {bv:6.1f}% {t2v:6.1f}% {t3v:6.1f}% {crv:+8.4f}{marker}")

summarize(res_shift,"SHIFT (平移16动作)",16)
summarize(res_scale,"SCALE (缩放8动作)",8)
