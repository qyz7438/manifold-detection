"""NWPU: does energy/phase/edge rank 24 actions? Same test as Penn-Fudan."""
import sys, json, torch, numpy as np, torch.nn.functional as F
from pathlib import Path
from PIL import Image
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
from torchvision.ops import box_iou

DEV="cuda";set_seed(42)
CKPT="runs/nwpu_baseline_best.pth"
DATA=Path("data/NWPU VHR-10 dataset");ANNOT=Path("data/NWPU_VHR10_coco.json")
MAX_SIZE=480

def build_actions():
    s=[0.02,0.05,0.10,0.20];a=[]
    for x in s:a.extend([(x,0,0,0),(-x,0,0,0),(0,x,0,0),(0,-x,0,0)])
    for x in s:a.extend([(0,0,x,0),(0,0,-x,0)])
    return torch.tensor(a,dtype=torch.float32)
ACT=build_actions().to(DEV);NA=24

num_classes=11
model=build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_fpn","model_name":"fasterrcnn_mobilenet_v3_large_fpn","pretrained":True,"num_classes":num_classes,"min_size":MAX_SIZE,"max_size":MAX_SIZE}}).to(DEV)
model.load_state_dict(torch.load(CKPT,map_location=DEV)["model"]);model.eval()

sp,bi={},{}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bi.update({"x":a[0]}))

coco=json.loads(ANNOT.read_text())
all_ids=list(set(img["id"] for img in coco["images"] if Path(DATA/"positive image set"/img["file_name"]).exists()))
np.random.seed(42);np.random.shuffle(all_ids)
val_ids=set(all_ids[int(0.7*len(all_ids)):])

class NWPUDataset:
    def __init__(self, root, coco_json, img_ids, max_size):
        self.root=Path(root);self.max_size=max_size
        self.coco=json.loads(Path(coco_json).read_text())
        self.imgs=[img for img in self.coco["images"] if img["id"] in img_ids]
        self.anns={}
        for a in self.coco["annotations"]:
            if a["image_id"] in img_ids:self.anns.setdefault(a["image_id"],[]).append(a)
    def __len__(self):return len(self.imgs)
    def __getitem__(self,idx):
        info=self.imgs[idx];img_id=info["id"]
        p=self.root/"positive image set"/info["file_name"]
        if not p.exists():p=self.root/"negative image set"/info["file_name"]
        img=F.interpolate(torchvision.transforms.functional.to_tensor(Image.open(str(p)).convert("RGB")).unsqueeze(0),size=(MAX_SIZE,MAX_SIZE),mode="bilinear").squeeze(0)
        boxes,labels=[],[]
        scale_x=MAX_SIZE/max(Image.open(str(p)).size);scale_y=MAX_SIZE/max(Image.open(str(p)).size)
        for a in self.anns.get(img_id,[]):
            x,y,w,h=a["bbox"];boxes.append([x*scale_x,y*scale_y,(x+w)*scale_x,(y+h)*scale_y]);labels.append(a["category_id"])
        return img,{"boxes":torch.tensor(boxes), "labels":torch.tensor(labels)}

import torchvision, torchvision.transforms as T
ds=NWPUDataset(DATA,ANNOT,val_ids,MAX_SIZE)

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

def energy(crops):
    M,C,H,W=crops.shape
    f=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho");amp=torch.abs(f)
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing="ij");r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.3).float();md=((rn>0.3)&(rn<=0.7)).float();hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2);am=(amp*md).flatten(2).sum(2);ah=(amp*hi).flatten(2).sum(2)
    return (al/(al+am+ah+1e-8)).mean(dim=1)

def edge_quality(crops):
    M,C,H,W=crops.shape;g=crops.mean(dim=1,keepdim=True)
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    sy=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=DEV).view(1,1,3,3)
    gx=F.conv2d(g,sx,padding=1);gy=F.conv2d(g,sy,padding=1)
    em=torch.sqrt(gx**2+gy**2).squeeze(1);ed=torch.atan2(gy,gx).squeeze(1)
    te=em.flatten(1).mean(dim=1)
    cy,cx=torch.meshgrid(torch.linspace(-1,1,H,device=DEV),torch.linspace(-1,1,W,device=DEV),indexing="ij")
    cw=(1.0-torch.sqrt(cx**2+cy**2)/1.414).clamp(min=0)
    ce=(em*cw).flatten(1).mean(dim=1)/te.clamp_min(1e-8)
    vw=torch.cos(ed).abs();vr=(em*vw).flatten(1).mean(dim=1)/te.clamp_min(1e-8)
    return te*ce*vr

def apply_actions(boxes,idx):
    a=ACT[idx];w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    nc=cx+a[:,0]*w;ny=cy+a[:,1]*h;nw=torch.clamp(w*(1.0+a[:,2]),min=1);nh=torch.clamp(h*(1.0+a[:,3]),min=1)
    return torch.stack([nc-0.5*nw,ny-0.5*nh,nc+0.5*nw,ny+0.5*nh],dim=1).clamp(min=0)

res=[]; n_samples=0
for idx in range(min(50,len(ds))):
    img,tgt=ds[idx]; raw=img.to(DEV); gt=tgt["boxes"].to(DEV)
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
        bs=base[k].unsqueeze(0).repeat(NA,1);aidx=torch.arange(NA,device=DEV)
        ref=apply_actions(bs,aidx)
        ious=box_iou(ref,gt).max(dim=1).values if len(gt)>0 else torch.zeros(NA,device=DEV)
        with torch.no_grad():cs=crop(raw,ref);en=energy(cs);eq=edge_quality(cs)
        best_iou=ious.argmax().item()
        res.append({"best":best_iou,"en":en.cpu().numpy(),"eq":eq.cpu().numpy()})
        n_samples+=1

print(f"NWPU: {n_samples} proposals on {min(50,len(ds))} val images\n")
for name,key in [("Energy","en"),("Edge Q","eq")]:
    bs=[1 if np.argmax(r[key])==r["best"] else 0 for r in res]
    t3=[1 if r["best"] in set(np.argsort(r[key])[::-1][:3]) else 0 for r in res]
    t5=[1 if r["best"] in set(np.argsort(r[key])[::-1][:5]) else 0 for r in res]
    print(f"{name:<10s}: Best={100*np.mean(bs):5.1f}% Top3={100*np.mean(t3):5.1f}% Top5={100*np.mean(t5):5.1f}%")
print(f"{'Random':<10s}: Best= 4.2% Top3=12.5% Top5=20.8%")
