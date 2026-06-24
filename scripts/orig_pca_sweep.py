import numpy as np, sys, torch, torch.nn.functional as F
from torchvision.ops import box_iou
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320, decode_boxes
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
set_seed(42); DEV="cuda"
model=bm().to(DEV); ckpt=torch.load("runs/round227_v1_baseline_20ep/checkpoint_best.pth",map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()
sp,bhi={},{}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bhi.update({"x":a[0]}))
_,vl=build_penn_fudan_loaders_320(batch_size=1)
X,iou_list,gt_list=[],[],[]
for img,tgt in vl:
    img_d=[img[0].to(DEV)]; tgt_d=[{k:v.to(DEV) for k,v in tgt[0].items()}]
    sp.clear();bhi.clear()
    with torch.no_grad():model(img_d,tgt_d)
    sr=sp.get("p");rf=bhi.get("x")
    if sr is None or rf is None:continue
    bf=model.roi_heads.box_head(rf)
    reg=model.roi_heads.box_predictor.bbox_pred(bf);sc=torch.cat(sr,dim=0)
    decoded=decode_boxes(sc,reg[:,2:6]);gt=tgt_d[0]["boxes"];full=img_d[0]
    if len(gt)>0:i=box_iou(decoded,gt);bi,bg=i.max(dim=1)
    else:bi=torch.zeros(sc.shape[0]);bg=torch.full((sc.shape[0],),-1,dtype=torch.long)
    for idx in range(sc.shape[0]):
        x1,y1,x2,y2=sc[idx].long()
        x1,y1=max(0,x1),max(0,y1);x2,y2=min(full.shape[2],x2),min(full.shape[1],y2)
        if x2<=x1 or y2<=y1:continue
        crop=full[:,y1:y2,x1:x2].unsqueeze(0)
        crop=F.interpolate(crop,(64,64),mode="bilinear",align_corners=False).squeeze(0)
        amp=torch.abs(torch.fft.rfft2(crop,dim=(-2,-1))).cpu().numpy().flatten()
        X.append(amp);iou_list.append(bi[idx].item());gt_list.append(bg[idx].item())

X=np.stack(X);ious=np.array(iou_list);gt_ids=np.array(gt_list)
print(f"Samples: {X.shape[0]}, dim: {X.shape[1]}")
tp=ious>0.5

for k in [6,50,200,500]:
    s=StandardScaler().fit(X[tp])
    p=PCA(n_components=k,whiten=True,random_state=42).fit(s.transform(X[tp]))
    w=p.transform(s.transform(X))
    ctr=np.median(w[tp],axis=0)
    d=-np.linalg.norm(w-ctr,axis=1)
    agree,tot=0,0
    for g in np.unique(gt_ids):
        if g<0:
            continue
        m=gt_ids==g
        n=m.sum()
        if n<2:
            continue
        idxs=np.where(m)[0]
        for ii in range(n):
            for jj in range(ii+1,n):
                iou_order=ious[idxs[ii]]>ious[idxs[jj]]
                met_order=d[idxs[ii]]>d[idxs[jj]]
                if iou_order==met_order:
                    agree+=1
                tot+=1
    pc=100*agree/max(tot,1)
    vr=p.explained_variance_ratio_.sum()
    print(f"PCA({k}): var={100*vr:.1f}% pair={pc:.1f}%")
print("Ref: Edge=70.2%, Full-PCA50=34.8%")
