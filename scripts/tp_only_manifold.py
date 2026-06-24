"""TP-only manifold for uncertain proposal evaluation — no leakage."""
import numpy as np, sys, torch, torch.nn.functional as F
from torchvision.ops import box_iou
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320,decode_boxes
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

set_seed(42); DEV="cuda"
model=bm().to(DEV); ckpt=torch.load("runs/round227_v1_baseline_20ep/checkpoint_best.pth",map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()
sp,bhi={},{}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bhi.update({"x":a[0]}))
_,vl=build_penn_fudan_loaders_320(batch_size=1)
X,iou_list,conf_list,gt_list=[],[],[],[]
for img,tgt in vl:
    img_d=[img[0].to(DEV)]; tgt_d=[{k:v.to(DEV) for k,v in tgt[0].items()}]
    sp.clear();bhi.clear()
    with torch.no_grad():model(img_d,tgt_d)
    sr=sp.get("p");rf=bhi.get("x")
    if sr is None or rf is None:continue
    bf=model.roi_heads.box_head(rf)
    conf=F.softmax(model.roi_heads.box_predictor.cls_score(bf),dim=-1)[:,1]
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
        X.append(amp);iou_list.append(bi[idx].item());conf_list.append(conf[idx].item());gt_list.append(bg[idx].item())

X=np.stack(X);ious=np.array(iou_list);confs=np.array(conf_list);gt_ids=np.array(gt_list)
print(f"Samples: {X.shape[0]}, dim: {X.shape[1]}")

# Split: TP for manifold construction, uncertain for evaluation
tp_mask=ious>0.5;fp_mask=ious<0.3
umask=(confs>=0.1)&(confs<=0.5)
u_tp=umask&(ious>0.5);u_fp=umask&(ious<0.3)
print(f"TP: {tp_mask.sum()}, Uncertain TP: {u_tp.sum()}, Uncertain FP: {u_fp.sum()}")

# Fit ONLY on TP
scaler=StandardScaler().fit(X[tp_mask])
Xt=scaler.transform(X)
pca=PCA(n_components=6,whiten=True,random_state=42).fit(Xt[tp_mask])
W=pca.transform(Xt)  # (N, 6)

# Build k-NN graph ONLY on TP points
tp_idx=np.where(tp_mask)[0]
nn=NearestNeighbors(n_neighbors=15,metric="euclidean").fit(W[tp_idx])

# For each uncertain proposal: distance to nearest TP neighbor
u_idx=np.where(umask)[0]
nn_dists,_=nn.kneighbors(W[u_idx],n_neighbors=5)
# Mean distance to 5 nearest TP neighbors (in whitened PCA space)
avg_nn_dist=nn_dists.mean(axis=1)  # (n_uncertain,)

# Also: residual after projecting onto TP PCA subspace
# PCA was fit on TP, so for FP proposals, the projection should have higher reconstruction error
W_u=W[u_idx]
X_recon=pca.inverse_transform(W_u)
X_orig=X[u_idx]
recon_err=np.linalg.norm(scaler.inverse_transform(pca.inverse_transform(W_u))-X_orig,axis=1)/(np.linalg.norm(X_orig,axis=1)+1e-8)

# Combine: NN distance + reconstruction error
for name,signal,higher_better in [
    ("NN dist(5 TP)",-avg_nn_dist,True),  # closer=better
    ("Recon error",-recon_err,True),        # lower error=better
    ("Fusion(rank avg)",None,False),
]:
    if name=="Fusion(rank avg)":
        r1=np.argsort(np.argsort(-avg_nn_dist))
        r2=np.argsort(np.argsort(-recon_err))
        signal=r1+r2
        higher_better=True
    if higher_better:
        thr=np.median(signal)
        tp_recall=(signal>thr)[u_tp[u_idx]].sum()/max(u_tp.sum(),1)
        fp_recall=(signal>thr)[u_fp[u_idx]].sum()/max(u_fp.sum(),1)
        print(f"{name:20s}: TP recall={tp_recall:.3f}, FP recall={fp_recall:.3f}")

print(f"\nBaselines: Old(all-data) Isomap=46.7%, Edge=47.3%, Curvature=50.9%")
