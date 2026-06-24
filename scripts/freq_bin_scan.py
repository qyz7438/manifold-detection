import sys, torch, numpy as np, torch.nn.functional as F
from torchvision.ops import box_iou
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320, decode_boxes
from scripts.round2102_runner import bm
from spectral_detection_posttrain.utils.seed import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
set_seed(42); DEV="cuda"
CKPT="runs/round227_v1_baseline_20ep/checkpoint_best.pth"
model=bm().to(DEV); ckpt=torch.load(CKPT,map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()

sampled_props,box_head_in={},{}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sampled_props.update({"p":[x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:box_head_in.update({"x":a[0]}))
_,vl=build_penn_fudan_loaders_320(batch_size=1)

all_amp=[]; all_iou=[]; all_conf=[]; all_gt=[]
for img,tgt in vl:
    img_d=[img[0].to(DEV)]; tgt_d=[{k:v.to(DEV) for k,v in tgt[0].items()}]
    sampled_props.clear(); box_head_in.clear()
    with torch.no_grad(): model(img_d,tgt_d)
    sp_raw=sampled_props.get("p"); rf=box_head_in.get("x")
    if sp_raw is None or rf is None: continue
    bf=model.roi_heads.box_head(rf)
    conf=F.softmax(model.roi_heads.box_predictor.cls_score(bf),dim=-1)[:,1]
    reg=model.roi_heads.box_predictor.bbox_pred(bf)
    sp_cat=torch.cat(sp_raw,dim=0); decoded=decode_boxes(sp_cat,reg[:,2:6])
    gt=tgt_d[0]["boxes"]; full=img_d[0]
    if len(gt)>0: iou_mat=box_iou(decoded,gt); best_iou,best_gt=iou_mat.max(dim=1)
    else: best_iou=torch.zeros(sp_cat.shape[0]); best_gt=torch.full((sp_cat.shape[0],),-1,dtype=torch.long)
    for idx in range(sp_cat.shape[0]):
        x1,y1,x2,y2=sp_cat[idx].long()
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(full.shape[2],x2),min(full.shape[1],y2)
        if x2<=x1 or y2<=y1: continue
        crop=full[:,y1:y2,x1:x2].unsqueeze(0)
        crop=F.interpolate(crop,(64,64),mode="bilinear",align_corners=False).squeeze(0)
        amp=torch.abs(torch.fft.rfft2(crop,dim=(-2,-1))).cpu().numpy().flatten()
        all_amp.append(amp); all_iou.append(best_iou[idx].item())
        all_conf.append(conf[idx].item()); all_gt.append(best_gt[idx].item())

X=np.stack(all_amp,axis=0); ious=np.array(all_iou); confs=np.array(all_conf); gt_ids=np.array(all_gt)
print(f"Samples: {X.shape[0]}, dim: {X.shape[1]}")

tp_mask=ious>0.5; fp_mask=ious<0.3
print(f"TP: {tp_mask.sum()}, FP: {fp_mask.sum()}")

# Per-bin Cohen's d
ds=np.zeros(X.shape[1])
for j in range(X.shape[1]):
    x_tp=X[tp_mask,j]; x_fp=X[fp_mask,j]
    gap=x_tp.mean()-x_fp.mean()
    ds[j]=gap/(np.sqrt((x_tp.var()+x_fp.var())/2+1e-8))

# Frequency location per bin
freq_y=np.fft.fftfreq(64); freq_x=np.fft.rfftfreq(64)
fy,fx=np.meshgrid(freq_y,freq_x,indexing='ij')
radius=np.sqrt(fy**2+fx**2).flatten(); radii=np.tile(radius,3)

def pair_consistency(metric, ious, gt_ids):
    agree,total=0,0
    for gid in np.unique(gt_ids):
        if gid<0: continue
        mask=gt_ids==gid; n=mask.sum()
        if n<2: continue
        idxs=np.where(mask)[0]
        for i in range(n):
            for j in range(i+1,n):
                iou_order=ious[idxs[i]]>ious[idxs[j]]
                met_order=metric[idxs[i]]>metric[idxs[j]]
                total+=1
                if iou_order==met_order: agree+=1
    return 100*agree/max(total,1)

for thr in [0.10,0.15,0.20]:
    keep=np.abs(ds)>thr; nk=keep.sum()
    lf=keep[radii<0.15].sum()/max(radii[radii<0.15].size,1)
    mf=keep[(radii>=0.15)&(radii<=0.4)].sum()/max(radii[(radii>=0.15)&(radii<=0.4)].size,1)
    hf=keep[radii>0.4].sum()/max(radii[radii>0.4].size,1)
    Xf=X[:,keep]
    scaler=StandardScaler().fit(Xf[tp_mask])
    pca=PCA(n_components=min(50,nk),whiten=True,random_state=42).fit(scaler.transform(Xf[tp_mask]))
    whitened=pca.transform(scaler.transform(Xf))
    tp_center=np.median(whitened[tp_mask],axis=0)
    dists=-np.linalg.norm(whitened-tp_center,axis=1)  # closer=better
    pc=pair_consistency(dists,ious,gt_ids)

    umask=(confs>=0.1)&(confs<=0.5)
    edge_med=np.median(dists[umask])
    u_tp=umask&(ious>0.5)
    recall=(dists[u_tp]>edge_med).sum()/max(u_tp.sum(),1)
    print(f"|d|>{thr}: {nk}/{X.shape[1]} ({100*nk/X.shape[1]:.1f}%) lo={100*lf:.0f}% mid={100*mf:.0f}% hi={100*hf:.0f}% pair={pc:.1f}% recall={recall:.3f}")

print(f"\nBaseline: full-spec pair={57.9}% Edge pair=70.2% Edge recall=0.473")
