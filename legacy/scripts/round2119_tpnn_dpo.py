"""Round 2.119: TP-only NN manifold distance as DPO pair selector."""
import copy, shutil, sys, numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320, decode_boxes, evaluate_model, unfreeze_rlvr
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV, SEED = "cuda", 42; EPOCHS, DPO_W, KL_W, BETA = 8, 0.1, 0.01, 2.0
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"

def bm():
    return build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}})
def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD([{"params":body,"lr":0.0001},{"params":head,"lr":0.001}],lr=0.001,momentum=0.9,weight_decay=0.0005)
def compute_stable_iou(sp_raw, bf, box_predictor, tgts_t):
    sp_cat=torch.cat(sp_raw,dim=0); N=sp_cat.shape[0]
    with torch.no_grad():
        reg=box_predictor.bbox_pred(bf[:N]); decoded=decode_boxes(sp_cat,reg[:,2:6])
    iou,gt_idx=torch.zeros(N,device=DEV),torch.full((N,),-1,dtype=torch.long,device=DEV); off=0
    for i_img,p_img in enumerate(sp_raw):
        n_p=p_img.shape[0]
        if n_p>0 and len(tgts_t[i_img]["boxes"])>0:
            i=box_iou(decoded[off:off+n_p],tgts_t[i_img]["boxes"])
            iou[off:off+n_p],gt_idx[off:off+n_p]=i.max(dim=1)
        off+=n_p
    return iou,gt_idx

def extract_fft_for_proposals(model, loader):
    sp,bhi={},{}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bhi.update({"x":a[0]}))
    all_amp,all_iou=[],[]
    for imgs,tgts in tqdm(loader,desc="Calib",leave=False):
        imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
        sp.clear();bhi.clear()
        with torch.no_grad(): model(imgs_d,tgts_t)
        sr=sp.get("p");rf=bhi.get("x")
        if sr is None or rf is None:continue
        bf=model.roi_heads.box_head(rf)
        reg=model.roi_heads.box_predictor.bbox_pred(bf);sc=torch.cat(sr,dim=0)
        decoded=decode_boxes(sc,reg[:,2:6]);off=0
        for i_img,p_img in enumerate(sr):
            full=imgs_d[i_img];gt=tgts_t[i_img]["boxes"]
            if len(gt)==0:off+=p_img.shape[0];continue
            i=box_iou(decoded[off:off+p_img.shape[0]],gt);bi,_=i.max(dim=1)
            for idx in range(p_img.shape[0]):
                x1,y1,x2,y2=p_img[idx].long()
                x1,y1=max(0,x1),max(0,y1);x2,y2=min(full.shape[2],x2),min(full.shape[1],y2)
                if x2<=x1 or y2<=y1:continue
                crop=full[:,y1:y2,x1:x2].unsqueeze(0)
                crop=F.interpolate(crop,(64,64),mode="bilinear",align_corners=False).squeeze(0)
                amp=torch.abs(torch.fft.rfft2(crop,dim=(-2,-1))).cpu().numpy().flatten()
                all_amp.append(amp);all_iou.append(bi[idx].item())
            off+=p_img.shape[0]
    return np.stack(all_amp),np.array(all_iou)

def run_one(cfg_name, mode, seed):
    run_name=f"round2119_{cfg_name}_s{seed}"; set_seed(seed)
    model=bm().to(DEV); ckpt=torch.load(CKPT,map_location=DEV)
    model.load_state_dict(ckpt["model"]); unfreeze_rlvr(model)
    baseline_model=copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad=False

    # Pre-compute TP manifold
    calib_model=bm().to(DEV);ck=torch.load(CKPT,map_location=DEV)
    calib_model.load_state_dict(ck["model"]);calib_model.eval()
    tl,vl=build_penn_fudan_loaders_320(batch_size=2)
    amp,ious=extract_fft_for_proposals(calib_model,tl);del calib_model
    tp=ious>0.5
    scaler=StandardScaler().fit(amp[tp]);amp_t=scaler.transform(amp)
    pca=PCA(n_components=7,whiten=True,random_state=42).fit(amp_t[tp]);W=pca.transform(amp_t)
    nn=NearestNeighbors(n_neighbors=5,metric="euclidean").fit(W[tp])
    tp_dists=nn.kneighbors(W[tp],n_neighbors=5)[0].mean(axis=1)
    tp_med,tp_std=np.median(tp_dists),tp_dists.std()

    sp,bhi={},{}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bhi.update({"x":a[0]}))
    opt=build_opt(model);bp=baseline_model.roi_heads.box_predictor
    run_dir=ensure_run_dir(run_name);shutil.copy(__file__,run_dir/"runner_snapshot.py")
    is_det=mode=="det_only"; h,best_ap75=[],-1.0

    for ep in range(1,EPOCHS+1):
        model.train()
        for imgs,tgts in tqdm(tl,desc=f"{run_name} e{ep}",leave=False):
            imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
            sp.clear();bhi.clear()
            ld=model(imgs_d,tgts_t);det=sum(ld.values())
            rf=bhi.get("x");sr=sp.get("p")
            dpo,kl=torch.tensor(0.0,device=DEV),torch.tensor(0.0,device=DEV)

            if not is_det and rf is not None and sr is not None and rf.shape[0]>0:
                bf=model.roi_heads.box_head(rf);cls_logits=model.roi_heads.box_predictor.cls_score(bf)
                person_logit=cls_logits[:,1]
                with torch.no_grad():
                    bb=baseline_model.roi_heads.box_head(rf)
                    b_person=bp.cls_score(bb)[:,1]
                iou_p,gt_idx=compute_stable_iou(sr,bb,bp,tgts_t)
                N=min(cls_logits.shape[0],iou_p.shape[0])

                # FFT for manifold distance
                amp_vecs=[];off=0
                for i_img,p_img in enumerate(sr):
                    full_=imgs_d[i_img];np_=p_img.shape[0]
                    for idx in range(min(np_,N-off)):
                        x1,y1,x2,y2=p_img[idx].long()
                        x1,y1=max(0,x1),max(0,y1);x2,y2=min(full_.shape[2],x2),min(full_.shape[1],y2)
                        if x2<=x1 or y2<=y1:amp_vecs.append(np.zeros(6336))
                        else:
                            crop=full_[:,y1:y2,x1:x2].unsqueeze(0)
                            crop=F.interpolate(crop,(64,64),mode="bilinear",align_corners=False).squeeze(0)
                            amp_vecs.append(torch.abs(torch.fft.rfft2(crop,dim=(-2,-1))).cpu().numpy().flatten())
                    off+=np_
                amp_np=np.stack(amp_vecs[:N]);W_live=pca.transform(scaler.transform(amp_np))
                manifold_dist=nn.kneighbors(W_live,n_neighbors=5)[0].mean(axis=1)
                manifold_score=-(manifold_dist-tp_med)/tp_std  # closer=higher

                n_pairs=0
                for gid in torch.unique(gt_idx[:N]):
                    if gid<0:continue
                    mask=gt_idx[:N]==gid;np_array=mask.cpu().numpy()
                    if np_array.sum()<2:continue
                    scores=manifold_score[np_array]
                    logits=person_logit[mask];ref=b_person[mask]
                    best_idx=int(np.argmax(scores));worst_idx=int(np.argmin(scores))
                    if best_idx==worst_idx:continue
                    lc,lr=logits[best_idx],logits[worst_idx];rc,rr=ref[best_idx],ref[worst_idx]
                    dpo=dpo-F.logsigmoid(BETA*((lc-rc)-(lr-rr)))
                    n_pairs+=1
                if n_pairs>0:dpo=dpo/n_pairs
                kl=KL_W*(person_logit[:N]-b_person[:N]).pow(2).mean()

            loss=det+DPO_W*dpo+kl
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.0);opt.step()

        em=evaluate_model(model,vl,DEV);h.append({"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"]})
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f}")
        if em["ap75"]>best_ap75:best_ap75=em["ap75"]

    best_h=max(h,key=lambda r:r["val_ap75"])
    em.update({"run_name":run_name,"config":cfg_name,"mode":mode,"seed":seed,"epochs":len(h),"best_ap50":best_h["val_ap50"],"best_ap75":best_ap75,"history":h})
    save_json(em,run_dir/"eval_metrics.json");return em

if __name__=="__main__":
    results=[]
    for cfg,mode in [("det_only","det_only"),("tpnn_dpo","tpnn_dpo")]:
        r=run_one(cfg,mode,42);results.append(r)
    print("\n## 2.119 TP-NN manifold DPO")
    for r in results:
        bh=max(r["history"],key=lambda x:x["val_ap75"])
        print(f"  {r['config']:<10s} s{r['seed']} AP75={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
