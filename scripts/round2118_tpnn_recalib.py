"""Round 2.118: TP-only NN manifold RLVR — per-epoch recalibration.
Fixes calibration drift by rebuilding TP manifold reference each epoch.
"""
import copy, shutil, subprocess, sys, numpy as np, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import build_penn_fudan_loaders_320, decode_boxes, evaluate_model, gaussian_log_prob, unfreeze_rlvr
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import set_seed

DEV, SEED = "cuda", 42; EPOCHS, G, RL_W, KL_W, BONUS_W = 8, 4, 0.0005, 0.01, 0.05
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"

def bm():
    return build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}})
def build_opt(model):
    body, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if "box_head" in n or "box_predictor" in n else body).append(p)
    return torch.optim.SGD([{"params":body,"lr":0.0001},{"params":head,"lr":0.001}],lr=0.001,momentum=0.9,weight_decay=0.0005)
def cross_proposal_grpo(reward, n_props):
    adv = torch.zeros_like(reward); off = 0
    for n_p in n_props:
        if n_p<=0: continue
        if n_p==1: adv[off]=0.0; off+=n_p; continue
        r=reward[off:off+n_p]; m=r.mean(); s=r.std().clamp_min(1e-6)
        adv[off:off+n_p]=(r-m)/s; off+=n_p
    return adv
def compute_stable_iou(sp_raw, bf, box_predictor, tgts_t):
    sp_cat=torch.cat(sp_raw,dim=0); N=sp_cat.shape[0]
    with torch.no_grad():
        reg=box_predictor.bbox_pred(bf[:N]); decoded=decode_boxes(sp_cat,reg[:,2:6])
    iou=torch.zeros(N,device=DEV); off=0
    for i_img,p_img in enumerate(sp_raw):
        n_p=p_img.shape[0]
        if n_p>0 and len(tgts_t[i_img]["boxes"])>0:
            i=box_iou(decoded[off:off+n_p],tgts_t[i_img]["boxes"])
            iou[off:off+n_p]=i.max(dim=1).values
        off+=n_p
    return iou

def extract_fft_for_proposals(model, loader):
    """Extract original-image FFT features + IoU from frozen model."""
    sp,bhi={},{}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bhi.update({"x":a[0]}))
    all_amp,all_iou=[],[]
    for imgs,tgts in tqdm(loader,desc="Recalib",leave=False):
        imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
        sp.clear();bhi.clear()
        with torch.no_grad(): model(imgs_d,tgts_t)
        sr=sp.get("p");rf=bhi.get("x")
        if sr is None or rf is None:continue
        bf=model.roi_heads.box_head(rf)
        reg=model.roi_heads.box_predictor.bbox_pred(bf);sc=torch.cat(sr,dim=0)
        decoded=decode_boxes(sc,reg[:,2:6])
        off=0
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

def rebuid_manifold(amp, ious):
    tp=ious>0.5;print(f"  TP={tp.sum()}/{len(ious)}")
    s=StandardScaler().fit(amp[tp]);Xt=s.transform(amp)
    p=PCA(n_components=7,whiten=True,random_state=42).fit(Xt[tp]);W=p.transform(Xt)
    nn=NearestNeighbors(n_neighbors=5,metric="euclidean").fit(W[tp])
    tp_dists=nn.kneighbors(W[tp],n_neighbors=5)[0].mean(axis=1)
    return s,p,nn,W[tp],np.median(tp_dists),tp_dists.std()

def nn_bonus(amp, s, p, nn, tp_med, tp_std):
    W=p.transform(s.transform(amp))
    dists=nn.kneighbors(W,n_neighbors=5)[0].mean(axis=1)
    return -((dists-tp_med)/(tp_std+1e-8))

def run_one(cfg_name, mode, seed):
    run_name=f"round2118_{cfg_name}_s{seed}"; set_seed(seed)
    model=bm().to(DEV); ckpt=torch.load(CKPT,map_location=DEV)
    model.load_state_dict(ckpt["model"]); unfreeze_rlvr(model)
    baseline_model=copy.deepcopy(model); baseline_model.eval()
    for p in baseline_model.parameters(): p.requires_grad=False

    sp,bhi={},{}
    model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a:sp.update({"p":[x.clone() for x in a[1]]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m,a:bhi.update({"x":a[0]}))
    tl,vl=build_penn_fudan_loaders_320(batch_size=2)
    opt=build_opt(model);bp=baseline_model.roi_heads.box_predictor
    run_dir=ensure_run_dir(run_name);shutil.copy(__file__,run_dir/"runner_snapshot.py")
    is_det=mode=="det_only"; h,best_ap75=[],-1.0

    # Initial calibration
    if not is_det:
        calib_model=bm().to(DEV);ck=torch.load(CKPT,map_location=DEV)
        calib_model.load_state_dict(ck["model"]);calib_model.eval()
        amp_init,ious_init=extract_fft_for_proposals(calib_model,tl)
        s,p,nn,W_tp,tp_med,tp_std=rebuid_manifold(amp_init,ious_init)
        del calib_model
        print(f"Initial manifold built. TP median={tp_med:.4f}")

    for ep in range(1,EPOCHS+1):
        # Recalibrate at epoch start
        if not is_det and ep>1:
            amp_ep,ious_ep=extract_fft_for_proposals(baseline_model,tl)
            s,p,nn,W_tp,tp_med,tp_std=rebuid_manifold(amp_ep,ious_ep)

        model.train(); td,trl,tkl=0.0,0.0,0.0
        for imgs,tgts in tqdm(tl,desc=f"{run_name} e{ep}",leave=False):
            imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
            sp.clear();bhi.clear()
            ld=model(imgs_d,tgts_t);det=sum(ld.values())
            rf=bhi.get("x");sr=sp.get("p")
            rl,kl=torch.tensor(0.0,device=DEV),torch.tensor(0.0,device=DEV)

            if not is_det and rf is not None and sr is not None and rf.shape[0]>0:
                bf=model.roi_heads.box_head(rf);cls_logits=model.roi_heads.box_predictor.cls_score(bf)
                with torch.no_grad():
                    bb=baseline_model.roi_heads.box_head(rf)
                    bl_conf=F.softmax(bp.cls_score(bb),dim=-1)[:,1]
                sp_sigma=0.05+0.2*(1.0-bl_conf)
                s_base=sp_sigma.unsqueeze(1).expand(-1,cls_logits.shape[1])
                with torch.no_grad():
                    bl_logits=bp.cls_score(bb)
                    perturbed=bl_logits.unsqueeze(1)+s_base.unsqueeze(1)*torch.randn(bl_logits.shape[0],G,cls_logits.shape[1],device=DEV)
                pert_conf=F.softmax(perturbed,dim=-1)[:,:,1]
                s_cls=sp_sigma.unsqueeze(1).expand(-1,cls_logits.shape[1])
                log_probs=gaussian_log_prob(perturbed,cls_logits,s_cls)
                iou_p=compute_stable_iou(sr,bb,bp,tgts_t);N=min(cls_logits.shape[0],iou_p.shape[0])
                reward=pert_conf[:N]*(2*iou_p[:N]-1).unsqueeze(1)

                # FFT feature extraction for live bonus
                live_conf=F.softmax(cls_logits,dim=-1)[:N,1]
                uncertain=((live_conf>=0.1)&(live_conf<=0.5)).float()
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
                amp_np=np.stack(amp_vecs[:N]);bonus_np=nn_bonus(amp_np,s,p,nn,tp_med,tp_std)
                bonus=torch.tensor(bonus_np,device=DEV).float()
                reward=reward+BONUS_W*bonus.unsqueeze(1)*uncertain.unsqueeze(1)

                reward_flat=reward.reshape(-1);npp=[p.shape[0]*G for p in sr]
                adv=cross_proposal_grpo(reward_flat,npp).view(N,G)
                rl=-(adv.detach()*log_probs[:N]).mean()
                kl=KL_W*(pert_conf[:N]-bl_conf[:N].unsqueeze(1)).pow(2).mean()

            loss=det+RL_W*rl+kl
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.0);opt.step()
            td+=det.item();trl+=rl.item();tkl+=kl.item()

        em=evaluate_model(model,vl,DEV)
        row={"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"],"ece":em.get("ece",0)}
        h.append(row);print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f}")
        if em["ap75"]>best_ap75:best_ap75=em["ap75"]

    best_h=max(h,key=lambda r:r["val_ap75"])
    em.update({"run_name":run_name,"config":cfg_name,"mode":mode,"seed":seed,"epochs":len(h),"best_ap50":best_h["val_ap50"],"best_ap75":best_ap75,"history":h})
    save_json(em,run_dir/"eval_metrics.json");return em

if __name__=="__main__":
    results=[]
    for cfg,mode in [("det_only","det_only"),("tpnn","tpnn")]:
        r=run_one(cfg,mode,42);results.append(r)
    print("\n## 2.118 TP-NN RLVR per-epoch recalib")
    for r in results:
        bh=max(r["history"],key=lambda x:x["val_ap75"])
        print(f"  {r['config']:<10s} s{r['seed']} AP75={r['best_ap75']:.4f} AP50={bh['val_ap50']:.4f}")
