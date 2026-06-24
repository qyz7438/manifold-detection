"""Plan 2.71: q-only reward, 3 seeds, 4 groups. Verifies FFT verifier signal transmission.

Groups: qonly_real, qonly_shuf, qonly_band, det_only
Seeds: 42, 123, 456
Metrics: AP75, best AP75, q_corr, rl_grad, Precision, ECE
Pass: real mean AP75 >= shuffled+0.01 AND ≥2/3 seeds real > shuffled
"""
import sys,json,subprocess,math,copy
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou
import numpy as np
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json,ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed
GIT=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
DEV="cuda";CKPT="runs/round227_v1_baseline_20ep/checkpoint_best.pth";G=4;EPOCHS=5;ROI_SIZE=14;SEEDS=[42,123,456]

class PerChanFFT(nn.Module):
    def __init__(self,fft_dim,hidden=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(fft_dim,hidden),nn.ReLU(),nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,x):return self.net(x).squeeze(-1)

def extract_perchan_fft(roi):
    C=roi.shape[1];H,W=roi.shape[-2],roi.shape[-1]
    fft=torch.fft.rfft2(roi,dim=(-2,-1),norm="ortho");amp=torch.abs(fft);pha=torch.angle(fft)
    freq=torch.fft.fftfreq(max(H,W),device=roi.device)
    Y,X=torch.meshgrid(freq[:H],freq[:W//2+1],indexing='ij')
    r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.15).float();md=((rn>0.15)&(rn<=0.4)).float();hi=(rn>0.4).float()
    a_lo=(amp*lo.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    a_md=(amp*md.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    a_hi=(amp*hi.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    p_lo=(pha*lo.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    p_md=(pha*md.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    p_hi=(pha*hi.unsqueeze(0).unsqueeze(0)).flatten(2).sum(2)
    return torch.cat([a_lo,a_md,a_hi,p_lo,p_md,p_hi],dim=1)

def band_permute(fft_f,C=256):
    B=fft_f.shape[0];ch_per=fft_f.shape[1]//6;out=torch.zeros_like(fft_f)
    for b in range(6):sl=slice(b*ch_per,(b+1)*ch_per);out[:,sl]=fft_f[torch.randperm(B,device=fft_f.device)][:,sl]
    return out

def decode_boxes(pr,d):
    w=pr[:,2]-pr[:,0];h=pr[:,3]-pr[:,1];cx=pr[:,0]+0.5*w;cy=pr[:,1]+0.5*h
    px=d[:,0]*w+cx-0.5*torch.exp(d[:,2])*w;py=d[:,1]*h+cy-0.5*torch.exp(d[:,3])*h
    return torch.stack([px,py,d[:,0]*w+cx+0.5*torch.exp(d[:,2])*w,d[:,1]*h+cy+0.5*torch.exp(d[:,3])*h],dim=1).clamp(min=0)

def glp(d,m,s):
    e=(d-m.unsqueeze(1))/s.unsqueeze(1)
    return -0.5*(e.pow(2)+2*torch.log(s.unsqueeze(1))+math.log(2*math.pi)).sum(dim=-1)

def bl():return build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})
def bm():return build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}})
def fe(m,parts):
    for p in m.parameters():p.requires_grad=False
    for part in parts:
        if isinstance(part,nn.Module):
            for p in part.parameters():p.requires_grad=True
@torch.no_grad()
def ev(model,vl):
    model.eval();ps,ts=[],[]
    for img,tgt in vl:
        out=model([i.to(DEV) for i in img]);ps.extend([{k:v.cpu() for k,v in o.items()} for o in out]);ts.extend([{k:v.cpu() for k,v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps,ts,iou_threshold=0.5,score_threshold=0.05)

def run_one(cfg_name,mode,seed):
    """mode: 'real','shuf','band','det_only'"""
    run_name=f"round271_{cfg_name}_s{seed}";set_seed(seed)
    model=bm().to(DEV);ckpt=torch.load(CKPT,map_location=DEV);model.load_state_dict(ckpt["model"])
    fe(model,[model.roi_heads.box_head,model.roi_heads.box_predictor]);vrf=None
    tl,vl=bl();rd=ensure_run_dir(run_name);h=[];best_ap75=-1.0
    pc={};rc={};fc={}
    model.rpn.register_forward_hook(lambda m,i,o:pc.update({"p":o[0]}))
    model.roi_heads.box_head.register_forward_pre_hook(lambda m,i:rc.update({"x":i[0]}))
    model.backbone.register_forward_hook(lambda m,i,o:fc.update({"f":{k:o[k] for k in o if k!="pool"}}))
    diag={"q_ious":[],"rl_grad_norm":[]}
    for ep in range(1,EPOCHS+1):
        model.train()
        if vrf is not None:
            vrf.train()
        td,trl,tv,pos=0.0,0.0,0.0,0
        for imgs,tgts in tqdm(tl,desc=f"{run_name} e{ep}"):
            imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
            pc.clear();rc.clear();fc.clear()
            ld=model(imgs_d,tgts_t)
            if isinstance(ld,dict):det=sum(ld.values())
            else:det=sum(sum(d.values()) for d in ld if isinstance(d,dict))
            rf=rc.get("x");pr=pc.get("p");fpn=fc.get("f")
            rl=vloss=torch.tensor(0.0,device=DEV)
            if mode!="det_only" and rf is not None and pr is not None and rf.shape[0]>0 and fpn is not None:
                N=rf.shape[0];bf=model.roi_heads.box_head(rf)
                mu=model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
                s=torch.full_like(mu,0.1,requires_grad=False)
                deltas=mu.detach().unsqueeze(1)+s.unsqueeze(1)*torch.randn(N,G,4,device=DEV)
                log_probs=glp(deltas,mu,s)
                pc_=torch.cat(pr,dim=0);N=min(N,pc_.shape[0])
                mu=mu[:N];deltas=deltas[:N];log_probs=log_probs[:N]
                ad=deltas.reshape(N*G,4);pe=pc_[:N].unsqueeze(1).expand(-1,G,-1).reshape(N*G,4)
                boxes=decode_boxes(pe,ad)
                npi=[p.shape[0] for p in pr];img_map=[];[img_map.extend([i_]*p.shape[0]) for i_,p in enumerate(pr)]
                img_map=img_map[:N];iou_r=torch.zeros(N,G,device=DEV)
                for pi in range(N):
                    gt=tgts_t[img_map[pi]]["boxes"]
                    if len(gt)>0:iou_r[pi]=box_iou(boxes[pi*G:(pi+1)*G],gt).max(dim=1).values
                roi_boxes=torch.zeros(N*G,5,device=DEV)
                for bi in range(N*G):roi_boxes[bi,0]=img_map[bi//G];roi_boxes[bi,1:]=boxes[bi]
                fpn_keys=sorted(fpn.keys(),key=int)
                bw=boxes[:,2]-boxes[:,0];bh=boxes[:,3]-boxes[:,1]
                area=(bw*bh).clamp_min(1);lvl=torch.floor(torch.log2(torch.sqrt(area)/224)+4).long().clamp(2,5)
                q_pred=torch.zeros(N*G,device=DEV)
                for ki,k in enumerate(fpn_keys):
                    ki_lvl=int(k)+2;mask=lvl==ki_lvl
                    if mask.sum()==0:continue
                    sc=1.0/(2**(ki_lvl+2))
                    r=torchvision.ops.roi_align(fpn[k],roi_boxes[mask],output_size=ROI_SIZE,spatial_scale=sc)
                    fft_f=extract_perchan_fft(r)
                    if mode=="shuf":fft_f=fft_f[torch.randperm(fft_f.shape[0],device=DEV)]
                    elif mode=="band":fft_f=band_permute(fft_f)
                    if vrf is None:
                        vrf=PerChanFFT(fft_f.shape[1]).to(DEV)
                        opt=torch.optim.SGD([p for p in list(model.parameters())+list(vrf.parameters()) if p.requires_grad],lr=0.001,momentum=0.9,weight_decay=0.0005)
                    q_pred[mask]=vrf(fft_f)
                q_pred=q_pred.view(N,G)
                diag["q_ious"].extend(list(zip(q_pred.flatten().tolist(),iou_r.flatten().tolist())))
                vloss=F.mse_loss(q_pred,iou_r.detach())
                q_norm=(q_pred-q_pred.mean(dim=1,keepdim=True))/(q_pred.std(dim=1,keepdim=True).clamp_min(1e-6))
                pm=iou_r.max(dim=1).values>0.3
                if pm.any():rl=-(q_norm[pm].detach()*log_probs[pm]).mean();pos+=pm.sum().item()
            if mode=="det_only":
                loss=det;opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.001,momentum=0.9,weight_decay=0.0005)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            elif vrf is not None:
                loss=det+vloss+0.05*rl;opt.zero_grad(set_to_none=True);loss.backward()
                gnorm=0.0
                for p in [model.roi_heads.box_predictor.bbox_pred.weight]:
                    if p.grad is not None:gnorm+=p.grad.norm().item()
                diag["rl_grad_norm"].append(gnorm)
                opt.step()
            td+=det.item();trl+=rl.item();tv+=vloss.item()
        em=ev(model,vl)
        if len(diag["q_ious"])>0:
            qs=np.array([x[0] for x in diag["q_ious"]]);iis=np.array([x[1] for x in diag["q_ious"]])
            q_corr=np.corrcoef(qs,iis)[0,1]
        else:q_corr=0.0
        rlg=np.mean(diag["rl_grad_norm"]) if len(diag["rl_grad_norm"])>0 else 0.0
        h.append({"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"],"q_iou_corr":float(q_corr),"rl_grad":float(rlg)})
        print(f"  e{ep}: AP75={em['ap75']:.4f} q_corr={q_corr:.4f} rl_grad={rlg:.4f}")
        if em["ap75"]>best_ap75:best_ap75=em["ap75"]
        diag["q_ious"].clear();diag["rl_grad_norm"].clear()
    best_h=max(h,key=lambda r:r["val_ap75"])
    em.update({"run_name":run_name,"config":cfg_name,"mode":mode,"seed":seed,"epochs":EPOCHS,"best_ap50":best_h["val_ap50"],"best_ap75":best_ap75,"history":h,"git_hash":GIT,"q_iou_corr_final":h[-1]["q_iou_corr"]})
    save_json(em,rd/"eval_metrics.json")
    return em

if __name__=="__main__":
    all_results=[]
    groups=["qonly_real","qonly_shuf","qonly_band","det_only"]
    modes=["real","shuf","band","det_only"]
    for cfg,mode in zip(groups,modes):
        for s in SEEDS:
            r=run_one(cfg,mode,s)
            all_results.append(r)
    print("\n## Plan 2.71 3-Seed Comparison")
    print(f"  {'Config':<15s} {'Seed':>5s} {'AP75':>8s} {'BestAP75':>8s} {'q_corr':>8s} {'rl_grad':>8s}")
    for r in all_results:
        print(f"  {r['config']:<15s} {r['seed']:5d} {r['ap75']:8.4f} {r['best_ap75']:8.4f} {r.get('q_iou_corr_final',0):8.4f} {r['history'][-1].get('rl_grad',0):8.4f}")
    # Judgment
    import numpy as np
    for mode_name in ["qonly_real","qonly_shuf","qonly_band"]:
        vals=[r["best_ap75"] for r in all_results if r["config"]==mode_name]
        print(f"  {mode_name}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
    real_vals=[r["best_ap75"] for r in all_results if r["config"]=="qonly_real"]
    shuf_vals=[r["best_ap75"] for r in all_results if r["config"]=="qonly_shuf"]
    band_vals=[r["best_ap75"] for r in all_results if r["config"]=="qonly_band"]
    real_wins_shuf=sum(1 for i in range(3) if real_vals[i]>shuf_vals[i])
    real_wins_band=sum(1 for i in range(3) if real_vals[i]>band_vals[i])
    delta_shuf=np.mean(real_vals)-np.mean(shuf_vals)
    delta_band=np.mean(real_vals)-np.mean(band_vals)
    print(f"\n  real vs shuf: +{delta_shuf:.4f} AP75, wins={real_wins_shuf}/3 → {'PASS' if delta_shuf>0.01 and real_wins_shuf>=2 else 'FAIL'}")
    print(f"  real vs band: +{delta_band:.4f} AP75, wins={real_wins_band}/3 → {'PASS' if delta_band>0.01 and real_wins_band>=2 else 'FAIL'}")
