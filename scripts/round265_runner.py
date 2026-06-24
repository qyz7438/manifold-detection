"""Plan 2.65: FFT-ONLY verifier GRPO — no spatial features, pure frequency domain. CONTROL."""
import sys,json,subprocess,math,copy
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json,ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed
GIT=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
DEV="cuda";CKPT="runs/round227_v1_baseline_20ep/checkpoint_best.pth";G=4;EPOCHS=5;ROI_SIZE=14

class FFTOnlyVerifier(nn.Module):
    """FFT features only — no spatial input."""
    def __init__(self,fft_dim,hidden=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(fft_dim,hidden),nn.ReLU(),nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,fft_f):return self.net(fft_f).squeeze(-1)

def extract_fft_feats(roi):
    fft=torch.fft.rfft2(roi,dim=(-2,-1),norm="ortho")
    amp=torch.log1p(torch.abs(fft)).mean(dim=1);pha=torch.angle(fft+0.01).mean(dim=1)
    return torch.cat([amp.flatten(1),pha.flatten(1)],dim=1)

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

def main():
    all_r=[]
    for lbd in [0.1,0.5]:
        run_name=f"round265_fftONLY_l{lbd}_s42";set_seed(42)
        model=bm().to(DEV);ckpt=torch.load(CKPT,map_location=DEV);model.load_state_dict(ckpt["model"])
        ref=copy.deepcopy(model);fe(ref,[]);ref.eval();vrf=None
        fe(model,[model.roi_heads.box_head,model.roi_heads.box_predictor])
        tl,vl=bl();rd=ensure_run_dir(run_name);h=[];best=-1.0
        pc={};rc={};fc={}
        model.rpn.register_forward_hook(lambda m,i,o:pc.update({"p":o[0]}))
        model.roi_heads.box_head.register_forward_pre_hook(lambda m,i:rc.update({"x":i[0]}))
        model.backbone.register_forward_hook(lambda m,i,o:fc.update({"f":{k:o[k] for k in o if k!="pool"}}))
        for ep in range(1,EPOCHS+1):
            model.train()
            if vrf is not None:vrf.train()
            td,trl,tv,pos=0.0,0.0,0.0,0;baseline=[None]
            for imgs,tgts in tqdm(tl,desc=f"{run_name} e{ep}"):
                imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
                pc.clear();rc.clear();fc.clear()
                ld=model(imgs_d,tgts_t)
                if isinstance(ld,dict):det=sum(ld.values())
                else:det=sum(sum(d.values()) for d in ld if isinstance(d,dict))
                rf=rc.get("x");pr=pc.get("p");fpn=fc.get("f")
                rl=torch.tensor(0.0,device=DEV);vloss=torch.tensor(0.0,device=DEV)
                if rf is not None and pr is not None and rf.shape[0]>0 and fpn is not None:
                    N=rf.shape[0];bf=model.roi_heads.box_head(rf)
                    mu=model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
                    s=torch.full_like(mu,0.1,requires_grad=False)
                    eps=torch.randn(N,G,4,device=DEV)
                    deltas=mu.detach().unsqueeze(1)+s.unsqueeze(1)*eps
                    log_probs=glp(deltas,mu,s)
                    pc_=torch.cat(pr,dim=0);N=min(N,pc_.shape[0])
                    mu=mu[:N];deltas=deltas[:N];log_probs=log_probs[:N]
                    ad=deltas.reshape(N*G,4);pe=pc_[:N].unsqueeze(1).expand(-1,G,-1).reshape(N*G,4)
                    boxes=decode_boxes(pe,ad)
                    npi=[p.shape[0] for p in pr]
                    img_map=[];[img_map.extend([i_]*p.shape[0]) for i_,p in enumerate(pr)]
                    img_map=img_map[:N];iou_r=torch.zeros(N,G,device=DEV)
                    for pi in range(N):
                        gt=tgts_t[img_map[pi]]["boxes"]
                        if len(gt)>0:iou_r[pi]=box_iou(boxes[pi*G:(pi+1)*G],gt).max(dim=1).values
                    roi_boxes=torch.zeros(N*G,5,device=DEV)
                    for bi in range(N*G):roi_boxes[bi,0]=img_map[bi//G];roi_boxes[bi,1:]=boxes[bi]
                    fpn_keys=sorted(fpn.keys(),key=int)
                    bw=boxes[:,2]-boxes[:,0];bh=boxes[:,3]-boxes[:,1]
                    area=(bw*bh).clamp_min(1)
                    lvl=torch.floor(torch.log2(torch.sqrt(area)/224)+4).long().clamp(2,5)
                    q_pred=torch.zeros(N*G,device=DEV)
                    for ki,k in enumerate(fpn_keys):
                        ki_lvl=int(k)+2;mask=lvl==ki_lvl
                        if mask.sum()==0:continue
                        sc=1.0/(2**(ki_lvl+2))
                        r=torchvision.ops.roi_align(fpn[k],roi_boxes[mask],output_size=ROI_SIZE,spatial_scale=sc)
                        fft_f=extract_fft_feats(r)
                        if vrf is None:
                            vrf=FFTOnlyVerifier(fft_f.shape[1]).to(DEV)
                            params=[p for p in list(model.parameters())+list(vrf.parameters()) if p.requires_grad]
                            opt=torch.optim.SGD(params,lr=0.001,momentum=0.9,weight_decay=0.0005)
                        q_pred[mask]=vrf(fft_f)
                    q_pred=q_pred.view(N,G)
                    vloss=F.mse_loss(q_pred,iou_r.detach())
                    reward=iou_r+lbd*q_pred.detach()
                    if baseline[0] is None:baseline[0]=reward.float().mean().item()
                    else:baseline[0]=0.9*baseline[0]+0.1*reward.float().mean().item()
                    adv=reward-baseline[0]
                    pm=iou_r.max(dim=1).values>0.3
                    if pm.any():
                        rl=-(adv[pm].detach()*log_probs[pm]).mean()
                        pos+=pm.sum().item()
                if vrf is None:loss=det
                else:
                    loss=det+vloss+0.01*rl;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                td+=det.item();trl+=rl.item();tv+=vloss.item()
            em=ev(model,vl)
            row={"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"]};h.append(row)
            print(f"  e{ep}: AP50={em['ap50']:.4f} det={td:.1f} rl={trl:.3f} v={tv:.3f} pos={pos}")
            if em["ap50"]>best:best=em["ap50"]
        em.update({"run_name":run_name,"lambda":lbd,"epochs":EPOCHS,"seed":42,"best_ap50":best,"history":h,"git_hash":GIT})
        save_json(em,rd/"eval_metrics.json");all_r.append(em)
        print(f"  DONE l{lbd}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")
    print("\n## Plan 2.65 v5 FFT-Only Verifier Results")
    for r in all_r:print(f"  l{r['lambda']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")
if __name__=="__main__":main()
