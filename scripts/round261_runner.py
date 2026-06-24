"""Plan 2.61: IoU-only GRPO with action.detach().

CRITICAL FIX: deltas = mu.detach() + sigma * eps — action detached from mu.
This prevents the log_prob gradient cancellation that killed all previous DPO/RL experiments.

Pure IoU reward (oracle) — verifies the pipeline works before adding spectral verifier.
"""
import sys, json, subprocess, math, copy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import box_iou

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"; CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
G = 4; ALPHAS = [0.01, 0.1]; EPOCHS = 5

def decode_boxes(proposals, deltas):
    w = proposals[:,2]-proposals[:,0]; h = proposals[:,3]-proposals[:,1]
    cx = proposals[:,0]+0.5*w; cy = proposals[:,1]+0.5*h
    px = deltas[:,0]*w+cx-0.5*torch.exp(deltas[:,2])*w
    py = deltas[:,1]*h+cy-0.5*torch.exp(deltas[:,3])*h
    return torch.stack([px,py,deltas[:,0]*w+cx+0.5*torch.exp(deltas[:,2])*w,
                        deltas[:,1]*h+cy+0.5*torch.exp(deltas[:,3])*h],dim=1).clamp(min=0)

def gaussian_log_prob(deltas, mu, sigma):
    eps = (deltas - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    return -0.5*(eps.pow(2)+2*torch.log(sigma.unsqueeze(1))+math.log(2*math.pi)).sum(dim=-1)

def build_loaders():
    return build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})

def build_model():
    return build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}})

def freeze_except(m, parts):
    for p in m.parameters():p.requires_grad=False
    for part in parts:
        if isinstance(part, nn.Module):
            for p in part.parameters():p.requires_grad=True

@torch.no_grad()
def evaluate(model, vl):
    model.eval();preds,targs=[],[]
    for img,tgt in vl:
        out=model([i.to(DEV) for i in img])
        preds.extend([{k:v.cpu() for k,v in o.items()} for o in out])
        targs.extend([{k:v.cpu() for k,v in t.items()} for t in tgt])
    return evaluate_detection_predictions(preds,targs,iou_threshold=0.5,score_threshold=0.05)

def main():
    all_r=[]
    for alpha in ALPHAS:
        run_name=f"round261_iou_grpo_a{alpha}_s42";set_seed(42)
        model=build_model().to(DEV);ckpt=torch.load(CKPT,map_location=DEV);model.load_state_dict(ckpt["model"])
        ref_model=copy.deepcopy(model);freeze_except(ref_model,[]);ref_model.eval()
        freeze_except(model,[model.roi_heads.box_head,model.roi_heads.box_predictor])
        tl,vl=build_loaders()
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.SGD(params,lr=0.001,momentum=0.9,weight_decay=0.0005)
        rd=ensure_run_dir(run_name);h=[];best=-1.0
        pc={};rc={}
        model.rpn.register_forward_hook(lambda m,i,o:pc.update({"p":o[0]}))
        model.roi_heads.box_head.register_forward_pre_hook(lambda m,i:rc.update({"x":i[0]}))
        for ep in range(1,EPOCHS+1):
            model.train();td,trl,ta,tgdl=[0.0]*4;pos=0
            for imgs,tgts in tqdm(tl,desc=f"{run_name} e{ep}"):
                imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
                pc.clear();rc.clear()
                ld=model(imgs_d,tgts_t)
                if isinstance(ld,dict):det=sum(ld.values())
                else:det=sum(sum(d.values()) for d in ld if isinstance(d,dict))
                rf=rc.get("x");pr=pc.get("p");rl=torch.tensor(0.0,device=DEV)
                if rf is not None and pr is not None and rf.shape[0]>0:
                    N=rf.shape[0];bf=model.roi_heads.box_head(rf)
                    mu=model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
                    sigma=0.1;s=torch.full_like(mu,sigma,requires_grad=False)
                    eps=torch.randn(N,G,4,device=DEV)
                    # CRITICAL: detach mu so action doesn't follow policy
                    deltas=mu.detach().unsqueeze(1)+s.unsqueeze(1)*eps
                    log_probs=gaussian_log_prob(deltas,mu,s)
                    pc_=torch.cat(pr,dim=0);N=min(N,pc_.shape[0])
                    mu=mu[:N];deltas=deltas[:N];s=s[:N];log_probs=log_probs[:N]
                    ad=deltas.reshape(N*G,4);pe=pc_[:N].unsqueeze(1).expand(-1,G,-1).reshape(N*G,4)
                    boxes=decode_boxes(pe,ad)
                    # IoU reward per sample
                    npi=[p.shape[0] for p in pr]
                    img_map=[];[img_map.extend([i_]*p.shape[0]) for i_,p in enumerate(pr)]
                    img_map=img_map[:N]
                    iou_reward=torch.zeros(N,G,device=DEV)
                    for pi in range(N):
                        gt_boxes=tgts_t[img_map[pi]]["boxes"]
                        if len(gt_boxes)>0:
                            ious=box_iou(boxes[pi*G:(pi+1)*G],gt_boxes)
                            iou_reward[pi]=ious.max(dim=1).values
                    # GRPO advantage
                    r_mean=iou_reward.mean(dim=1,keepdim=True)
                    r_std=iou_reward.std(dim=1,keepdim=True).clamp_min(1e-6)
                    adv=(iou_reward-r_mean)/r_std
                    # Only positive proposals (best IoU > 0.3)
                    pos_mask=iou_reward.max(dim=1).values>0.3
                    if pos_mask.any():
                        rl=-(adv[pos_mask].detach()*log_probs[pos_mask]).mean()
                        pos+=pos_mask.sum().item()
                        ta+=(adv[pos_mask].abs()).mean().item()
                loss=det+alpha*rl
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                td+=det.item();trl+=rl.item();tgdl+=rl.item()
            em=evaluate(model,vl)
            row={"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"]};h.append(row)
            print(f"  e{ep}: AP50={em['ap50']:.4f} det={td:.1f} rl={trl:.3f} pos={pos} adv_abs={ta/max(pos,1):.4f}")
            if em["ap50"]>best:best=em["ap50"]
        em.update({"run_name":run_name,"alpha":alpha,"epochs":EPOCHS,"seed":42,"best_ap50":best,"history":h,"git_hash":GIT})
        save_json(em,rd/"eval_metrics.json");all_r.append(em)
        print(f"  DONE a{alpha}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")
    print("\n## Plan 2.61 IoU-GRPO Results")
    for r in all_r:print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

if __name__=="__main__":main()
