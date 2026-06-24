"""Plan 2.68: det-only continuation baseline. No verifier, no RL, pure fine-tune."""
import sys,json,subprocess,math,copy
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
import torchvision
from tqdm import tqdm
sys.path.insert(0,"E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json,ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed
GIT=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
DEV="cuda";CKPT="runs/round227_v1_baseline_20ep/checkpoint_best.pth";EPOCHS=5
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
    for seed in [42]:  # single seed, extendable
        run_name=f"round268_det_s{seed}";set_seed(seed)
        model=build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":2,"min_size":320,"max_size":320}}).to(DEV)
        ckpt=torch.load(CKPT,map_location=DEV);model.load_state_dict(ckpt["model"])
        fe(model,[model.roi_heads.box_head,model.roi_heads.box_predictor])
        tl,vl=build_penn_fudan_loaders({"data":{"root":"./data","max_size":320,"train_fraction":0.8,"num_workers":0},"train":{"batch_size":2}})
        rd=ensure_run_dir(run_name);h=[];best=-1.0
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.SGD(params,lr=0.001,momentum=0.9,weight_decay=0.0005)
        for ep in range(1,EPOCHS+1):
            model.train();td=0.0
            for imgs,tgts in tqdm(tl,desc=f"{run_name} e{ep}"):
                imgs_d=[i.to(DEV) for i in imgs];tgts_t=[{k:v.to(DEV) for k,v in t.items()} for t in tgts]
                ld=model(imgs_d,tgts_t)
                if isinstance(ld,dict):det=sum(ld.values())
                else:det=sum(sum(d.values()) for d in ld if isinstance(d,dict))
                opt.zero_grad(set_to_none=True);det.backward();opt.step()
                td+=det.item()
            em=ev(model,vl)
            row={"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"]};h.append(row)
            print(f"  e{ep}: AP50={em['ap50']:.4f} det={td:.1f}")
            if em["ap50"]>best:best=em["ap50"]
        em.update({"run_name":run_name,"config":"det_only","epochs":EPOCHS,"seed":seed,"best_ap50":best,"history":h,"git_hash":GIT})
        save_json(em,rd/"eval_metrics.json");all_r.append(em)
        print(f"  DONE s{seed}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")
    print("\n## Plan 2.68 Det-Only Results")
    for r in all_r:print(f"  s{r['seed']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")
if __name__=="__main__":main()
