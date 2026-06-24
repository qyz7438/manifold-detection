"""NWPU PG post-training on converged baseline. Tests if PG can improve over saturated model."""
import sys, json, copy
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import numpy as np
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"; SEED = 42; BATCH = 2; EPOCHS = 10; MAX_SIZE = 480
CKPT_PATH = "runs/nwpu_baseline_best.pth"
DATA = Path("data/NWPU VHR-10 dataset"); ANNOT = Path("data/NWPU_VHR10_coco.json")
G = 4; SIGMA = 1.0; ENERGY_BETA = 0.05; RL_WEIGHT = 0.02; KL_WEIGHT = 0.1
IOU_LO = 0.3; IOU_HI = 0.7
set_seed(SEED)

# Action set
def build_actions():
    scales = [0.02, 0.05, 0.10, 0.20]; acts = []
    for s in scales: acts.extend([(s,0,0,0),(-s,0,0,0),(0,s,0,0),(0,-s,0,0)])
    for s in scales: acts.extend([(0,0,s,0),(0,0,-s,0)])
    return torch.tensor(acts, dtype=torch.float32)
ACTIONS = build_actions().to(DEV); N_ACTIONS = 24

class NWPUDataset(Dataset):
    def __init__(self, root, coco_json, img_ids, max_size):
        self.root = Path(root); self.max_size = max_size
        self.coco = json.loads(Path(coco_json).read_text())
        self.img_infos = {img["id"]: img for img in self.coco["images"] if img["id"] in img_ids}
        self.img_ids = list(self.img_infos.keys())
        anns = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in img_ids:
                anns.setdefault(ann["image_id"], []).append(ann)
        self.anns = anns
    def __len__(self): return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]; info = self.img_infos[img_id]
        img_path = self.root/"positive image set"/info["file_name"]
        if not img_path.exists(): img_path = self.root/"negative image set"/info["file_name"]
        img = TF.to_tensor(Image.open(str(img_path)).convert("RGB"))
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x,y,w,h = ann["bbox"]; boxes.append([x,y,x+w,y+h]); labels.append(ann["category_id"])
        tgt = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}
        _, H, W = img.shape
        if max(H,W) > self.max_size:
            scale = self.max_size/max(H,W); nh, nw = int(H*scale), int(W*scale)
            img = F.interpolate(img.unsqueeze(0), size=(nh,nw), mode="bilinear").squeeze(0)
            tgt["boxes"] = tgt["boxes"] * scale
        return img, tgt

def collate(batch): return tuple(zip(*batch))

# Data
coco = json.loads(ANNOT.read_text())
all_ids = list(set(img["id"] for img in coco["images"] if Path(DATA/"positive image set"/img["file_name"]).exists()))
np.random.seed(42); np.random.shuffle(all_ids)
n_train = int(0.7*len(all_ids))
train_ids = set(all_ids[:n_train]); val_ids = set(all_ids[n_train:])
train_ds = NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE)
val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

# Helpers
def crop_img(raw, boxes):
    if raw.dim()==4: raw=raw[0]
    M=boxes.shape[0]; _,H,W=raw.shape; crops=[]
    for i in range(M):
        x1,y1,x2,y2 = boxes[i].long().clamp(0)
        x1=x1.clamp(max=W-2);x2=x2.clamp(max=W-1);y1=y1.clamp(max=H-2);y2=y2.clamp(max=H-1)
        if x2<=x1+1: x2=x1+2
        if y2<=y1+1: y2=y1+2
        c = raw[:,y1:y2,x1:x2].float()/255.0
        crops.append(F.interpolate(c.unsqueeze(0), size=(7,7), mode="bilinear", align_corners=False).squeeze(0))
    return torch.stack(crops)

def energy_img(crops):
    M,C,H,W=crops.shape
    fft=torch.fft.rfft2(crops,dim=(-2,-1),norm="ortho");amp=torch.abs(fft)
    fh=torch.fft.fftfreq(H,device=DEV);fw=torch.fft.rfftfreq(W,device=DEV)
    Y,X=torch.meshgrid(fh,fw,indexing="ij");r=torch.sqrt(X**2+Y**2);R=r.max().clamp_min(1e-6);rn=r/R
    lo=(rn<=0.3).float();md=((rn>0.3)&(rn<=0.7)).float();hi=(rn>0.7).float()
    al=(amp*lo).flatten(2).sum(2);am=(amp*md).flatten(2).sum(2);ah=(amp*hi).flatten(2).sum(2)
    return (al/(al+am+ah+1e-8)).mean(dim=1)

def apply_actions_batch(boxes, indices):
    a=ACTIONS[indices]; w=boxes[:,2]-boxes[:,0];h=boxes[:,3]-boxes[:,1]
    cx=boxes[:,0]+0.5*w;cy=boxes[:,1]+0.5*h
    nc=cx+a[:,0]*w;ny=cy+a[:,1]*h;nw=torch.clamp(w*(1.0+a[:,2]),min=1);nh=torch.clamp(h*(1.0+a[:,3]),min=1)
    return torch.stack([nc-0.5*nw,ny-0.5*nh,nc+0.5*nw,ny+0.5*nh],dim=1).clamp(min=0)

def grpo(reward):
    m=reward.mean(dim=1,keepdim=True);s=reward.std(dim=1,keepdim=True).clamp_min(1e-6)
    return (reward-m)/s

# Model
num_classes=11
model=build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_fpn","model_name":"fasterrcnn_mobilenet_v3_large_fpn","pretrained":True,"num_classes":num_classes,"min_size":MAX_SIZE,"max_size":MAX_SIZE}}).to(DEV)
ckpt=torch.load(CKPT_PATH, map_location=DEV); model.load_state_dict(ckpt["model"])

# Unfreeze
for p in model.backbone.body.parameters(): p.requires_grad=False
for p in model.backbone.fpn.parameters(): p.requires_grad=True
for p in model.rpn.parameters(): p.requires_grad=True
for p in model.roi_heads.box_head.parameters(): p.requires_grad=True
for p in model.roi_heads.box_predictor.parameters(): p.requires_grad=True

sampled_props, box_head_in = {}, {}
model.roi_heads.box_roi_pool.register_forward_pre_hook(lambda m,a: sampled_props.update({"p":[x.clone() for x in a[1]]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,a: box_head_in.update({"x":a[0]}))

# Each config
for cfg_name, use_energy, use_shuffle in [("det_only_unf",False,False), ("pg_energy",True,False), ("pg_shuffle",True,True)]:
    print(f"\n=== {cfg_name} ===")
    model_cfg = copy.deepcopy(model)
    baseline = copy.deepcopy(model)
    baseline.eval()
    for p in baseline.parameters(): p.requires_grad=False

    body_params, head_params = [], []
    for n,p in model_cfg.named_parameters():
        if not p.requires_grad: continue
        if "box_head" in n or "box_predictor" in n: head_params.append(p)
        else: body_params.append(p)
    opt = torch.optim.SGD([{"params":body_params,"lr":0.0001},{"params":head_params,"lr":0.001}], lr=0.001, momentum=0.9, weight_decay=0.0005)

    bw_base = baseline.roi_heads.box_predictor.bbox_pred.weight.detach().clone()
    bb_base = baseline.roi_heads.box_predictor.bbox_pred.bias.detach().clone()

    for ep in range(1, EPOCHS+1):
        model_cfg.train(); td, trl, tkl = 0, 0, 0
        for imgs, tgts in tqdm(train_loader, desc=f"{cfg_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k:v.to(DEV) for k,v in t.items()} for t in tgts]
            raw = imgs_d[0]; sh = (raw.shape[-2], raw.shape[-1])
            sampled_props.clear(); box_head_in.clear()

            ld = model_cfg(imgs_d, tgts_t)
            det = sum(v for v in ld.values())
            rf = box_head_in.get("x"); sr = sampled_props.get("p")
            rl = torch.tensor(0.0, device=DEV); kl_loss = torch.tensor(0.0, device=DEV)

            if not use_energy and not use_shuffle:
                loss = det
            elif rf is not None and sr is not None and rf.shape[0] > 0:
                N = rf.shape[0]
                kl_loss = KL_WEIGHT*((model_cfg.roi_heads.box_predictor.bbox_pred.weight-bw_base).pow(2).sum()+(model_cfg.roi_heads.box_predictor.bbox_pred.bias-bb_base).pow(2).sum())

                sc = torch.cat(sr, dim=0)[:N]
                bf = model_cfg.roi_heads.box_head(rf)
                mu = model_cfg.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
                bw=sc[:,2]-sc[:,0]; bh=sc[:,3]-sc[:,1]; bc=sc[:,0]+0.5*bw; by=sc[:,1]+0.5*bh
                dx_b=mu[:,0]/10.0; dy_b=mu[:,1]/10.0; dw_b=mu[:,2]/5.0; dh_b=mu[:,3]/5.0
                base = torch.stack([dx_b*bw+bc-0.5*torch.exp(dw_b)*bw, dy_b*bh+by-0.5*torch.exp(dh_b)*bh, dx_b*bw+bc+0.5*torch.exp(dw_b)*bw, dy_b*bh+by+0.5*torch.exp(dh_b)*bh], dim=1).clamp(min=0)

                # Sigma=1.0 perturbations
                s = torch.full_like(mu, SIGMA)
                deltas = mu.detach().unsqueeze(1)+s.unsqueeze(1)*torch.randn(N,G,4,device=DEV)
                delta_flat = deltas.reshape(-1,4)
                se = sc.repeat_interleave(G, dim=0)
                bw2=se[:,2]-se[:,0];bh2=se[:,3]-se[:,1];bc2=se[:,0]+0.5*bw2;by2=se[:,1]+0.5*bh2
                dx=delta_flat[:,0]/10.0;dy=delta_flat[:,1]/10.0;dw=delta_flat[:,2]/5.0;dh=delta_flat[:,3]/5.0
                pboxes = torch.stack([dx*bw2+bc2-0.5*torch.exp(dw)*bw2,dy*bh2+by2-0.5*torch.exp(dh)*bh2,dx*bw2+bc2+0.5*torch.exp(dw)*bw2,dy*bh2+by2+0.5*torch.exp(dh)*bh2],dim=1).clamp(min=0)

                # IoU
                iou_r = torch.zeros(N,G,device=DEV)
                off_p = 0
                for i_img, p in enumerate(sr):
                    np_i = min(p.shape[0], N-off_p)
                    if np_i<=0: break
                    idx_s=off_p*G;idx_e=(off_p+np_i)*G
                    gt = tgts_t[i_img]["boxes"]
                    if len(gt)>0:
                        ious = box_iou(pboxes[idx_s:idx_e], gt)
                        iou_r[off_p:off_p+np_i] = ious.max(dim=1).values.view(np_i,G)
                    off_p += np_i
                reward = 2*iou_r - 1

                # Energy
                gated = torch.zeros(N,G,device=DEV)
                if use_energy or use_shuffle:
                    crops = crop_img(raw, pboxes)
                    en = energy_img(crops).view(N,G)
                    if use_shuffle:
                        en = en.reshape(-1)[torch.randperm(N*G, device=DEV)].view(N,G)
                    en_pen = -torch.sigmoid(15*(en-0.5))*ENERGY_BETA
                    gmax = iou_r.max(dim=1).values
                    bmask = ((gmax>=IOU_LO)&(gmax<IOU_HI)).unsqueeze(1).float()
                    gated = en_pen * bmask

                s = torch.full_like(mu, SIGMA)
                log_probs = -0.5*(((deltas-mu.unsqueeze(1))/s.unsqueeze(1))**2+2*torch.log(s.unsqueeze(1))).sum(dim=-1) - 0.5*4*torch.log(torch.tensor(2*3.14159,device=DEV))
                adv = grpo(reward) + gated
                soft_w = iou_r.max(dim=1).values.clamp(0,1).unsqueeze(1)
                rl = -(adv.detach()*log_probs*soft_w).mean()
                loss = det + RL_WEIGHT*rl + kl_loss
            else:
                loss = det

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            td += det.item(); trl += rl.item() if torch.is_tensor(rl) else 0; tkl += kl_loss.item() if torch.is_tensor(kl_loss) else 0

        # Eval
        model_cfg.eval(); ps, ts = [], []
        for img, tgt in val_loader:
            with torch.no_grad(): pred = model_cfg([img[0].to(DEV)])[0]
            ps.append({k:v.cpu() for k,v in pred.items()}); ts.append({k:v.cpu() for k,v in tgt[0].items()})
        from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
        em = evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)
        print(f"  e{ep}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f} det={td:.1f} rl={trl:.2f}")
