"""Plan 2.59: FFT high-pass filter -> iFFT -> edge truncation quality DPO.

Combines: FFT high-pass (keep r>0.3, removes smooth background) ->
iFFT -> reconstructed edge-enhanced image -> Sobel boundary detection.

The FFT high-pass amplifies edges by removing low-frequency smooth regions,
making the Sobel response at boundaries cleaner and more sensitive to 1-2px shifts.
"""
import sys, json, subprocess, math, copy
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"; CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
K_SAMPLES = 2; BETAS = [0.5, 1.0]; EPOCHS = 10; PIX = 64; BW = 3


def fft_edge_truncation(patches):
    """FFT high-pass (r>0.3) -> iFFT -> edge -> boundary quality."""
    gray = patches.float().mean(dim=1)  # (N, 64, 64)
    H, W = gray.shape[-2], gray.shape[-1]
    fft = torch.fft.fft2(gray)
    # High-pass filter: keep only r > 0.3 (edges and textures, no smooth regions)
    freq = torch.fft.fftfreq(max(H, W), device=patches.device)
    Y, X = torch.meshgrid(freq[:H], freq[:W], indexing='ij')
    r = torch.sqrt(X**2 + Y**2)
    high_mask = r > 0.15  # keep medium+high frequencies
    fft_hp = fft * high_mask.float().unsqueeze(0)
    # iFFT back to spatial domain: edge-enhanced image
    edge_enhanced = torch.fft.ifft2(fft_hp).real
    # Sobel on edge-enhanced image
    gx = edge_enhanced[:, :, 1:] - edge_enhanced[:, :, :-1]
    gy = edge_enhanced[:, 1:, :] - edge_enhanced[:, :-1, :]
    edge = torch.sqrt(gx[:, :-1, :].pow(2) + gy[:, :, :-1].pow(2) + 1e-6)
    total = edge.flatten(1).sum(dim=1).clamp_min(1e-6)
    # Boundary mask
    H_e, W_e = edge.shape[-2], edge.shape[-1]
    bm = torch.zeros(H_e, W_e, device=patches.device, dtype=torch.bool)
    bm[:BW, :] = True; bm[-BW:, :] = True; bm[:, :BW] = True; bm[:, -BW:] = True
    boundary = (edge * bm.float().unsqueeze(0)).flatten(1).sum(dim=1)
    return (1.0 - boundary / total).clamp(0, 1)


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
    for p in m.parameters(): p.requires_grad = False
    for part in parts:
        if isinstance(part, nn.Module):
            for p in part.parameters(): p.requires_grad = True

@torch.no_grad()
def evaluate(model, vl):
    model.eval(); preds, targs = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        preds.extend([{k:v.cpu() for k,v in o.items()} for o in out])
        targs.extend([{k:v.cpu() for k,v in t.items()} for t in tgt])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


def main():
    all_r = []
    for beta in BETAS:
        run_name = f"round259_fft_edge_b{beta}_s42"; set_seed(42)
        model = build_model().to(DEV); ckpt = torch.load(CKPT, map_location=DEV); model.load_state_dict(ckpt["model"])
        ref = copy.deepcopy(model); freeze_except(ref, []); ref.eval()
        freeze_except(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])
        tl, vl = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        rd = ensure_run_dir(run_name); h = []; best = -1.0
        pc = {}; rc = {}
        model.rpn.register_forward_hook(lambda m,i,o: pc.update({"p":o[0]}))
        model.roi_heads.box_head.register_forward_pre_hook(lambda m,i: rc.update({"x":i[0]}))

        for ep in range(1, EPOCHS+1):
            model.train(); td, tdpo = 0.0, 0.0
            for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
                imgs_d = [i.to(DEV) for i in imgs]; tgts_t = [{k:v.to(DEV) for k,v in t.items()} for t in tgts]
                pc.clear(); rc.clear()
                ld = model(imgs_d, tgts_t)
                if isinstance(ld, dict): det = sum(ld.values())
                elif isinstance(ld, (list,tuple)): det = sum(sum(d.values()) for d in ld if isinstance(d,dict))
                else: det = sum(ld)
                rf = rc.get("x"); pr = pc.get("p"); dpo = torch.tensor(0.0,device=DEV)
                if rf is not None and pr is not None and rf.shape[0]>0:
                    N = rf.shape[0]; bf = model.roi_heads.box_head(rf)
                    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
                    s = torch.full_like(mu, 0.1, requires_grad=False)
                    eps = torch.randn(N, K_SAMPLES, 4, device=DEV)
                    deltas = mu.unsqueeze(1)+s.unsqueeze(1)*eps
                    lp = gaussian_log_prob(deltas, mu, s)
                    with torch.no_grad():
                        rf_ref = ref.roi_heads.box_head(rf); rm = ref.roi_heads.box_predictor.bbox_pred(rf_ref)[:,-4:]
                    rd_ = deltas.detach(); rs = torch.full_like(rm, 0.1)
                    lpr = gaussian_log_prob(rd_, rm, rs)
                    pc_ = torch.cat(pr, dim=0); N = min(N, pc_.shape[0])
                    mu=mu[:N]; deltas=deltas[:N]; s=s[:N]; lp=lp[:N]; lpr=lpr[:N]
                    ad = deltas.reshape(N*K_SAMPLES,4)
                    pe = pc_[:N].unsqueeze(1).expand(-1,K_SAMPLES,-1).reshape(N*K_SAMPLES,4)
                    ab = decode_boxes(pe, ad)
                    npi = [p.shape[0] for p in pr]
                    ii = torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(npi)], dim=0)[:N]
                    patches = []
                    for idx in range(min(N*K_SAMPLES,256)):
                        pj = min(idx//K_SAMPLES,N-1); img_i=ii[pj].item(); img=imgs[img_i]; box=ab[idx]
                        x1,y1=max(0,int(box[0].round().item())),max(0,int(box[1].round().item()))
                        x2,y2=min(img.shape[-1],max(x1+1,int(box[2].round().item()))),min(img.shape[-2],max(y1+1,int(box[3].round().item())))
                        crop = img[:,y1:y2,x1:x2]
                        if crop.shape[-1]>=4 and crop.shape[-2]>=4:
                            crop=F.interpolate(crop.unsqueeze(0).float(),(PIX,PIX),mode='bilinear',align_corners=False).squeeze(0)
                            patches.append(crop)
                        else: patches.append(torch.zeros(3,PIX,PIX))
                    Kt = N*K_SAMPLES
                    if patches:
                        pb = torch.stack(patches).to(DEV); qa = fft_edge_truncation(pb)
                        qp = torch.zeros(Kt,device=DEV); qp[:len(qa)]=qa; qm=qp.view(N,K_SAMPLES)
                    else: qm = torch.zeros(N,K_SAMPLES,device=DEV)
                    qd = (qm[:,0]-qm[:,1]).abs(); ch = qm[:,0]>=qm[:,1]; vd = qd>0.02
                    lc=torch.where(ch&vd,lp[:,0],lp[:,1]); lr_=torch.where(ch&vd,lp[:,1],lp[:,0])
                    lrc=torch.where(ch&vd,lpr[:,0],lpr[:,1]); lrr_=torch.where(ch&vd,lpr[:,1],lpr[:,0])
                    ratio=lc-lrc-lr_+lrr_
                    if vd.any(): dpo = -F.logsigmoid(beta*ratio[vd]).mean()
                loss = det + dpo; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                td+=det.item(); tdpo+=dpo.item()
            em = evaluate(model, vl); row={"epoch":ep,"val_ap50":em["ap50"],"val_ap75":em["ap75"]}; h.append(row)
            print(f"  e{ep}: AP50={em['ap50']:.4f} det={td:.1f} dpo={tdpo:.3f}")
            if em["ap50"]>best: best=em["ap50"]
        em.update({"run_name":run_name,"beta":beta,"epochs":EPOCHS,"seed":42,"best_ap50":best,"history":h,"git_hash":GIT})
        save_json(em, rd/"eval_metrics.json"); all_r.append(em)
        print(f"  DONE b{beta}: AP50={em['ap50']:.4f} AP75={em['ap75']:.4f}")
    print("\n## Plan 2.59 FFT Edge Truncation Results")
    for r in all_r: print(f"  b{r['beta']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

if __name__=="__main__": main()
