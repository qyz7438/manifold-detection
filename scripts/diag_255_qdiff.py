"""Diagnostic: check if pixel FFT quality differs between two deltas."""
import sys,torch,torch.nn.functional as F,numpy as np
sys.path.insert(0,'E:/CLIproject/RLimage')
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
set_seed(42)
DEV='cuda';K=2;PIX=64

def pix_q(patches):
    gray=patches.float().mean(dim=1)
    fft=torch.fft.fft2(gray).abs();mf=fft.flatten(1)
    t=mf.sum(dim=1,keepdim=True).clamp_min(1e-6)
    hf=mf[:,mf.shape[1]//2:].sum(dim=1)/t.squeeze(1)
    mn=mf/t;ent=-(mn*torch.log(mn+1e-6)).sum(dim=1)
    me=torch.log(torch.tensor(float(mf.shape[1]),device=DEV));en=1.0-ent/me
    pv=torch.angle(torch.fft.fft2(gray)+1e-6).flatten(1).std(dim=1).clamp_max(1.0)
    return (0.3*hf+0.4*en+0.3*(1.0-pv)).clamp(0,1)

cfg={'model':{'name':'fasterrcnn_mobilenet_v3_large_320_fpn','model_name':'fasterrcnn_mobilenet_v3_large_320_fpn','pretrained':True,'num_classes':2,'min_size':320,'max_size':320}}
model=build_detector(cfg).to(DEV)
ckpt=torch.load('runs/round227_v1_baseline_20ep/checkpoint_best.pth',map_location=DEV)
model.load_state_dict(ckpt['model']);model.eval()
loaders=build_penn_fudan_loaders({'data':{'root':'./data','max_size':320,'train_fraction':0.8,'num_workers':0},'train':{'batch_size':2}})
pc={};rc={}
model.rpn.register_forward_hook(lambda m,i,o:pc.update({'p':o[0]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,i:rc.update({'x':i[0]}))

same_count=0;total_pairs=0;all_diffs=[]
for images,targets in loaders[0]:
    pc.clear();rc.clear()
    model([img.to(DEV) for img in images],[{k:v.to(DEV) for k,v in t.items()} for t in targets])
    rf=rc.get('x');pr=pc.get('p')
    if rf is None or rf.shape[0]==0:continue
    N=rf.shape[0];bf=model.roi_heads.box_head(rf)
    mu=model.roi_heads.box_predictor.bbox_pred(bf)[:,-4:]
    pc_=torch.cat(pr,dim=0);N=min(N,pc_.shape[0],200);mu=mu[:N]
    eps=torch.randn(N,K,4,device=DEV);deltas=mu.unsqueeze(1)+0.1*eps
    ad=deltas.reshape(N*K,4);pe=pc_[:N].unsqueeze(1).expand(-1,K,-1).reshape(N*K,4)
    w=pe[:,2]-pe[:,0];h=pe[:,3]-pe[:,1];cx=pe[:,0]+0.5*w;cy=pe[:,1]+0.5*h
    px=ad[:,0]*w+cx-0.5*torch.exp(ad[:,2])*w
    py=ad[:,1]*h+cy-0.5*torch.exp(ad[:,3])*h
    all_boxes=torch.stack([px,py,ad[:,0]*w+cx+0.5*torch.exp(ad[:,2])*w,ad[:,1]*h+cy+0.5*torch.exp(ad[:,3])*h],dim=1).clamp(min=0)
    npi=[p.shape[0] for p in pr]
    ii=torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(npi)],dim=0)[:N]
    patches=[]
    for idx in range(min(N*K,256)):
        pj=min(idx//K,N-1);img_i=ii[pj].item();img=images[img_i];box=all_boxes[idx]
        x1,y1=max(0,int(box[0].round().item())),max(0,int(box[1].round().item()))
        x2,y2=min(img.shape[-1],max(x1+1,int(box[2].round().item()))),min(img.shape[-2],max(y1+1,int(box[3].round().item())))
        crop=img[:,y1:y2,x1:x2]
        if crop.shape[-1]>=4 and crop.shape[-2]>=4:
            crop=F.interpolate(crop.unsqueeze(0).float(),(PIX,PIX),mode='bilinear',align_corners=False).squeeze(0)
            patches.append(crop)
        else:patches.append(torch.zeros(3,PIX,PIX))
    if patches:
        pb=torch.stack(patches).to(DEV);q_all=pix_q(pb)
        Kt=N*K;qp=torch.zeros(Kt,device=DEV);qp[:len(q_all)]=q_all;qm=qp.view(N,K)
        for pi in range(N):
            d=abs(qm[pi,0].item()-qm[pi,1].item())
            all_diffs.append(d)
            if d<1e-5:same_count+=1
            total_pairs+=1
    break

diffs=np.array(all_diffs)
print(f'Total pairs: {total_pairs}')
print(f'Identical (diff<1e-5): {same_count} ({100*same_count/total_pairs:.1f}%)')
print(f'Quality diff mean: {diffs.mean():.6f}')
print(f'Quality diff std: {diffs.std():.6f}')
print(f'Quality diff max: {diffs.max():.6f}')
print(f'Quality diff range: [{diffs.min():.6f}, {diffs.max():.6f}]')
print(f'Chosen fraction (delta0 wins): {np.mean([qm.reshape(-1,2)[i].argmax()==0 for i in range(N)]):.2%}')
if same_count/total_pairs > 0.5:
    print('VERDICT: Quality near-identical -> DPO preference is noise')
elif diffs.mean() < 0.001:
    print('VERDICT: Quality diff too small (<0.001) -> DPO preference is noise')
else:
    print('VERDICT: Quality has real signal')
