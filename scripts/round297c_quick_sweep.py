#!/usr/bin/env python3
"""Round297c quick optimizer/diagnostic sweep.

Purpose: resolve Round297b contradiction where supervised MLP top-3 can match/exceed
DPO while DPO has better top-1/top-5. This script is read-only w.r.t. source code and
writes only .agent_reports/action_pref/round297c_quick_sweep.*.
"""
from __future__ import annotations

import json, math, random, time
from pathlib import Path
from typing import Iterable, List, Dict
import numpy as np
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / '.agent_reports' / 'action_pref'
DATA = OUT / 'round296_internal_features.npz'
JSON = OUT / 'round297c_quick_sweep.json'
MD = OUT / 'round297c_quick_sweep.md'

BATCH = 4096
EPOCHS = 30
LR = 1e-3
DPO_SEEDS = [297, 298, 299]
MLP_SEEDS = [297, 298, 299, 300, 301]
MARGINS = [0.0, 0.005, 0.01, 0.02, 0.03]
BETAS = [0.5, 1.0, 2.0]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def groups(prop_ids):
    order=np.argsort(prop_ids, kind='stable'); ids=prop_ids[order]
    starts=np.r_[0, np.flatnonzero(ids[1:]!=ids[:-1])+1]; ends=np.r_[starts[1:], len(order)]
    for s,e in zip(starts,ends): yield order[s:e]


def pairs(prop_ids, y, margin):
    out=[]
    for idx in groups(prop_ids):
        yy=y[idx]; gi,bi=np.where((yy[:,None]-yy[None,:])>margin)
        if len(gi): out.append(np.stack([idx[gi],idx[bi]],1))
    return np.concatenate(out,0).astype(np.int64) if out else np.zeros((0,2),np.int64)


def eval_scores(scores, prop_ids, y):
    top1=top3=top5=0; sp=[]; n=0
    for idx in groups(prop_ids):
        yy=y[idx]; ss=scores[idx]; order=np.argsort(-ss, kind='stable')
        best=yy.max(); mask=yy>=best-1e-12
        top1 += bool(mask[order[:1]].any()); top3 += bool(mask[order[:3]].any()); top5 += bool(mask[order[:5]].any())
        if len(idx)>1 and np.std(ss)>1e-12 and np.std(yy)>1e-12:
            r=spearmanr(ss, yy).correlation
            if r is not None and math.isfinite(float(r)): sp.append(float(r))
        n+=1
    return {'top1':top1/n,'top3':top3/n,'top5':top5/n,'spearman':float(np.mean(sp)) if sp else float('nan')}

class Scorer(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train_dpo(seed, beta, margin, Xtr, ytr, ptr, Xva, yva, pva):
    set_seed(seed); t=time.time()
    pp=pairs(ptr,ytr,margin)
    sc=StandardScaler(); xtr=sc.fit_transform(Xtr).astype('float32'); xva=sc.transform(Xva).astype('float32')
    model=Scorer(xtr.shape[1]); opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    xt=torch.from_numpy(xtr); pt=torch.from_numpy(pp)
    losses=[]
    for ep in range(EPOCHS):
        perm=torch.randperm(pt.shape[0]); ls=[]
        for st in range(0, pt.shape[0], BATCH):
            b=pt[perm[st:st+BATCH]]; sg=model(xt[b[:,0]]); sb=model(xt[b[:,1]])
            loss=-F.logsigmoid(beta*(sg-sb)).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ls.append(float(loss))
        losses.append(float(np.mean(ls)))
    with torch.no_grad(): s=model(torch.from_numpy(xva)).numpy().astype('float32')
    m=eval_scores(s,pva,yva); m.update({'seed':seed,'beta':beta,'margin':margin,'pairs':int(pp.shape[0]),'loss_last':losses[-1],'seconds':time.time()-t})
    return m

def train_mlp(seed, Xtr,ytr,pva,Xva,yva):
    t=time.time()
    model=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, early_stopping=True, random_state=seed))
    model.fit(Xtr,ytr); s=model.predict(Xva).astype('float32')
    m=eval_scores(s,pva,yva); m.update({'seed':seed,'seconds':time.time()-t}); return m

def main():
    z=np.load(DATA, allow_pickle=True)
    Xtr=np.concatenate([z['train_X_A_geom'],z['train_X_D_logits']],1).astype('float32')
    Xva=np.concatenate([z['val_X_A_geom'],z['val_X_D_logits']],1).astype('float32')
    ytr=z['train_iou'].astype('float32'); yva=z['val_iou'].astype('float32')
    ptr=z['train_prop_id'].astype('int64'); pva=z['val_prop_id'].astype('int64')
    mlp=[]
    for s in MLP_SEEDS:
        r=train_mlp(s,Xtr,ytr,pva,Xva,yva); mlp.append(r); print('MLP',r,flush=True)
    dpo=[]
    for margin in MARGINS:
        for beta in BETAS:
            runs=[]
            for seed in DPO_SEEDS:
                r=train_dpo(seed,beta,margin,Xtr,ytr,ptr,Xva,yva,pva); runs.append(r); print('DPO',r,flush=True)
            mean={k:float(np.mean([r[k] for r in runs])) for k in ['top1','top3','top5','spearman','loss_last','seconds']}
            std={k:float(np.std([r[k] for r in runs])) for k in ['top1','top3','top5','spearman']}
            dpo.append({'beta':beta,'margin':margin,'runs':runs,'mean':mean,'std':std,'pairs':runs[0]['pairs']})
    out={'status':'ok','mlp_runs':mlp,'mlp_mean':{k:float(np.mean([r[k] for r in mlp])) for k in ['top1','top3','top5','spearman']},'mlp_std':{k:float(np.std([r[k] for r in mlp])) for k in ['top1','top3','top5','spearman']},'dpo_grid':dpo}
    JSON.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    best_top3=max(dpo,key=lambda x:x['mean']['top3']); best_top1=max(dpo,key=lambda x:x['mean']['top1']); best_top5=max(dpo,key=lambda x:x['mean']['top5'])
    lines=['# Round297c Quick Sweep','',f"MLP mean: top1={out['mlp_mean']['top1']:.4f}, top3={out['mlp_mean']['top3']:.4f}, top5={out['mlp_mean']['top5']:.4f}, spearman={out['mlp_mean']['spearman']:.4f}",'','| beta | margin | pairs | top1 | top3 | top5 | spearman |','|---:|---:|---:|---:|---:|---:|---:|']
    for g in sorted(dpo, key=lambda x:(-x['mean']['top3'], -x['mean']['top1'])):
        m=g['mean']; lines.append(f"| {g['beta']} | {g['margin']} | {g['pairs']} | {m['top1']:.4f} | {m['top3']:.4f} | {m['top5']:.4f} | {m['spearman']:.4f} |")
    lines += ['',f"Best top3 DPO: beta={best_top3['beta']} margin={best_top3['margin']} mean={best_top3['mean']}",f"Best top1 DPO: beta={best_top1['beta']} margin={best_top1['margin']} mean={best_top1['mean']}",f"Best top5 DPO: beta={best_top5['beta']} margin={best_top5['margin']} mean={best_top5['mean']}",f"Artifacts: {JSON}, {MD}"]
    MD.write_text('\n'.join(lines),encoding='utf-8')
    print('WROTE',JSON); print('WROTE',MD)
if __name__=='__main__': main()
