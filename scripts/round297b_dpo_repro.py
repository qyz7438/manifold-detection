#!/usr/bin/env python3
"""Round297b minimal reproducibility rerun for G1_logits_geom DPO.

Input: .agent_reports/action_pref/round296_internal_features.npz
Output:
  - .agent_reports/action_pref/round297b_dpo_repro.json
  - .agent_reports/action_pref/round297b_dpo_repro.md
  - .agent_reports/action_pref/round297b_dpo_predictions.npz

This intentionally reruns only the claimed best setting from Round297:
G1_logits_geom = X_A_geom + X_D_logits, DPO beta=1.0, margin=0.01.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / ".agent_reports" / "action_pref"
DATA_PATH = OUT_DIR / "round296_internal_features.npz"
JSON_PATH = OUT_DIR / "round297b_dpo_repro.json"
MD_PATH = OUT_DIR / "round297b_dpo_repro.md"
NPZ_PATH = OUT_DIR / "round297b_dpo_predictions.npz"

BETA = 1.0
MARGIN = 0.01
EPOCHS = 30
BATCH_SIZE = 4096
LR = 1e-3
SEEDS = [297, 298, 299]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def group_indices(prop_ids: np.ndarray) -> Iterable[np.ndarray]:
    order = np.argsort(prop_ids, kind="stable")
    ids = prop_ids[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        yield order[s:e]


def make_pairs(prop_ids: np.ndarray, y: np.ndarray, margin: float) -> np.ndarray:
    pairs: List[np.ndarray] = []
    for idx in group_indices(prop_ids):
        yy = y[idx]
        # good action row index first, bad action row index second
        diff = yy[:, None] - yy[None, :]
        gi, bi = np.where(diff > margin)
        if len(gi):
            pairs.append(np.stack([idx[gi], idx[bi]], axis=1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(pairs, axis=0).astype(np.int64)


def eval_scores(scores: np.ndarray, prop_ids: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    top1 = top3 = top5 = 0
    spears: List[float] = []
    n = 0
    for idx in group_indices(prop_ids):
        yy = y[idx]
        ss = scores[idx]
        order = np.argsort(-ss, kind="stable")
        best_y = float(np.max(yy))
        # If ties exist in true IoU, any max-IoU action is accepted.
        best_mask = yy >= best_y - 1e-12
        top1 += bool(best_mask[order[:1]].any())
        top3 += bool(best_mask[order[:3]].any())
        top5 += bool(best_mask[order[:5]].any())
        if len(idx) > 1 and np.std(ss) > 1e-12 and np.std(yy) > 1e-12:
            r = spearmanr(ss, yy).correlation
            if r is not None and math.isfinite(float(r)):
                spears.append(float(r))
        n += 1
    return {
        "n_props": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "top5": top5 / n,
        "spearman": float(np.mean(spears)) if spears else float("nan"),
    }


class Scorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_dpo(seed: int, X_train: np.ndarray, y_train: np.ndarray, prop_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, prop_val: np.ndarray, pairs: np.ndarray) -> Dict[str, object]:
    set_seed(seed)
    t0 = time.time()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)

    model = Scorer(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    xt = torch.from_numpy(Xtr)
    pair_t = torch.from_numpy(pairs)
    losses: List[float] = []

    for epoch in range(EPOCHS):
        perm = torch.randperm(pair_t.shape[0])
        epoch_losses: List[float] = []
        for start in range(0, pair_t.shape[0], BATCH_SIZE):
            b = pair_t[perm[start:start + BATCH_SIZE]]
            sg = model(xt[b[:, 0]])
            sb = model(xt[b[:, 1]])
            loss = -F.logsigmoid(BETA * (sg - sb)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))

    with torch.no_grad():
        val_scores = model(torch.from_numpy(Xva)).numpy().astype(np.float32)
        train_scores = model(torch.from_numpy(Xtr)).numpy().astype(np.float32)
    metrics = eval_scores(val_scores, prop_val, y_val)
    metrics.update({
        "seed": seed,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "seconds": time.time() - t0,
    })
    return {"metrics": metrics, "val_scores": val_scores, "train_scores": train_scores, "losses": np.array(losses, dtype=np.float32)}


def train_baselines(X_train: np.ndarray, y_train: np.ndarray, prop_val: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=297)),
        "MLPRegressor": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=297, early_stopping=True)),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=297, n_jobs=-1),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict(X_val).astype(np.float32)
        out[name] = eval_scores(scores, prop_val, y_val)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA_PATH, allow_pickle=True)
    X_train = np.concatenate([data["train_X_A_geom"], data["train_X_D_logits"]], axis=1).astype(np.float32)
    X_val = np.concatenate([data["val_X_A_geom"], data["val_X_D_logits"]], axis=1).astype(np.float32)
    y_train = data["train_iou"].astype(np.float32)
    y_val = data["val_iou"].astype(np.float32)
    prop_train = data["train_prop_id"].astype(np.int64)
    prop_val = data["val_prop_id"].astype(np.int64)

    pairs = make_pairs(prop_train, y_train, MARGIN)
    print(f"Round297b G1_logits_geom: train={X_train.shape}, val={X_val.shape}, pairs={pairs.shape[0]}, margin={MARGIN}", flush=True)

    random_scores = np.zeros_like(y_val, dtype=np.float32)
    random_eval = {"note": "constant-score stable ordering; random expectation for 24 actions top3=0.125", **eval_scores(random_scores, prop_val, y_val)}
    baselines = train_baselines(X_train, y_train, prop_val, X_val, y_val)

    dpo_runs = []
    npz_payload = {
        "val_iou": y_val,
        "val_prop_id": prop_val,
        "val_action_id": data["val_action_id"].astype(np.int64),
        "val_base_iou": data["val_base_iou"].astype(np.float32),
        "train_pair_good_bad": pairs,
    }
    for seed in SEEDS:
        run = train_dpo(seed, X_train, y_train, prop_train, X_val, y_val, prop_val, pairs)
        dpo_runs.append(run["metrics"])
        npz_payload[f"dpo_seed_{seed}_val_scores"] = run["val_scores"]
        npz_payload[f"dpo_seed_{seed}_train_scores"] = run["train_scores"]
        npz_payload[f"dpo_seed_{seed}_losses"] = run["losses"]
        m = run["metrics"]
        print(f"seed={seed} top1={m['top1']:.4f} top3={m['top3']:.4f} top5={m['top5']:.4f} spearman={m['spearman']:.4f} loss={m['loss_first']:.4f}->{m['loss_last']:.4f} sec={m['seconds']:.1f}", flush=True)

    top3s = np.array([r["top3"] for r in dpo_runs], dtype=float)
    summary = {
        "round": "297b",
        "status": "PASS" if float(top3s.mean()) >= 0.30 else "PARTIAL_OR_FAIL",
        "feature_group": "G1_logits_geom",
        "feature_dim": int(X_train.shape[1]),
        "method": "DPO",
        "beta": BETA,
        "margin": MARGIN,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "data": {
            "train_rows": int(X_train.shape[0]),
            "train_props": int(len(set(prop_train.tolist()))),
            "val_rows": int(X_val.shape[0]),
            "val_props": int(len(set(prop_val.tolist()))),
            "train_pairs": int(pairs.shape[0]),
        },
        "random_constant_order": random_eval,
        "baselines": baselines,
        "dpo_runs": dpo_runs,
        "dpo_mean": {
            k: float(np.mean([r[k] for r in dpo_runs]))
            for k in ["top1", "top3", "top5", "spearman", "loss_first", "loss_last", "seconds"]
        },
        "dpo_std": {
            k: float(np.std([r[k] for r in dpo_runs], ddof=0))
            for k in ["top1", "top3", "top5", "spearman"]
        },
        "comparison_to_recovered_round297": {
            "recovered_top1": 0.1404,
            "recovered_top3": 0.3034,
            "recovered_top5": 0.4663,
            "recovered_spearman": 0.6361,
        },
    }

    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(NPZ_PATH, **npz_payload)

    def pct(x: float) -> str:
        return f"{100*x:.2f}%"

    rows = []
    for name, m in baselines.items():
        rows.append(f"| supervised/{name} | - | {pct(m['top1'])} | {pct(m['top3'])} | {pct(m['top5'])} | {m['spearman']:+.4f} |")
    for m in dpo_runs:
        rows.append(f"| DPO seed={m['seed']} | β={BETA}, m={MARGIN} | {pct(m['top1'])} | {pct(m['top3'])} | {pct(m['top5'])} | {m['spearman']:+.4f} |")
    dm = summary["dpo_mean"]
    rows.append(f"| DPO mean±std | 3 seeds | {pct(dm['top1'])}±{pct(summary['dpo_std']['top1'])} | {pct(dm['top3'])}±{pct(summary['dpo_std']['top3'])} | {pct(dm['top5'])}±{pct(summary['dpo_std']['top5'])} | {dm['spearman']:+.4f}±{summary['dpo_std']['spearman']:.4f} |")

    md = f"""# Round297b — G1_logits_geom DPO Reproducibility Rerun

## Setup

- Feature group: `G1_logits_geom = train_X_A_geom + train_X_D_logits`
- Feature dim: {X_train.shape[1]}
- DPO setting: β={BETA}, margin={MARGIN}, epochs={EPOCHS}
- Seeds: {SEEDS}
- Train rows/proposals: {X_train.shape[0]} / {len(set(prop_train.tolist()))}
- Val rows/proposals: {X_val.shape[0]} / {len(set(prop_val.tolist()))}
- Train preference pairs: {pairs.shape[0]}

## Results

| Method | Setting | Top-1 | Top-3 | Top-5 | Spearman |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Verdict

Status: `{summary['status']}`

Recovered Round297 best was Top-1 14.04%, Top-3 30.34%, Top-5 46.63%, Spearman +0.6361.
Round297b 3-seed DPO mean is Top-1 {pct(dm['top1'])}, Top-3 {pct(dm['top3'])}, Top-5 {pct(dm['top5'])}, Spearman {dm['spearman']:+.4f}.

Artifacts:
- `{JSON_PATH}`
- `{MD_PATH}`
- `{NPZ_PATH}`
"""
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"WROTE {JSON_PATH}")
    print(f"WROTE {MD_PATH}")
    print(f"WROTE {NPZ_PATH}")


if __name__ == "__main__":
    main()
