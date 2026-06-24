"""Band-wise FFT manifold analysis for ROI crops.

Compares per-frequency-band structural features against the global 7168-dim
flatten baseline.  For each amplitude band (lo/mid/hi, each Nx256x7x4) we
extract several feature views, whiten them, and measure pair-consistency
with IoU ranking.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
)
from spectral_detection_posttrain.models import build_detector

warnings.filterwarnings("ignore", category=UserWarning)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def bm():
    return build_detector({
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
        }
    })

# ---------------------------------------------------------------------------
# FFT extraction — keep per-band structure
# ---------------------------------------------------------------------------

def extract_fft_bands(x: torch.Tensor, bands: tuple[float, float] = (0.3, 0.7)) -> dict[str, torch.Tensor]:
    """Extract per-band FFT amplitude tensors preserving (C, Hf, Wf) structure.

    Args:
        x: (N, C, H, W) spatial tensor (e.g. ROI features 7x7).

    Returns:
        Dict with keys "amp_lo", "amp_mid", "amp_hi", each (N, C, Hf, Wf).
        Hf = x.shape[-2], Wf = x.shape[-1] // 2 + 1  (rfft output size).
    """
    lo_thr, hi_thr = bands
    fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)  # (N, C, Hf, Wf)

    freq_h = torch.fft.fftfreq(x.shape[-2], device=x.device)
    freq_w = torch.fft.rfftfreq(x.shape[-1], device=x.device)
    grid_y, grid_x = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = radius / radius.max().clamp_min(1e-6)

    lo_mask = (radius <= lo_thr).float()   # (Hf, Wf)
    mid_mask = ((radius > lo_thr) & (radius <= hi_thr)).float()
    hi_mask = (radius > hi_thr).float()

    # Expand masks to (1, 1, Hf, Wf) for broadcasting
    lo_m = lo_mask.unsqueeze(0).unsqueeze(0)
    mid_m = mid_mask.unsqueeze(0).unsqueeze(0)
    hi_m = hi_mask.unsqueeze(0).unsqueeze(0)

    return {
        "amp_lo": amp * lo_m,
        "amp_mid": amp * mid_m,
        "amp_hi": amp * hi_m,
    }

# ---------------------------------------------------------------------------
# Per-band feature extractors — each returns (N, D)
# ---------------------------------------------------------------------------

def feat_flatten(band: torch.Tensor) -> np.ndarray:
    """Flatten to (D,) — single sample. Input: (C, Hf, Wf) or (N, C, Hf, Wf)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    return band.flatten(1).cpu().numpy().reshape(-1)


def feat_topk_energy(band: torch.Tensor, k: int = 8) -> np.ndarray:
    """Per-channel top-K frequency components. Input: (C, Hf, Wf) or (N, C, Hf, Wf)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    N, C, Hf, Wf = band.shape
    flat = band.reshape(N, C, -1)
    topk_vals, _ = torch.topk(flat, k=min(k, flat.shape[-1]), dim=-1)  # (N, C, K)
    return topk_vals.flatten(1).cpu().numpy().reshape(-1)


def feat_per_channel_stats(band: torch.Tensor) -> np.ndarray:
    """Per-channel mean/std/max over freq bins. Input: (C, Hf, Wf) or (N, C, Hf, Wf)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    flat = band.reshape(band.shape[0], band.shape[1], -1)  # (N, C, 28)
    mu = flat.mean(dim=-1)
    sg = flat.std(dim=-1)
    mx = flat.max(dim=-1).values
    return torch.cat([mu, sg, mx], dim=1).cpu().numpy().reshape(-1)


def feat_cross_channel_stats(band: torch.Tensor) -> np.ndarray:
    """Cross-channel stats per freq bin. Input: (C, Hf, Wf) or (N, C, Hf, Wf)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    N, C, Hf, Wf = band.shape
    flat = band.reshape(N, C, -1)  # (N, C, 28)
    mu = flat.mean(dim=1)      # (N, 28)
    va = flat.var(dim=1)       # (N, 28)
    std = (va + 1e-8).sqrt().unsqueeze(1)  # (N, 1, 28)
    z = (flat - mu.unsqueeze(1)) / std
    sk = (z ** 3).mean(dim=1)  # (N, 28)
    return torch.cat([mu, va, sk], dim=1).cpu().numpy().reshape(-1)


def feat_spatial_pool(band: torch.Tensor) -> np.ndarray:
    """2D spatial pooling on frequency grid. Input: (C, Hf, Wf) or (N, C, Hf, Wf)."""
    if band.dim() == 3:
        band = band.unsqueeze(0)
    avg = band.mean(dim=(-2, -1))   # (N, C)
    mx = band.max(dim=-1).values.max(dim=-1).values  # (N, C)
    return torch.cat([avg, mx], dim=1).cpu().numpy().reshape(-1)


# map name -> extractor function
EXTRACTORS: dict[str, Any] = {
    "flatten": feat_flatten,
    "topk8": lambda b: feat_topk_energy(b, k=8),
    "topk16": lambda b: feat_topk_energy(b, k=16),
    "perchan_stats": feat_per_channel_stats,
    "crosschan_stats": feat_cross_channel_stats,
    "spatial_pool": feat_spatial_pool,
}

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_proposals(model, val_loader, device):
    """Collect per-proposal IoU, confidence, and per-band FFT tensors.

    Returns list of dicts per image:
      - iou: (N,)
      - conf: (N,)
      - bands: dict[str, (N, C, Hf, Wf)]  — amplitude bands
      - gt_idx: (N,)
      - boxes: (N, 4)
    """
    model.eval()
    records = []
    sampled_props, box_head_in = {}, {}

    model.roi_heads.box_roi_pool.register_forward_pre_hook(
        lambda m, args: sampled_props.update({"p": [a.clone() for a in args[1]]})
    )
    model.roi_heads.box_head.register_forward_pre_hook(
        lambda m, args: box_head_in.update({"x": args[0]})
    )

    for images, targets in tqdm(val_loader, desc="inference"):
        images_d = [img.to(device) for img in images]
        tgts_t = [{k: v.to(device) for k, v in t.items()} for t in targets]
        sampled_props.clear()
        box_head_in.clear()

        model(images_d, tgts_t)

        sp_raw = sampled_props.get("p")
        rf = box_head_in.get("x")
        if sp_raw is None or rf is None or rf.shape[0] == 0:
            continue

        # classifier confidence
        bf = model.roi_heads.box_head(rf)
        cls_logits = model.roi_heads.box_predictor.cls_score(bf)
        conf = F.softmax(cls_logits, dim=-1)[:, 1].cpu()

        sp_cat = torch.cat(sp_raw, dim=0)
        N = sp_cat.shape[0]
        reg_out = model.roi_heads.box_predictor.bbox_pred(bf[:N])
        person_deltas = reg_out[:, 2:6]
        decoded = decode_boxes(sp_cat, person_deltas)

        # FFT bands on ROI features (before box_head)
        bands = extract_fft_bands(rf[:N].cpu())  # dict of (N, C, Hf, Wf)

        offset = 0
        for i_img, p_img in enumerate(sp_raw):
            n_p = p_img.shape[0]
            if n_p == 0:
                continue
            gt = tgts_t[i_img]["boxes"]
            gt_labels = tgts_t[i_img]["labels"]
            person_mask = gt_labels == 1
            gt_person = gt[person_mask] if person_mask.any() else gt

            iou = torch.zeros(n_p)
            gt_idx = torch.full((n_p,), -1, dtype=torch.long)
            if len(gt_person) > 0:
                iou_mat = box_iou(decoded[offset:offset + n_p], gt_person)
                iou, best_gt = iou_mat.max(dim=1)
                gt_idx = best_gt
                iou = iou.cpu()
                gt_idx = gt_idx.cpu()

            img_bands = {k: v[offset:offset + n_p] for k, v in bands.items()}

            records.append({
                "iou": iou,
                "conf": conf[offset:offset + n_p],
                "bands": img_bands,
                "gt_idx": gt_idx,
                "boxes": sp_cat[offset:offset + n_p].cpu(),
            })
            offset += n_p

    return records

# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def build_pairs(records):
    """Group by GT and build all intra-group pairs."""
    pairs = []
    for img_rec in records:
        iou = img_rec["iou"]
        bands = img_rec["bands"]
        gt_idx = img_rec["gt_idx"]
        groups = defaultdict(list)
        for p in range(len(iou)):
            if gt_idx[p] >= 0:
                groups[int(gt_idx[p])].append(p)
        for gid, idxs in groups.items():
            if len(idxs) < 2:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    borderline = (0.3 <= iou[a] <= 0.5) or (0.3 <= iou[b] <= 0.5)
                    pairs.append({
                        "iou_a": float(iou[a]),
                        "iou_b": float(iou[b]),
                        "bands_a": {k: v[a].numpy() for k, v in bands.items()},
                        "bands_b": {k: v[b].numpy() for k, v in bands.items()},
                        "borderline": borderline,
                    })
    return pairs

# ---------------------------------------------------------------------------
# Metric computation per band + extractor
# ---------------------------------------------------------------------------

def whiten(X: np.ndarray, n_comp: int = 50) -> np.ndarray:
    """Z-score + PCA whitening."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 2:
        return X
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(n_comp, Xs.shape[0] - 1, Xs.shape[1])
    if k < 1:
        return Xs
    pca = PCA(n_components=k, whiten=True, random_state=42)
    return pca.fit_transform(Xs)


def whiten_incremental(X: np.ndarray, n_comp: int = 50, batch_size: int = 8192) -> np.ndarray:
    """Z-score + IncrementalPCA for arrays that don't fit in memory."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 2:
        return X
    # Online standardization: compute mean/var in chunks, then transform
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    std = np.sqrt(var + 1e-8)
    # We can't easily do full PCA in chunks without IncrementalPCA
    from sklearn.decomposition import IncrementalPCA
    k = min(n_comp, X.shape[0] - 1, X.shape[1])
    if k < 1:
        return (X - mean) / std
    ipca = IncrementalPCA(n_components=k, whiten=True, batch_size=batch_size)
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = (X[start:end] - mean) / std
        ipca.partial_fit(chunk)
    # Transform in chunks
    out = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out[start:end] = ipca.transform((X[start:end] - mean) / std)
    return out


def compute_band_metrics(pairs, band_name: str, extractor_name: str, extractor_fn):
    """Compute pair-consistency for a single band + extractor combination.

    Returns dict with consistency scores.
    """
    # Stack all vectors for this band
    all_a = [p["bands_a"][band_name] for p in pairs]
    all_b = [p["bands_b"][band_name] for p in pairs]
    all_vec = all_a + all_b
    X = np.stack([extractor_fn(torch.from_numpy(v)) for v in all_vec], axis=0).astype(np.float32)

    # Whiten
    X_w = whiten(X, n_comp=50)

    M = len(pairs)

    # Metric: negative distance to TP centroid in whitened space
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    tp_center = X_w[tp_mask].mean(axis=0) if tp_mask.any() else X_w.mean(axis=0)
    d = np.linalg.norm(X_w - tp_center, axis=1)
    scores = -d  # higher = closer to TP = better

    s_a = scores[:M]
    s_b = scores[M:]

    agree = 0
    total = 0
    border_agree = 0
    border_total = 0
    overlap = 0
    for i, p in enumerate(pairs):
        metric_order = s_a[i] > s_b[i]
        iou_order = p["iou_a"] > p["iou_b"]
        if metric_order == iou_order:
            agree += 1
        total += 1
        if p["borderline"]:
            if metric_order == iou_order:
                border_agree += 1
            border_total += 1
        metric_best = "a" if s_a[i] > s_b[i] else "b"
        iou_best = "a" if p["iou_a"] > p["iou_b"] else "b"
        if metric_best == iou_best:
            overlap += 1

    return {
        "pair_consistency": agree / total if total else 0.0,
        "borderline_consistency": border_agree / border_total if border_total else 0.0,
        "pair_overlap": overlap / total if total else 0.0,
        "n_pairs": total,
        "n_borderline": border_total,
        "dim": X.shape[1],
    }


def compute_global_baseline(pairs):
    """Compute the old 7168-dim flattened-all-bands baseline for comparison."""
    # Concatenate all 3 bands -> flatten -> same as old pipeline
    all_a = []
    all_b = []
    for p in pairs:
        a = np.concatenate([p["bands_a"][k].flatten() for k in ["amp_lo", "amp_mid", "amp_hi"]]).astype(np.float32)
        b = np.concatenate([p["bands_b"][k].flatten() for k in ["amp_lo", "amp_mid", "amp_hi"]]).astype(np.float32)
        all_a.append(a)
        all_b.append(b)
    X = np.stack(all_a + all_b, axis=0)
    X_w = whiten(X, n_comp=50)
    M = len(pairs)

    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    tp_center = X_w[tp_mask].mean(axis=0) if tp_mask.any() else X_w.mean(axis=0)
    d = np.linalg.norm(X_w - tp_center, axis=1)
    scores = -d
    s_a = scores[:M]
    s_b = scores[M:]

    agree = 0; total = 0; border_agree = 0; border_total = 0; overlap = 0
    for i, p in enumerate(pairs):
        metric_order = s_a[i] > s_b[i]
        iou_order = p["iou_a"] > p["iou_b"]
        if metric_order == iou_order: agree += 1
        total += 1
        if p["borderline"]:
            if metric_order == iou_order: border_agree += 1
            border_total += 1
        metric_best = "a" if s_a[i] > s_b[i] else "b"
        iou_best = "a" if p["iou_a"] > p["iou_b"] else "b"
        if metric_best == iou_best: overlap += 1

    return {
        "pair_consistency": agree / total if total else 0.0,
        "borderline_consistency": border_agree / border_total if border_total else 0.0,
        "pair_overlap": overlap / total if total else 0.0,
        "n_pairs": total,
        "n_borderline": border_total,
        "dim": X.shape[1],
    }

# ---------------------------------------------------------------------------
# Multi-band combination
# ---------------------------------------------------------------------------

def compute_multi_band(pairs, band_names: list[str], extractor_name: str, extractor_fn):
    """Concatenate features from multiple bands, then whiten and score."""
    all_a = []
    all_b = []
    for p in pairs:
        a = np.concatenate([extractor_fn(torch.from_numpy(p["bands_a"][bn])) for bn in band_names], axis=0).astype(np.float32)
        b = np.concatenate([extractor_fn(torch.from_numpy(p["bands_b"][bn])) for bn in band_names], axis=0).astype(np.float32)
        all_a.append(a)
        all_b.append(b)
    X = np.stack(all_a + all_b, axis=0)
    X = np.asarray(X, dtype=np.float32)
    X_w = whiten(X, n_comp=50)
    M = len(pairs)

    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    tp_center = X_w[tp_mask].mean(axis=0) if tp_mask.any() else X_w.mean(axis=0)
    d = np.linalg.norm(X_w - tp_center, axis=1)
    scores = -d
    s_a = scores[:M]
    s_b = scores[M:]

    agree = 0; total = 0; border_agree = 0; border_total = 0; overlap = 0
    for i, p in enumerate(pairs):
        metric_order = s_a[i] > s_b[i]
        iou_order = p["iou_a"] > p["iou_b"]
        if metric_order == iou_order: agree += 1
        total += 1
        if p["borderline"]:
            if metric_order == iou_order: border_agree += 1
            border_total += 1
        metric_best = "a" if s_a[i] > s_b[i] else "b"
        iou_best = "a" if p["iou_a"] > p["iou_b"] else "b"
        if metric_best == iou_best: overlap += 1

    return {
        "pair_consistency": agree / total if total else 0.0,
        "borderline_consistency": border_agree / border_total if border_total else 0.0,
        "pair_overlap": overlap / total if total else 0.0,
        "n_pairs": total,
        "n_borderline": border_total,
        "dim": X.shape[1],
    }

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: dict):
    print("\n" + "=" * 90)
    print("Band-wise FFT Manifold Pair Consistency vs IoU Ranking")
    print("=" * 90)
    header = f"{'Config':<28} {'Dim':>6} {'Pairs':>7} {'Consist%':>10} {'Border%':>10} {'Overlap%':>10} {'nBorder':>7}"
    print(header)
    print("-" * 90)

    # Global baseline first
    r = results.get("global_7168")
    if r:
        print(f"{'GLOBAL_7168 (baseline)':<28} {r['dim']:>6} {r['n_pairs']:>7} {r['pair_consistency']*100:>9.1f}% {r['borderline_consistency']*100:>9.1f}% {r['pair_overlap']*100:>9.1f}% {r['n_borderline']:>7}")
        print("-" * 90)

    # Per-band, per-extractor
    for key in sorted(results.keys()):
        if key == "global_7168":
            continue
        r = results[key]
        print(f"{key:<28} {r['dim']:>6} {r['n_pairs']:>7} {r['pair_consistency']*100:>9.1f}% {r['borderline_consistency']*100:>9.1f}% {r['pair_overlap']*100:>9.1f}% {r['n_borderline']:>7}")

    print("-" * 90)
    print("Target: >70% consistency for DPO viability")
    print("=" * 90)


def find_best(results: dict):
    """Print summary of which band/extractor wins."""
    # Exclude global baseline
    band_results = {k: v for k, v in results.items() if k != "global_7168"}
    if not band_results:
        return

    best_key = max(band_results, key=lambda k: band_results[k]["pair_consistency"])
    best = band_results[best_key]

    print(f"\nBest config: {best_key}")
    print(f"  Consistency = {best['pair_consistency']*100:.1f}%")
    print(f"  Borderline  = {best['borderline_consistency']*100:.1f}%")
    print(f"  Overlap     = {best['pair_overlap']*100:.1f}%")
    print(f"  Dimension   = {best['dim']}")

    baseline = results.get("global_7168")
    if baseline:
        delta = (best["pair_consistency"] - baseline["pair_consistency"]) * 100
        print(f"  vs Global 7168 baseline: {delta:+.1f} pp")

    # Best per band
    print("\nBest per band:")
    for band in ["amp_lo", "amp_mid", "amp_hi"]:
        band_keys = [k for k in band_results if k.startswith(band)]
        if not band_keys:
            continue
        bk = max(band_keys, key=lambda k: band_results[k]["pair_consistency"])
        br = band_results[bk]
        print(f"  {band}: {bk} -> {br['pair_consistency']*100:.1f}% (dim={br['dim']})")

    # Best per extractor
    print("\nBest per extractor:")
    for ext in EXTRACTORS.keys():
        ext_keys = [k for k in band_results if k.endswith(ext)]
        if not ext_keys:
            continue
        ek = max(ext_keys, key=lambda k: band_results[k]["pair_consistency"])
        er = band_results[ek]
        print(f"  {ext}: {ek} -> {er['pair_consistency']*100:.1f}% (dim={er['dim']})")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading model...")
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])

    print("Loading val data...")
    _, val_loader = build_penn_fudan_loaders_320(batch_size=1)

    print("Collecting proposals with per-band FFT...")
    records = collect_proposals(model, val_loader, DEV)
    print(f"  -> {len(records)} images with proposals")

    print("Building pairs within same GT...")
    pairs = build_pairs(records)
    print(f"  -> {len(pairs)} pairs")
    if len(pairs) == 0:
        print("No pairs found. Exiting.")
        return

    # Subsample to manageable size for memory/speed
    MAX_PAIRS = 10000
    rng = np.random.RandomState(42)
    if len(pairs) > MAX_PAIRS:
        idxs = rng.choice(len(pairs), size=MAX_PAIRS, replace=False)
        pairs = [pairs[i] for i in idxs]
        print(f"  -> subsampled to {len(pairs)} pairs for analysis")

    results = {}

    # 1. Global 7168-dim baseline (all bands flattened)
    print("\n[1/3] Global 7168-dim baseline...")
    results["global_7168"] = compute_global_baseline(pairs)

    # 2. Per-band + per-extractor
    print("[2/3] Per-band feature extractors...")
    for band_name in ["amp_lo", "amp_mid", "amp_hi"]:
        for ext_name, ext_fn in EXTRACTORS.items():
            key = f"{band_name}_{ext_name}"
            print(f"  {key} ...")
            results[key] = compute_band_metrics(pairs, band_name, ext_name, ext_fn)

    # 3. Multi-band combinations
    print("[3/3] Multi-band combinations...")
    for ext_name, ext_fn in EXTRACTORS.items():
        # lo+mid
        key = f"lo+mid_{ext_name}"
        print(f"  {key} ...")
        results[key] = compute_multi_band(pairs, ["amp_lo", "amp_mid"], ext_name, ext_fn)
        # mid+hi
        key = f"mid+hi_{ext_name}"
        print(f"  {key} ...")
        results[key] = compute_multi_band(pairs, ["amp_mid", "amp_hi"], ext_name, ext_fn)
        # lo+hi
        key = f"lo+hi_{ext_name}"
        print(f"  {key} ...")
        results[key] = compute_multi_band(pairs, ["amp_lo", "amp_hi"], ext_name, ext_fn)
        # all three
        key = f"all_{ext_name}"
        print(f"  {key} ...")
        results[key] = compute_multi_band(pairs, ["amp_lo", "amp_mid", "amp_hi"], ext_name, ext_fn)

    # Report
    print_results(results)
    find_best(results)

    # Save
    import json
    out = {
        "results": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()} for k, v in results.items()},
        "n_images": len(records),
        "n_pairs": len(pairs),
    }
    with open("bandwise_manifold_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved: bandwise_manifold_results.json")


if __name__ == "__main__":
    main()
