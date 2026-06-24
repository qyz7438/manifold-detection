"""Scan 5 manifold distance metrics on baseline val set for DPO pair consistency.

Goal: measure whether manifold distance ranking agrees with IoU ranking
for proposals matched to the same GT. >70% consistency means DPO is viable.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import NearestNeighbors
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.experiments.runner_utils import (
    build_penn_fudan_loaders_320,
    decode_boxes,
    extract_perchan_fft,
)
from spectral_detection_posttrain.models import build_detector

warnings.filterwarnings("ignore", category=UserWarning)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"


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


@torch.no_grad()
def collect_proposals(model, val_loader, device):
    """Run inference on val set and collect per-proposal data.

    Returns list of dicts, one per image:
      - iou: (N,) max IoU with any GT
      - conf: (N,) person-class confidence
      - fft: (N, 7168) amplitude features (3 bands * channels)
      - gt_idx: (N,) matched GT index (-1 if none)
      - boxes: (N, 4) proposals
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

        # forward (training mode to get proposals)
        model(images_d, tgts_t)

        sp_raw = sampled_props.get("p")
        rf = box_head_in.get("x")
        if sp_raw is None or rf is None or rf.shape[0] == 0:
            continue

        # ROI features -> classifier confidence
        bf = model.roi_heads.box_head(rf)
        cls_logits = model.roi_heads.box_predictor.cls_score(bf)
        conf = F.softmax(cls_logits, dim=-1)[:, 1].cpu()  # person confidence

        # Decode boxes to compute IoU
        sp_cat = torch.cat(sp_raw, dim=0)
        N = sp_cat.shape[0]
        reg_out = model.roi_heads.box_predictor.bbox_pred(bf[:N])
        person_deltas = reg_out[:, 2:6]
        decoded = decode_boxes(sp_cat, person_deltas)

        # FFT on ROI-pooled features: (N, C, H, W)
        # extract_perchan_fft returns (N, C*6); we take amplitude bands [:, 0*C:3*C]
        fft_full = extract_perchan_fft(rf[:N].cpu())  # (N, C*6)
        C = fft_full.shape[1] // 6
        fft_amp = fft_full[:, :3 * C].reshape(N, -1)  # (N, 3*C) -> flatten to (N, 7168) when C=256

        # Per-image IoU + GT matching
        offset = 0
        for i_img, p_img in enumerate(sp_raw):
            n_p = p_img.shape[0]
            if n_p == 0:
                continue
            gt = tgts_t[i_img]["boxes"]
            gt_labels = tgts_t[i_img]["labels"]
            # only person class (label == 1)
            person_mask = gt_labels == 1
            gt_person = gt[person_mask] if person_mask.any() else gt

            iou = torch.zeros(n_p)
            gt_idx = torch.full((n_p,), -1, dtype=torch.long)
            if len(gt_person) > 0:
                iou_mat = box_iou(decoded[offset:offset + n_p], gt_person)  # (n_p, n_gt)
                iou, best_gt = iou_mat.max(dim=1)
                gt_idx = best_gt
                # zero out if best match is not person (shouldn't happen after mask)
                iou = iou.cpu()
                gt_idx = gt_idx.cpu()

            records.append({
                "iou": iou,
                "conf": conf[offset:offset + n_p],
                "fft": fft_amp[offset:offset + n_p],
                "gt_idx": gt_idx,
                "boxes": sp_cat[offset:offset + n_p].cpu(),
            })
            offset += n_p

    return records


def build_pairs(records):
    """Group proposals by (image_idx, gt_idx) and build all pairs within each group.

    Returns list of dicts with keys: iou_a, iou_b, fft_a, fft_b, borderline
    """
    pairs = []
    for img_rec in records:
        iou = img_rec["iou"]
        fft = img_rec["fft"]
        gt_idx = img_rec["gt_idx"]
        # group by gt
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
                        "fft_a": fft[a].numpy(),
                        "fft_b": fft[b].numpy(),
                        "borderline": borderline,
                    })
    return pairs


def compute_manifold_metrics(pairs):
    """Compute 5 manifold distance metrics for all pairs.

    Returns dict of metric_name -> list of scores (higher = better quality).
    """
    # Stack all unique FFT vectors to fit global models
    all_fft = []
    for p in pairs:
        all_fft.append(p["fft_a"])
        all_fft.append(p["fft_b"])
    X = np.stack(all_fft, axis=0)  # (2*M, D)

    metrics = {}
    M = len(pairs)

    # 1. PCA distance to TP cluster center (IoU > 0.5)
    print("  fitting PCA...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    # identify TP indices (IoU > 0.5) for centroid
    tp_mask = np.array([p["iou_a"] > 0.5 for p in pairs] + [p["iou_b"] > 0.5 for p in pairs])
    tp_center = X_pca[tp_mask].mean(axis=0) if tp_mask.any() else X_pca.mean(axis=0)
    d_pca = np.linalg.norm(X_pca - tp_center, axis=1)
    metrics["pca"] = -d_pca  # higher = closer to TP center = better

    # 2. k-NN local density (PCA space, k=30)
    print("  fitting k-NN...")
    knn = NearestNeighbors(n_neighbors=min(30, X_pca.shape[0] - 1))
    knn.fit(X_pca)
    dists, _ = knn.kneighbors(X_pca)
    mean_knn = dists[:, 1:].mean(axis=1)  # exclude self
    metrics["knn_density"] = 1.0 / (mean_knn + 1e-8)

    # 3. Spectral clustering label as score
    print("  fitting spectral clustering...")
    n_clusters = min(3, X_pca.shape[0])
    if X_pca.shape[0] >= 3:
        sc = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42)
        labels = sc.fit_predict(X_pca)
        # treat label 0 as "best" (arbitrary, but consistent)
        metrics["spectral"] = -labels.astype(float)
    else:
        metrics["spectral"] = np.zeros(2 * M)

    # 4. Diffusion distance (spectral embedding n_components=10)
    print("  fitting diffusion embedding...")
    n_comp = min(10, X_pca.shape[0] - 1)
    if n_comp >= 2:
        try:
            # spectral_embedding expects affinity matrix, not data matrix
            # Build k-NN affinity first
            from sklearn.neighbors import kneighbors_graph
            knn_aff = kneighbors_graph(X_pca, n_neighbors=min(30, X_pca.shape[0]-1), mode='connectivity', include_self=False)
            X_diff = spectral_embedding(knn_aff, n_components=n_comp, random_state=42)
            tp_center_diff = X_diff[tp_mask].mean(axis=0) if tp_mask.any() else X_diff.mean(axis=0)
            d_diff = np.linalg.norm(X_diff - tp_center_diff, axis=1)
            metrics["diffusion"] = -d_diff
        except Exception as e:
            print(f"  diffusion skipped ({e})")
            metrics["diffusion"] = np.zeros(2 * M)
    else:
        metrics["diffusion"] = np.zeros(2 * M)

    # 5. UMAP distance (optional)
    try:
        import umap
        print("  fitting UMAP...")
        reducer = umap.UMAP(n_components=10, n_neighbors=min(15, X_pca.shape[0] - 1), random_state=42)
        X_umap = reducer.fit_transform(X_pca)
        tp_center_umap = X_umap[tp_mask].mean(axis=0) if tp_mask.any() else X_umap.mean(axis=0)
        d_umap = np.linalg.norm(X_umap - tp_center_umap, axis=1)
        metrics["umap"] = -d_umap
    except Exception as e:
        print(f"  UMAP skipped ({e})")
        metrics["umap"] = np.zeros(2 * M)

    return metrics


def evaluate_consistency(pairs, metrics):
    """Compute pair consistency and overlap metrics.

    Returns dict of results per metric.
    """
    M = len(pairs)
    results = {}

    for name, scores in metrics.items():
        s_a = scores[:M]
        s_b = scores[M:]

        # Pair consistency: metric order == IoU order
        agree = 0
        total = 0
        border_agree = 0
        border_total = 0
        for i, p in enumerate(pairs):
            # metric higher = better; iou higher = better
            metric_order = s_a[i] > s_b[i]
            iou_order = p["iou_a"] > p["iou_b"]
            if metric_order == iou_order:
                agree += 1
            total += 1
            if p["borderline"]:
                if metric_order == iou_order:
                    border_agree += 1
                border_total += 1

        # Pair overlap: best-vs-worst match
        overlap = 0
        for i, p in enumerate(pairs):
            metric_best = "a" if s_a[i] > s_b[i] else "b"
            iou_best = "a" if p["iou_a"] > p["iou_b"] else "b"
            if metric_best == iou_best:
                overlap += 1

        results[name] = {
            "pair_consistency": agree / total if total else 0.0,
            "borderline_consistency": border_agree / border_total if border_total else 0.0,
            "pair_overlap": overlap / total if total else 0.0,
            "n_pairs": total,
            "n_borderline": border_total,
        }

    return results


def print_table(results):
    print("\n" + "=" * 70)
    print("Manifold Distance vs IoU Ranking Consistency (Baseline Val Set)")
    print("=" * 70)
    header = f"{'Metric':<18} {'Pairs':>8} {'Consist%':>10} {'Border%':>10} {'Overlap%':>10} {'nBorder':>8}"
    print(header)
    print("-" * 70)
    for name in ["pca", "knn_density", "spectral", "diffusion", "umap"]:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<18} {r['n_pairs']:>8} {r['pair_consistency']*100:>9.1f}% {r['borderline_consistency']*100:>9.1f}% {r['pair_overlap']*100:>9.1f}% {r['n_borderline']:>8}")
    print("-" * 70)
    print("Target: >70% consistency for DPO viability\n")


def main():
    print("Loading model...")
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])

    print("Loading val data...")
    _, val_loader = build_penn_fudan_loaders_320(batch_size=1)

    print("Collecting proposals with IoU, confidence, FFT...")
    records = collect_proposals(model, val_loader, DEV)
    print(f"  -> {len(records)} images with proposals")

    print("Building pairs within same GT...")
    pairs = build_pairs(records)
    print(f"  -> {len(pairs)} pairs")
    if len(pairs) == 0:
        print("No pairs found. Exiting.")
        return

    print("Computing manifold metrics...")
    metrics = compute_manifold_metrics(pairs)

    print("Evaluating consistency...")
    results = evaluate_consistency(pairs, metrics)

    print_table(results)

    # Save raw results
    import json
    out = {
        "metrics": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                         for kk, vv in v.items()} for k, v in results.items()},
        "n_images": len(records),
        "n_pairs": len(pairs),
    }
    with open("manifold_dpo_consistency.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: manifold_dpo_consistency.json")


if __name__ == "__main__":
    main()
