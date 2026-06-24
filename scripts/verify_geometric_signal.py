"""Perceptual manifold geometry validation for FFT spectral features."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.matching.box_iou import box_iou
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.spectral.roi_crop import crop_and_resize_roi
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import load_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def compute_full_fft_amplitude(roi: torch.Tensor) -> torch.Tensor:
    """Compute flattened log-amplitude spectrum for a [C,H,W] ROI."""
    gray = roi.mean(dim=0)  # [H, W]
    h, w = gray.shape
    window = torch.outer(torch.hann_window(h, device=roi.device), torch.hann_window(w, device=roi.device))
    gray = gray * window
    fft = torch.fft.fft2(gray, dim=(-2, -1))
    amp = torch.fft.fftshift(torch.abs(fft))
    amp = torch.log1p(amp)
    amp = (amp - amp.min()) / (amp.max() - amp.min()).clamp(min=1e-6)
    return amp.flatten()  # [H*W]


def extract_dense_proposals(model, image, device, score_thresh=0.01):
    """Run inference and return all proposals before NMS, with scores."""
    model.eval()
    with torch.no_grad():
        # FasterRCNN returns detections after NMS by default
        # We need to hook into rpn to get pre-NMS proposals
        # For simplicity: use model's forward but with a lower score threshold
        # Actually, let's get RPN proposals directly
        transformed_images, _ = model.transform([image.to(device)], None)
        features = model.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        # RPN proposals
        proposals, _ = model.rpn(transformed_images, features, None)
        proposals = proposals[0]  # [N, 4] on transformed image scale

        # Get original image size for scaling back
        orig_h, orig_w = image.shape[-2:]
        new_h, new_w = transformed_images.image_sizes[0]

        # Scale proposals back to original image coordinates
        scale_x = orig_w / float(new_w)
        scale_y = orig_h / float(new_h)
        proposals[:, [0, 2]] *= scale_x
        proposals[:, [1, 3]] *= scale_y

        # Get box features and scores from box_head + box_predictor
        if len(proposals) > 0:
            pooled = model.roi_heads.box_roi_pool(features, [proposals], transformed_images.image_sizes)
            box_features = model.roi_heads.box_head(pooled)
            class_logits, box_regression = model.roi_heads.box_predictor(box_features)
            scores = torch.softmax(class_logits, dim=-1)[:, 1]  # person class score
        else:
            scores = torch.empty((0,), device=device)

    keep = scores >= score_thresh
    return proposals[keep].cpu(), scores[keep].cpu()


def match_proposals_to_gt(proposals, scores, gt_boxes, gt_labels, iou_thresh=0.5):
    """Match proposals to GT and return arrays with IoU, is_tp, etc."""
    if len(proposals) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    ious = box_iou(proposals, gt_boxes)  # [N, M]
    best_ious, best_gt_idx = ious.max(dim=1)
    best_ious = best_ious.numpy()
    best_gt_idx = best_gt_idx.numpy()

    # Only match if IoU >= threshold and same class (all are person=1 here)
    is_tp = best_ious >= iou_thresh
    matched_gt = best_gt_idx.copy()
    matched_gt[~is_tp] = -1

    return best_ious, is_tp, matched_gt, scores.numpy()


def main():
    config_path = Path("runs/mvp_pf_baseline/config.yaml")
    checkpoint_path = Path("runs/mvp_pf_baseline/checkpoint_last.pth")

    config = load_config(str(config_path))
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config)

    # Build model
    model_cfg = dict(config)
    model_cfg["model"] = dict(config["model"])
    model_cfg["model"]["pretrained"] = False
    model = build_detector(model_cfg).to(device)
    load_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    # Build val loader
    _, val_loader = build_penn_fudan_loaders(config, limit_val=None, batch_size=1)

    all_features = []
    all_ious = []
    all_scores = []
    all_is_tp = []

    print("Extracting dense proposals and FFT features...")
    for images, targets in tqdm(val_loader, desc="val images"):
        image = images[0]
        target = targets[0]
        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()

        proposals, scores = extract_dense_proposals(model, image, device, score_thresh=0.01)
        if len(proposals) == 0:
            continue

        ious, is_tp, matched_gt, score_np = match_proposals_to_gt(proposals, scores, gt_boxes, gt_labels)

        # Compute FFT features for each proposal
        for i in range(len(proposals)):
            roi = crop_and_resize_roi(image, proposals[i], size=128)
            feat = compute_full_fft_amplitude(roi).cpu().numpy()
            all_features.append(feat)
            all_ious.append(ious[i])
            all_scores.append(score_np[i])
            all_is_tp.append(is_tp[i])

    X = np.stack(all_features, axis=0)  # [N, D]
    ious = np.array(all_ious)
    scores = np.array(all_scores)
    is_tp = np.array(all_is_tp)

    print(f"Total proposals: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"TP (IoU>0.5): {is_tp.sum()}")
    print(f"FP (IoU<0.3): {(ious < 0.3).sum()}")

    # Split groups
    tp_mask = is_tp
    fp_mask = ious < 0.3

    X_tp = X[tp_mask]
    X_fp = X[fp_mask]
    ious_tp = ious[tp_mask]
    ious_fp = ious[fp_mask]
    scores_tp = scores[tp_mask]
    scores_fp = scores[fp_mask]

    print(f"\nTP samples: {len(X_tp)}, FP samples: {len(X_fp)}")

    # 3. Global intrinsic dimension with skdim
    try:
        import skdim
        tle = skdim.id.TLE()
        if len(X_tp) > 50:
            id_tp = tle.fit_transform(X_tp)
            print(f"\nGlobal intrinsic dimension — TP: {id_tp:.2f}")
        if len(X_fp) > 50:
            id_fp = tle.fit_transform(X_fp)
            print(f"Global intrinsic dimension — FP: {id_fp:.2f}")
        if len(X_tp) > 50 and len(X_fp) > 50:
            print(f"ID difference (FP - TP): {id_fp - id_tp:.2f}  (positive = FP more complex)")
    except Exception as e:
        print(f"skdim TLE failed: {e}")
        id_tp = id_fp = None

    # 4. Persistent homology H1 with ripser
    if len(X_tp) > 500 and len(X_fp) > 500:
        try:
            from ripser import ripser
            print("\nRunning persistent homology (ripser) on subsampled 500 points each...")
            idx_tp = np.random.choice(len(X_tp), 500, replace=False)
            idx_fp = np.random.choice(len(X_fp), 500, replace=False)
            dgm_tp = ripser(X_tp[idx_tp], maxdim=1)["dgms"]
            dgm_fp = ripser(X_fp[idx_fp], maxdim=1)["dgms"]

            # Count significant H1 holes (birth < death, persistence > 0.1 * max death)
            def count_holes(dgm, thresh_ratio=0.1):
                if len(dgm) < 2 or len(dgm[1]) == 0:
                    return 0, []
                h1 = dgm[1]
                finite = h1[h1[:, 1] < np.inf]
                if len(finite) == 0:
                    return 0, []
                max_death = finite[:, 1].max()
                thresh = max_death * thresh_ratio
                pers = finite[:, 1] - finite[:, 0]
                significant = pers > thresh
                return significant.sum(), pers.tolist()

            n_tp, pers_tp = count_holes(dgm_tp)
            n_fp, pers_fp = count_holes(dgm_fp)
            print(f"H1 holes — TP: {n_tp}, FP: {n_fp}")
            print(f"Mean persistence — TP: {np.mean(pers_tp):.4f}, FP: {np.mean(pers_fp):.4f}")
        except Exception as e:
            print(f"ripser failed: {e}")
    else:
        print(f"\nSkipping ripser: TP={len(X_tp)}, FP={len(X_fp)} (need >500 each)")

    # 5. Local intrinsic dimension (k=30) and correlation
    k = 30
    if len(X) > k + 10:
        try:
            import skdim
            tle_local = skdim.id.TLE()
            # skdim can do local ID via fit_pw
            local_id = tle_local.fit_transform_pw(X, n_neighbors=k)

            # Correlations
            corr_iou_pearson, p_iou = pearsonr(local_id, ious)
            corr_iou_spearman, _ = spearmanr(local_id, ious)
            corr_conf_pearson, p_conf = pearsonr(local_id, scores)
            corr_conf_spearman, _ = spearmanr(local_id, scores)

            print(f"\nLocal ID ~ IoU correlation:")
            print(f"  Pearson r = {corr_iou_pearson:.4f} (p={p_iou:.4e})")
            print(f"  Spearman rho = {corr_iou_spearman:.4f}")

            print(f"\nLocal ID ~ Confidence correlation:")
            print(f"  Pearson r = {corr_conf_pearson:.4f} (p={p_conf:.4e})")
            print(f"  Spearman rho = {corr_conf_spearman:.4f}")

            # Also check within TP and FP separately
            if len(X_tp) > k + 10:
                corr_tp_iou, p_tp_iou = pearsonr(local_id[tp_mask], ious[tp_mask])
                corr_tp_conf, p_tp_conf = pearsonr(local_id[tp_mask], scores[tp_mask])
                print(f"\nWithin TP group:")
                print(f"  Local ID ~ IoU: r={corr_tp_iou:.4f} (p={p_tp_iou:.4e})")
                print(f"  Local ID ~ Conf: r={corr_tp_conf:.4f} (p={p_tp_conf:.4e})")

            if len(X_fp) > k + 10:
                corr_fp_iou, p_fp_iou = pearsonr(local_id[fp_mask], ious[fp_mask])
                corr_fp_conf, p_fp_conf = pearsonr(local_id[fp_mask], scores[fp_mask])
                print(f"\nWithin FP group:")
                print(f"  Local ID ~ IoU: r={corr_fp_iou:.4f} (p={p_fp_iou:.4e})")
                print(f"  Local ID ~ Conf: r={corr_fp_conf:.4f} (p={p_fp_conf:.4e})")

        except Exception as e:
            print(f"Local ID computation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipping local ID: total samples {len(X)} (need >{k+10})")

    # 6. Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if id_tp is not None and id_fp is not None:
        if id_tp < id_fp - 1.0:
            print("TP intrinsic dimension is LOWER than FP — structured signal detected.")
            usable = True
        elif id_tp > id_fp + 1.0:
            print("TP intrinsic dimension is HIGHER than FP — unexpected.")
            usable = False
        else:
            print("TP and FP intrinsic dimensions are similar — no geometric separation.")
            usable = False
    else:
        print("Intrinsic dimension estimation failed.")
        usable = False

    print(f"\nFinal verdict: {'USABLE' if usable else 'NOT USABLE'} as auxiliary RLVR reward dimension.")


if __name__ == "__main__":
    main()
