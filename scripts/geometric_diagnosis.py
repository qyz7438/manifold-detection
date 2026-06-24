import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.spectral.fft_features import compute_fft_amplitude
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import load_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def extract_roi_from_image(image, box, target_size=(64, 64)):
    """Extract ROI from image tensor [C, H, W] given box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box.int().tolist()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[2], x2)
    y2 = min(image.shape[1], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image[:, y1:y2, x1:x2]
    roi = F.interpolate(roi.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    return roi


def flatten_fft_amplitude(roi, target_size=(64, 64)):
    """Compute and flatten FFT amplitude spectrum to fixed dimension."""
    amp = compute_fft_amplitude(roi, use_hann=True)
    # Resize to fixed size if needed
    if amp.shape != target_size:
        amp = F.interpolate(amp.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze().squeeze()
    return amp.flatten().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='spectral_detection_posttrain/configs/baseline.yaml')
    parser.add_argument('--checkpoint', default='runs/mvp_pf_baseline/checkpoint_last.pth')
    parser.add_argument('--output', default='runs/geometric_diagnosis.json')
    parser.add_argument('--limit-val', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--roi-size', type=int, default=64)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = resolve_device(config)

    _, val_loader = build_penn_fudan_loaders(
        config,
        limit_train=1,
        limit_val=args.limit_val,
        batch_size=int(config['eval'].get('batch_size', 2)),
    )

    model_cfg = dict(config)
    model_cfg['model'] = dict(config['model'])
    model_cfg['model']['pretrained'] = False
    model = build_detector(model_cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    target_size = (args.roi_size, args.roi_size)
    fft_dim = args.roi_size * args.roi_size

    tp_features = []
    fp_features = []
    tp_ious = []
    fp_ious = []
    tp_scores = []
    fp_scores = []

    all_local_dims = []
    all_ious = []
    all_scores = []

    print(f"Extracting proposals from {len(val_loader)} val batches...")

    with torch.no_grad():
        for images, batch_targets in tqdm(val_loader, desc="Val images"):
            outputs = model([img.to(device) for img in images])
            for img, output, target in zip(images, outputs, batch_targets):
                img_cpu = img.cpu()
                pred = {k: v.detach().cpu() for k, v in output.items()}
                tgt = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}

                match_result = match_predictions_to_gt(
                    pred, tgt,
                    iou_threshold=0.5,
                    score_threshold=0.05,
                )
                matched_pred_indices = {m['pred_index']: m['iou'] for m in match_result['matches']}
                unmatched_pred_indices = set(match_result['unmatched_predictions'])

                boxes = pred.get('boxes', torch.empty((0, 4)))
                scores = pred.get('scores', torch.empty((0,)))

                for i in range(len(boxes)):
                    if i in matched_pred_indices:
                        iou = matched_pred_indices[i]
                        if iou > 0.5:
                            roi = extract_roi_from_image(img_cpu, boxes[i], target_size)
                            if roi is not None:
                                feat = flatten_fft_amplitude(roi, target_size)
                                tp_features.append(feat)
                                tp_ious.append(iou)
                                tp_scores.append(float(scores[i].item()))
                    elif i in unmatched_pred_indices:
                        # Check max IoU with any GT to confirm it's truly low
                        from spectral_detection_posttrain.matching.box_iou import box_iou
                        gt_boxes = tgt.get('boxes', torch.empty((0, 4)))
                        if len(gt_boxes) > 0:
                            ious = box_iou(boxes[i:i+1], gt_boxes)
                            max_iou = ious.max().item()
                        else:
                            max_iou = 0.0
                        if max_iou < 0.3:
                            roi = extract_roi_from_image(img_cpu, boxes[i], target_size)
                            if roi is not None:
                                feat = flatten_fft_amplitude(roi, target_size)
                                fp_features.append(feat)
                                fp_ious.append(max_iou)
                                fp_scores.append(float(scores[i].item()))

    print(f"TP count: {len(tp_features)}, FP count: {len(fp_features)}")

    if len(tp_features) == 0 or len(fp_features) == 0:
        print("Insufficient samples for analysis.")
        return

    tp_features = np.array(tp_features)
    fp_features = np.array(fp_features)

    # Save features for potential reuse
    np.savez('runs/geometric_features.npz', tp=tp_features, fp=fp_features)

    # Intrinsic dimension estimation using skdim
    try:
        import skdim.id
        tle = skdim.id.TLE()
        tp_id = tle.fit_transform(tp_features)
        fp_id = tle.fit_transform(fp_features)
        print(f"\n=== Global Intrinsic Dimension (TLE) ===")
        print(f"TP intrinsic dimension: {tp_id:.2f}")
        print(f"FP intrinsic dimension: {fp_id:.2f}")
        print(f"Difference (FP - TP): {fp_id - tp_id:.2f}")
    except Exception as e:
        print(f"skdim TLE failed: {e}")
        tp_id = fp_id = None

    # Persistent homology with ripser
    h1_results = {}
    if len(tp_features) > 500 and len(fp_features) > 500:
        try:
            from ripser import ripser
            print(f"\n=== Persistent Homology H1 ===")
            # Subsample to manageable size
            tp_sub = tp_features[np.random.choice(len(tp_features), 500, replace=False)]
            fp_sub = fp_features[np.random.choice(len(fp_features), 500, replace=False)]

            tp_dgm = ripser(tp_sub, maxdim=1)['dgms']
            fp_dgm = ripser(fp_sub, maxdim=1)['dgms']

            tp_h1 = tp_dgm[1] if len(tp_dgm) > 1 else np.array([])
            fp_h1 = fp_dgm[1] if len(fp_dgm) > 1 else np.array([])

            print(f"TP H1 holes: {len(tp_h1)}")
            print(f"FP H1 holes: {len(fp_h1)}")
            if len(tp_h1) > 0:
                print(f"TP H1 persistence mean: {np.mean(tp_h1[:,1] - tp_h1[:,0]):.4f}")
            if len(fp_h1) > 0:
                print(f"FP H1 persistence mean: {np.mean(fp_h1[:,1] - fp_h1[:,0]):.4f}")

            h1_results = {
                'tp_h1_count': int(len(tp_h1)),
                'fp_h1_count': int(len(fp_h1)),
                'tp_h1_persistence_mean': float(np.mean(tp_h1[:,1] - tp_h1[:,0])) if len(tp_h1) > 0 else 0.0,
                'fp_h1_persistence_mean': float(np.mean(fp_h1[:,1] - fp_h1[:,0])) if len(fp_h1) > 0 else 0.0,
            }
        except Exception as e:
            print(f"Ripser failed: {e}")
    else:
        print(f"\n=== Persistent Homology H1 ===")
        print(f"Skipped: TP={len(tp_features)}, FP={len(fp_features)} (need >500 each)")

    # Per-proposal local intrinsic dimension
    print(f"\n=== Per-Proposal Local Intrinsic Dimension ===")
    all_features = np.vstack([tp_features, fp_features])
    all_ious = np.array(tp_ious + fp_ious)
    all_scores = np.array(tp_scores + fp_scores)

    try:
        import skdim.id
        from sklearn.neighbors import NearestNeighbors

        k = 30
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(all_features)
        distances, indices = nbrs.kneighbors(all_features)

        local_dims = []
        tle_local = skdim.id.TLE()
        for i in tqdm(range(len(all_features)), desc="Local dim"):
            nn_indices = indices[i][1:]  # exclude self
            nn_features = all_features[nn_indices]
            ld = tle_local.fit_transform(nn_features)
            local_dims.append(ld)

        local_dims = np.array(local_dims)

        # Correlations
        corr_iou_pearson, p_iou = pearsonr(local_dims, all_ious)
        corr_iou_spearman, _ = spearmanr(local_dims, all_ious)
        corr_score_pearson, p_score = pearsonr(local_dims, all_scores)
        corr_score_spearman, _ = spearmanr(local_dims, all_scores)

        print(f"Local dim ~ IoU Pearson r={corr_iou_pearson:.4f} (p={p_iou:.4f})")
        print(f"Local dim ~ IoU Spearman rho={corr_iou_spearman:.4f}")
        print(f"Local dim ~ Score Pearson r={corr_score_pearson:.4f} (p={p_score:.4f})")
        print(f"Local dim ~ Score Spearman rho={corr_score_spearman:.4f}")

        # TP vs FP local dim
        tp_local = local_dims[:len(tp_features)]
        fp_local = local_dims[len(tp_features):]
        print(f"TP local dim mean: {np.mean(tp_local):.2f} +/- {np.std(tp_local):.2f}")
        print(f"FP local dim mean: {np.mean(fp_local):.2f} +/- {np.std(fp_local):.2f}")

        local_results = {
            'local_dim_iou_pearson': float(corr_iou_pearson),
            'local_dim_iou_pvalue': float(p_iou),
            'local_dim_iou_spearman': float(corr_iou_spearman),
            'local_dim_score_pearson': float(corr_score_pearson),
            'local_dim_score_pvalue': float(p_score),
            'local_dim_score_spearman': float(corr_score_spearman),
            'tp_local_dim_mean': float(np.mean(tp_local)),
            'tp_local_dim_std': float(np.std(tp_local)),
            'fp_local_dim_mean': float(np.mean(fp_local)),
            'fp_local_dim_std': float(np.std(fp_local)),
        }
    except Exception as e:
        print(f"Local dimension analysis failed: {e}")
        local_results = {}

    # Save results
    results = {
        'tp_count': len(tp_features),
        'fp_count': len(fp_features),
        'fft_dim': fft_dim,
        'tp_intrinsic_dim': float(tp_id) if tp_id is not None else None,
        'fp_intrinsic_dim': float(fp_id) if fp_id is not None else None,
        'intrinsic_dim_diff': float(fp_id - tp_id) if (tp_id is not None and fp_id is not None) else None,
        'h1': h1_results,
        'local': local_results,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Conclusion
    print(f"\n=== CONCLUSION ===")
    usable = False
    if tp_id is not None and fp_id is not None:
        if fp_id - tp_id > 1.0:
            print(f"FP intrinsic dimension is significantly higher than TP by {fp_id - tp_id:.2f}.")
            print("This suggests geometric structure exists and could be used as auxiliary signal.")
            usable = True
        else:
            print(f"FP-TP dimension gap is small ({fp_id - tp_id:.2f}), geometric signal is weak.")
    if local_results:
        if abs(corr_iou_pearson) > 0.1 and p_iou < 0.05:
            print(f"Local dimension correlates with IoU (r={corr_iou_pearson:.4f}), usable as auxiliary.")
            usable = True
        else:
            print(f"Local dimension ~ IoU correlation is weak (r={corr_iou_pearson:.4f}), not a strong auxiliary signal.")
    print(f"\nVerdict: {'USABLE as auxiliary RLVR reward dimension' if usable else 'NOT USABLE as auxiliary RLVR reward dimension'}")


if __name__ == '__main__':
    main()
