"""Analyze false negatives: which GT boxes the model misses, and why."""
import sys, json
from pathlib import Path
import torch, numpy as np
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda"
CKPT = "runs/round280_det_only_unf_s42/eval_metrics.json"
MODEL_CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SCORE_THRESH = 0.05
IOU_THRESH = 0.5

set_seed(42)

# Build model and load checkpoint
model = build_detector({
    "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
              "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
              "pretrained": True, "num_classes": 2,
              "min_size": 320, "max_size": 320}
}).to(DEV)
model.eval()

# We want the trained model, not random init
# Load round227 baseline first, then we'd need the fine-tuned weights
# For now, use the baseline checkpoint directly
ckpt = torch.load(MODEL_CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])

# Also try loading the round280 det_only_unf checkpoint if it exists
det_unf_ckpt = Path("runs/round280_det_only_unf_s42/checkpoint_best.pth")
if det_unf_ckpt.exists():
    print("Loading round280 det_only_unf checkpoint...")
    ckpt2 = torch.load(det_unf_ckpt, map_location=DEV)
    model.load_state_dict(ckpt2["model"])

# Build data loaders — use the same loader builder as training
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
tl, vl = build_penn_fudan_loaders({
    "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
    "train": {"batch_size": 1}
})
# Get the val dataset from the loader
val_dataset = vl.dataset

@torch.no_grad()
def analyze():
    results = []
    all_fns = []

    for idx, (img, target) in enumerate(tqdm(vl, desc="Analyzing")):
        # img is a list of one tensor, target is a list of one dict
        img_t = img[0].to(DEV) if isinstance(img, list) else img.to(DEV)
        if isinstance(target, list):
            target = target[0]
        img_h, img_w = img_t.shape[-2], img_t.shape[-1]
        gt_boxes = target["boxes"]
        gt_labels = target.get("labels", torch.ones(len(gt_boxes)))

        pred = model([img_t])[0]
        pred_boxes = pred["boxes"].cpu()
        pred_scores = pred["scores"].cpu()
        pred_labels = pred["labels"].cpu()

        # Filter by score
        keep = pred_scores >= SCORE_THRESH
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # Match predictions to GT
        matched_gt = set()
        matched_pred = set()
        fp_boxes = []
        fn_boxes = []

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            # For each GT, find best matching prediction
            for gi in range(len(gt_boxes)):
                if ious[:, gi].numel() > 0:
                    best_iou, best_pi = ious[:, gi].max(0)
                    if best_iou >= IOU_THRESH:
                        matched_gt.add(gi)
                        matched_pred.add(best_pi.item())

        # False negatives: GT boxes not matched
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                box = gt_boxes[gi]
                # Find the closest prediction even if below threshold
                closest_iou = 0.0
                closest_score = 0.0
                closest_box = None
                if len(pred_boxes) > 0:
                    ious_to_gt = box_iou(pred_boxes, box.unsqueeze(0))
                    if ious_to_gt.numel() > 0:
                        best_idx = ious_to_gt.argmax().item()
                        closest_iou = ious_to_gt[best_idx].item()
                        closest_score = pred_scores[best_idx].item()
                        closest_box = pred_boxes[best_idx]

                w = (box[2] - box[0]).item()
                h = (box[3] - box[1]).item()
                area = w * h
                cx = ((box[0] + box[2]) / 2).item()
                cy = ((box[1] + box[3]) / 2).item()

                fn_info = {
                    "image_idx": idx,
                    "gt_box": box.tolist(),
                    "width": w, "height": h, "area": area,
                    "cx": cx, "cy": cy,
                    "aspect_ratio": w / max(h, 1),
                    "edge_dist_left": cx,
                    "edge_dist_right": img_w - cx,
                    "edge_dist_top": cy,
                    "edge_dist_bottom": img_h - cy,
                    "closest_pred_iou": closest_iou,
                    "closest_pred_score": closest_score,
                    "closest_pred_box": closest_box.tolist() if closest_box is not None else None,
                    "num_gt_in_image": len(gt_boxes),
                    "num_pred_in_image": len(pred_boxes),
                }
                fn_boxes.append(fn_info)
                all_fns.append(fn_info)

        # False positives: predictions not matched to any GT
        for pi in range(len(pred_boxes)):
            if pi not in matched_pred:
                fp_boxes.append({
                    "box": pred_boxes[pi].tolist(),
                    "score": pred_scores[pi].item(),
                })

        results.append({
            "image_idx": idx,
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_boxes),
            "num_fn": len(fn_boxes),
            "num_fp": len(fp_boxes),
            "fn_boxes": fn_boxes,
            "fp_boxes": fp_boxes,
            "pred_scores": pred_scores.tolist(),
        })

    return results, all_fns

results, all_fns = analyze()

# Summary statistics
total_gt = sum(r["num_gt"] for r in results)
total_fn = sum(r["num_fn"] for r in results)
total_fp = sum(r["num_fp"] for r in results)
total_pred = sum(r["num_pred"] for r in results)
images_with_fn = sum(1 for r in results if r["num_fn"] > 0)
total_images = len(results)

print(f"\n=== False Negative Summary ===")
print(f"Total images: {total_images}")
print(f"Total GT boxes: {total_gt}")
print(f"Total predictions: {total_pred}")
print(f"False negatives: {total_fn} ({total_fn/total_gt*100:.1f}% of GT)")
print(f"False positives: {total_fp}")
print(f"Images with FN: {images_with_fn}/{total_images} ({images_with_fn/total_images*100:.1f}%)")
print(f"Recall (box-level): {(total_gt - total_fn)/total_gt*100:.1f}%")

# Analyze FN characteristics
if all_fns:
    fns = all_fns
    areas = [f["area"] for f in fns]
    widths = [f["width"] for f in fns]
    heights = [f["height"] for f in fns]
    aspects = [f["aspect_ratio"] for f in fns]
    closest_ious = [f["closest_pred_iou"] for f in fns]
    closest_scores = [f["closest_pred_score"] for f in fns]
    edge_dists = [min(f["edge_dist_left"], f["edge_dist_right"],
                      f["edge_dist_top"], f["edge_dist_bottom"]) for f in fns]

    print(f"\n=== FN Box Characteristics ===")
    print(f"Area:       min={min(areas):.0f}  max={max(areas):.0f}  mean={np.mean(areas):.0f}  median={np.median(areas):.0f}")
    print(f"Width:      min={min(widths):.1f}  max={max(widths):.1f}  mean={np.mean(widths):.1f}")
    print(f"Height:     min={min(heights):.1f}  max={max(heights):.1f}  mean={np.mean(heights):.1f}")
    print(f"AspectRatio: min={min(aspects):.2f}  max={max(aspects):.2f}  mean={np.mean(aspects):.2f}")
    print(f"ClosestIoU:  min={min(closest_ious):.3f}  max={max(closest_ious):.3f}  mean={np.mean(closest_ious):.3f}  median={np.median(closest_ious):.3f}")
    print(f"ClosestScore: min={min(closest_scores):.4f}  max={max(closest_scores):.4f}  mean={np.mean(closest_scores):.4f}")
    print(f"EdgeDist:    min={min(edge_dists):.1f}  max={max(edge_dists):.1f}  mean={np.mean(edge_dists):.1f}")

    # Categorize FN by likely cause
    print(f"\n=== FN by likely cause ===")
    no_pred = sum(1 for f in fns if f["closest_pred_iou"] < 0.1)  # no nearby prediction at all
    low_iou = sum(1 for f in fns if 0.1 <= f["closest_pred_iou"] < 0.5)  # prediction nearby but misaligned
    low_score = sum(1 for f in fns if f["closest_pred_iou"] >= 0.5 and f["closest_pred_score"] < 0.05)  # good box but low score (shouldn't happen much)
    confusing = sum(1 for f in fns if f["closest_pred_iou"] >= 0.5 and f["closest_pred_score"] >= 0.05)  # should have been detected but wasn't matched

    print(f"  No prediction nearby (IoU<0.1):     {no_pred} ({no_pred/len(fns)*100:.0f}%)")
    print(f"  Prediction nearby but misaligned:    {low_iou} ({low_iou/len(fns)*100:.0f}%)")
    print(f"  Good box, score too low (<0.05):     {low_score} ({low_score/len(fns)*100:.0f}%)")
    print(f"  Should be detected (IoU>=0.5):       {confusing} ({confusing/len(fns)*100:.0f}%)")

    # FN by image position
    print(f"\n=== FN by position ===")
    left = sum(1 for f in fns if f["cx"] < 80)
    right = sum(1 for f in fns if f["cx"] > 240)
    top = sum(1 for f in fns if f["cy"] < 80)
    bottom = sum(1 for f in fns if f["cy"] > 240)
    center = sum(1 for f in fns if 80 <= f["cx"] <= 240 and 80 <= f["cy"] <= 240)
    print(f"  Left edge (<80px):   {left} ({left/len(fns)*100:.0f}%)")
    print(f"  Right edge (>240px): {right} ({right/len(fns)*100:.0f}%)")
    print(f"  Top edge (<80px):    {top} ({top/len(fns)*100:.0f}%)")
    print(f"  Bottom edge (>240px):{bottom} ({bottom/len(fns)*100:.0f}%)")
    print(f"  Center region:       {center} ({center/len(fns)*100:.0f}%)")

    # Detailed: list top FN cases (worst misses)
    fns_sorted = sorted(fns, key=lambda f: f["area"], reverse=True)  # largest missed boxes
    print(f"\n=== Top-10 Largest Missed Boxes ===")
    print(f"{'Img':>4s} {'Area':>7s} {'W':>5s} {'H':>5s} {'AR':>5s} {'ClsIoU':>7s} {'ClsSc':>7s} {'#GT':>4s} {'#Pred':>5s}")
    for f in fns_sorted[:10]:
        print(f"{f['image_idx']:4d} {f['area']:7.0f} {f['width']:5.0f} {f['height']:5.0f} {f['aspect_ratio']:5.2f} {f['closest_pred_iou']:7.3f} {f['closest_pred_score']:7.4f} {f['num_gt_in_image']:4d} {f['num_pred_in_image']:5d}")

    # Bottom cases: smallest misses that DID have a good prediction
    hard_borderline = [f for f in fns if 0.4 <= f["closest_pred_iou"] < 0.5]
    if hard_borderline:
        print(f"\n=== Hard borderline (IoU 0.4-0.5): {len(hard_borderline)} cases ===")
        print(f"  These are boxes that almost matched but just missed the 0.5 threshold")
        for f in sorted(hard_borderline, key=lambda x: -x["area"])[:5]:
            print(f"  Img{f['image_idx']}: area={f['area']:.0f} IoU={f['closest_pred_iou']:.3f} score={f['closest_pred_score']:.4f}")

    # Images with most FN
    img_fn_counts = {}
    for f in fns:
        img_fn_counts[f["image_idx"]] = img_fn_counts.get(f["image_idx"], 0) + 1
    worst_imgs = sorted(img_fn_counts.items(), key=lambda x: -x[1])
    print(f"\n=== Images with most FN ===")
    for img_idx, count in worst_imgs[:5]:
        r = results[img_idx]
        print(f"  Image {img_idx}: {count} FN / {r['num_gt']} GT, {r['num_pred']} preds")

# Save FN details for further inspection
Path("runs/fn_analysis.json").write_text(json.dumps({
    "summary": {
        "total_gt": total_gt,
        "total_fn": total_fn,
        "fn_rate": total_fn/total_gt,
        "total_fp": total_fp,
        "images_with_fn": images_with_fn,
        "total_images": total_images,
    },
    "all_fn_boxes": all_fns,
}, indent=2))
print(f"\nFull FN data saved to runs/fn_analysis.json")
