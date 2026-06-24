"""NWPU VHR-10: load COCO annotations, train baseline, check proposal IoU distribution."""
import sys, json
import torch, numpy as np
from pathlib import Path
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
set_seed(42)

# Load COCO annotations
coco = json.loads(ANNOT.read_text())
categories = {c["id"]: c["name"] for c in coco["categories"]}
images = {img["id"]: img for img in coco["images"]}
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

print(f"Categories: {categories}")
print(f"Images: {len(images)}, Annotations: {len(coco['annotations'])}")
# COCO annotations don't have split field, use custom split
all_img_ids = list(images.keys())
np.random.seed(42); np.random.shuffle(all_img_ids)
n_train = int(0.7 * len(all_img_ids))
train_ids = set(all_img_ids[:n_train])
val_ids = set(all_img_ids[n_train:])
print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")

# Check GT box sizes
boxes_all = []
for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    boxes_all.append({"area": w*h, "w": w, "h": h, "aspect": w/max(h,1)})

areas = [b["area"] for b in boxes_all]
widths = [b["w"] for b in boxes_all]
heights = [b["h"] for b in boxes_all]
aspects = [b["aspect"] for b in boxes_all]

print(f"\nGT box stats:")
print(f"  Area: min={min(areas):.0f} max={max(areas):.0f} mean={np.mean(areas):.0f} median={np.median(areas):.0f}")
print(f"  Width: min={min(widths):.0f} max={max(widths):.0f} mean={np.mean(widths):.0f}")
print(f"  Height: min={min(heights):.0f} max={max(heights):.0f} mean={np.mean(heights):.0f}")
print(f"  Small (<32px): {sum(1 for a in areas if a<1024)}")
print(f"  Medium (32-96px): {sum(1 for a in areas if 1024<=a<9216)}")
print(f"  Large (>96px): {sum(1 for a in areas if a>=9216)}")

# Check images per image
gts_per_img = [len(anns) for anns in anns_by_image.values()]
print(f"\nGT per image: min={min(gts_per_img)} max={max(gts_per_img)} mean={np.mean(gts_per_img):.1f}")
print(f"  Dense (>10 GT): {sum(1 for g in gts_per_img if g>10)}/{len(gts_per_img)}")

# Quick model test: load pretrained Faster R-CNN on one image, check proposals
from spectral_detection_posttrain.models import build_detector
from torchvision.ops import box_iou
from PIL import Image
from torchvision.transforms import functional as TF
import torch.nn.functional as F

model = build_detector({"model":{"name":"fasterrcnn_mobilenet_v3_large_320_fpn","model_name":"fasterrcnn_mobilenet_v3_large_320_fpn","pretrained":True,"num_classes":len(categories)+1,"min_size":480,"max_size":480}}).to(DEV)
model.eval()

# Sample a few images, check prediction IoU distribution
sample_imgs = [img for img in coco["images"] if img["id"] in train_ids][:5]
all_proposal_ious = []

for img_info in sample_imgs:
    img_path = DATA / "positive image set" / img_info["file_name"]
    if not img_path.exists():
        img_path = DATA / "negative image set" / img_info["file_name"]
    if not img_path.exists(): continue

    from PIL import Image
    pil_img = Image.open(str(img_path)).convert("RGB")
    img = TF.to_tensor(pil_img).to(DEV)

    # Resize
    _, H, W = img.shape
    if max(H, W) > 480:
        scale = 480 / max(H, W)
        img = F.interpolate(img.unsqueeze(0), size=(int(H*scale), int(W*scale)), mode="bilinear").squeeze(0)

    gt_boxes = []
    for ann in anns_by_image.get(img_info["id"], []):
        x, y, w, h = ann["bbox"]
        if max(H, W) > 480:
            scale = 480 / max(H, W)
            x, y, w, h = x*scale, y*scale, w*scale, h*scale
        gt_boxes.append([x, y, x+w, y+h])

    if not gt_boxes: continue
    gt_boxes = torch.tensor(gt_boxes, device=DEV)

    # Run RPN only (use model with GT targets to trigger RPN)
    with torch.no_grad():
        model.train()  # need train mode for proposals
        images, targets = [img], [{"boxes": gt_boxes, "labels": torch.ones(len(gt_boxes), dtype=torch.int64, device=DEV)}]
        _ = model(images, targets)

    # Get RPN proposals from the box_roi_pool hook
    # (Proposals are sampled before box_roi_pool - need a different hook)
    # Instead, let's just run the full forward and check
    model.eval()
    with torch.no_grad():
        pred = model([img])[0]

    pred_boxes = pred["boxes"]
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        ious = box_iou(pred_boxes, gt_boxes)
        best_ious = ious.max(dim=1).values.cpu().numpy()
        all_proposal_ious.extend(best_ious.tolist())

if all_proposal_ious:
    ious = np.array(all_proposal_ious)
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print(f"\nPrediction IoU distribution ({len(ious)} predictions on {len(sample_imgs)} images):")
    for lo, hi in bins:
        cnt = sum(1 for i in ious if lo <= i < hi)
        print(f"  [{lo:.1f}, {hi:.1f}): {cnt:4d} ({100*cnt/len(ious):5.1f}%)")

    borderline = sum(1 for i in ious if 0.3 <= i < 0.7)
    print(f"\n  Borderline (0.3-0.7): {borderline}/{len(ious)} ({100*borderline/len(ious):.1f}%)")
    print(f"  vs Penn-Fudan: 0/150 (0.0%)")
