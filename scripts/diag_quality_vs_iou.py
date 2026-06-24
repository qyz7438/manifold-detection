"""Diagnostic: spectral quality rank vs IoU rank for 8 sampled deltas."""
import sys, torch, torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
M_SAMPLES = 8
PIXEL_SIZE = 64
MAX_PROPOSALS = 200

set_seed(42)


def pixel_fft_quality(pixel_patches):
    gray = pixel_patches.mean(dim=1)
    fft = torch.fft.fft2(gray.float()).abs()
    mag_flat = fft.flatten(1)
    total = mag_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf = mag_flat[:, mag_flat.shape[1] // 2:].sum(dim=1) / total.squeeze(1)
    mag_norm = mag_flat / total
    entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum(dim=1)
    max_e = torch.log(torch.tensor(float(mag_flat.shape[1]), device=pixel_patches.device))
    e_norm = 1.0 - entropy / max_e
    pha_var = torch.angle(torch.fft.fft2(gray.float()) + 1e-6).flatten(1).std(dim=1).clamp_max(1.0)
    quality = 0.3 * hf + 0.4 * e_norm + 0.3 * (1.0 - pha_var)
    return quality.clamp(0.0, 1.0)


def decode_boxes(proposals, deltas):
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights
    ref = torch.zeros_like(deltas)
    ref[:, 0] = deltas[:, 0] * widths + ctr_x - 0.5 * torch.exp(deltas[:, 2]) * widths
    ref[:, 1] = deltas[:, 1] * heights + ctr_y - 0.5 * torch.exp(deltas[:, 3]) * heights
    ref[:, 2] = deltas[:, 0] * widths + ctr_x + 0.5 * torch.exp(deltas[:, 2]) * widths
    ref[:, 3] = deltas[:, 1] * heights + ctr_y + 0.5 * torch.exp(deltas[:, 3]) * heights
    return ref.clamp(min=0)


cfg = {"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                 "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                 "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
model = build_detector(cfg).to(DEV)
ckpt = torch.load(CKPT, map_location=DEV)
model.load_state_dict(ckpt["model"])
model.eval()

loaders = build_penn_fudan_loaders({
    "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
    "train": {"batch_size": 2},
})

all_correct = 0
all_total = 0
all_q_spread = []
all_iou_spread = []
all_quality = []
all_iou = []

@torch.no_grad()
def main():
    global all_correct, all_total
    proposal_cache = {}
    roi_cache = {}

    def rpn_hook(m, i, o): proposal_cache["p"] = o[0]
    def roi_hook(m, i): roi_cache["x"] = i[0]
    hk_rpn = model.rpn.register_forward_hook(rpn_hook)
    hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)

    images_processed = 0
    for images, targets in tqdm(loaders[0]):
        images_dev = [img.to(DEV) for img in images]
        targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
        proposal_cache.clear(); roi_cache.clear()

        model(images_dev, targets_t)

        roi_feats = roi_cache.get("x")
        proposals = proposal_cache.get("p")
        if roi_feats is None or proposals is None or roi_feats.shape[0] == 0:
            continue

        N = roi_feats.shape[0]
        box_ft = model.roi_heads.box_head(roi_feats)
        mu = model.roi_heads.box_predictor.bbox_pred(box_ft)[:, -4:]

        proposals_cat = torch.cat(proposals, dim=0)
        N = min(N, proposals_cat.shape[0], MAX_PROPOSALS)
        mu = mu[:N]
        sigma = torch.full_like(mu, 0.1)

        eps = torch.randn(N, M_SAMPLES, 4, device=DEV)
        deltas = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps

        all_deltas = deltas.reshape(N * M_SAMPLES, 4)
        props_exp = proposals_cat[:N].unsqueeze(1).expand(-1, M_SAMPLES, -1).reshape(N * M_SAMPLES, 4)
        all_boxes = decode_boxes(props_exp, all_deltas)

        # Pixel FFT quality
        pixel_patches = []
        for bi in range(N * M_SAMPLES):
            img_i = bi // (M_SAMPLES * N // len(images))
            img_i = min(img_i, len(images) - 1)
            img = images[img_i]
            box = all_boxes[bi]
            x1, y1, x2, y2 = box.round().long().clamp(min=0)
            x1, x2 = max(0, min(x1, img.shape[-1]-1)), max(x1+1, min(x2, img.shape[-1]))
            y1, y2 = max(0, min(y1, img.shape[-2]-1)), max(y1+1, min(y2, img.shape[-2]))
            patch = img[:, y1:y2, x1:x2]
            if patch.shape[-1] >= 4 and patch.shape[-2] >= 4:
                patch = F.interpolate(patch.float().unsqueeze(0), size=(PIXEL_SIZE, PIXEL_SIZE),
                                      mode='bilinear', align_corners=False).squeeze(0)
                pixel_patches.append(patch)
            else:
                pixel_patches.append(torch.zeros(3, PIXEL_SIZE, PIXEL_SIZE))

        patch_batch = torch.stack(pixel_patches).to(DEV)
        qualities = pixel_fft_quality(patch_batch)
        q_matrix = qualities.view(N, M_SAMPLES)

        # IoU with GT (per proposal, best GT)
        prop_img_map = []
        n_before = 0
        for img_i, p in enumerate(proposals):
            for j in range(p.shape[0]):
                if n_before < N:
                    prop_img_map.append(img_i)
                n_before += 1
        prop_img_map = prop_img_map[:N]

        iou_matrix = torch.zeros(N, M_SAMPLES)
        for pi in range(N):
            img_i = prop_img_map[pi]
            gt_boxes = targets_t[img_i]["boxes"]
            sampled_boxes = all_boxes[pi * M_SAMPLES:(pi + 1) * M_SAMPLES]
            if len(gt_boxes) > 0:
                ious = box_iou(sampled_boxes, gt_boxes)
                iou_matrix[pi] = ious.max(dim=1).values

        # Per-proposal: quality rank vs IoU rank
        for pi in range(N):
            q_row = q_matrix[pi]  # (M,)
            i_row = iou_matrix[pi]  # (M,)

            q_top = q_row.argmax().item()
            i_top = i_row.argmax().item()
            if q_top == i_top:
                all_correct += 1
            all_total += 1

            q_spread = (q_row.max() - q_row.min()).item()
            i_spread = (i_row.max() - i_row.min()).item()
            all_q_spread.append(q_spread)
            all_iou_spread.append(i_spread)
            all_quality.extend(q_row.tolist())
            all_iou.extend(i_row.tolist())

        images_processed += len(images)
        if images_processed >= 10:
            break

    hk_rpn.remove(); hk_roi.remove()

    # Stats
    import numpy as np
    q_spread = np.array(all_q_spread)
    i_spread = np.array(all_iou_spread)
    all_q = np.array(all_quality)
    all_i = np.array(all_iou)

    print(f"\nTotal (N, M) pairs: {all_total}")
    print(f"Top-1 match rate (quality argmax == IoU argmax): {all_correct/all_total:.3f} "
          f"(random baseline = {1.0/M_SAMPLES:.3f})")
    print(f"\nQuality spread within 8 deltas:  mean={q_spread.mean():.5f}  std={q_spread.std():.5f}  max={q_spread.max():.5f}")
    print(f"IoU spread within 8 deltas:       mean={i_spread.mean():.5f}  std={i_spread.std():.5f}  max={i_spread.max():.5f}")
    print(f"\nQuality overall:  mean={all_q.mean():.4f}  std={all_q.std():.4f}  min={all_q.min():.4f}  max={all_q.max():.4f}")
    print(f"IoU overall:       mean={all_i.mean():.4f}  std={all_i.std():.4f}  min={all_i.min():.4f}  max={all_i.max():.4f}")

    # Correlation
    corr = np.corrcoef(all_q, all_i)[0, 1]
    print(f"\nQuality-IoU Pearson r: {corr:.4f}")

    # Per-IoU-bin analysis
    bins = [0, 0.2, 0.4, 0.6, 0.8]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (all_i >= lo) & (all_i < hi)
        if mask.sum() > 0:
            print(f"  IoU [{lo:.1f},{hi:.1f}): n={mask.sum():4d}  q_mean={all_q[mask].mean():.4f}  q_std={all_q[mask].std():.5f}")


if __name__ == "__main__":
    main()
