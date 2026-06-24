"""Plan 2.49: Learned Spectral Reward — CNN on complex FFT → predict IoU.

Architecture:
  FPN ROI Align (14×14) → FFT2D → stack |F|,Re,Im → CNN → quality_pred
  Target: matched IoU per proposal
  Loss: det_loss + α*(quality_pred * box_reg_loss) + β*MSE(quality_pred, IoU_target)

The CNN learns from IoU labels which frequency patterns indicate good localization.
"""
import sys, json, subprocess, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.ops import box_iou

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
ALPHAS = [0.5, 1.0]
BETA = 0.5  # IoU regression weight
EPOCHS = 10
ROI_SIZE = 14


class LearnedSpectralHead(nn.Module):
    """CNN on complex FFT features → per-ROI quality prediction (IoU regression)."""

    def __init__(self, roi_size=ROI_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, roi_features):
        N = roi_features.shape[0]
        fft = torch.fft.fft2(roi_features.float(), dim=(-2, -1), norm="ortho")
        mag = torch.abs(fft).mean(dim=1, keepdim=True)
        real = fft.real.mean(dim=1, keepdim=True)
        imag = fft.imag.mean(dim=1, keepdim=True)
        spectral = torch.cat([mag, real, imag], dim=1)
        q_map = self.net(spectral)
        quality = torch.sigmoid(q_map.flatten(1).mean(dim=1))
        return quality


def compute_iou_targets(proposals, targets_t):
    """Compute max IoU for each proposal against its image's GT boxes.

    proposals: List[Tensor(N_i, 4)] per image
    targets_t: List[dict] per image, each with "boxes" key
    Returns: Tensor(total_N,) of max IoU per proposal
    """
    iou_list = []
    for img_i, props in enumerate(proposals):
        Np = props.shape[0]
        gt_boxes = targets_t[img_i]["boxes"]
        if len(gt_boxes) > 0:
            ious = box_iou(props, gt_boxes)  # (Np, G)
            max_iou, _ = ious.max(dim=1)
            iou_list.append(max_iou)
        else:
            iou_list.append(torch.zeros(Np, device=DEV))
    if iou_list:
        return torch.cat(iou_list, dim=0)
    return torch.zeros(0, device=DEV)


def build_loaders():
    return build_penn_fudan_loaders({
        "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 2},
    })


def build_model():
    cfg = {"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
    return build_detector(cfg)


def freeze_except(model, trainable_parts):
    for p in model.parameters():
        p.requires_grad = False
    for part in trainable_parts:
        if isinstance(part, nn.Module):
            for p in part.parameters():
                p.requires_grad = True


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        out = model([img.to(DEV) for img in images])
        preds.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


def main():
    all_r = []

    for alpha in ALPHAS:
        run_name = f"round249_lsr_a{alpha}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        spectral_head = LearnedSpectralHead().to(DEV)
        freeze_except(model, [model.rpn.head, model.roi_heads.box_head,
                      model.roi_heads.box_predictor])

        train_loader, val_loader = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        params += list(spectral_head.parameters())
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0

        fpn_cache = {}
        proposal_cache = {}

        def fpn_hook(module, inp, out):
            fpn_cache["f"] = {k: out[k] for k in out if k != "pool"}

        def rpn_hook(module, inp, out):
            proposal_cache["p"] = out[0]

        hk_fpn = model.backbone.register_forward_hook(fpn_hook)
        hk_rpn = model.rpn.register_forward_hook(rpn_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            spectral_head.train()
            total_det, total_spec, total_iou_reg = 0.0, 0.0, 0.0
            avg_q, avg_iou = 0.0, 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images_dev = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                fpn_cache.clear()
                proposal_cache.clear()

                ld = model(images_dev, targets_t)
                det_loss = sum(ld.values())
                box_reg = ld.get("loss_box_reg", torch.tensor(0.0, device=DEV))

                fpn_feats = fpn_cache.get("f")
                proposals = proposal_cache.get("p")
                spec_loss = torch.tensor(0.0, device=DEV)
                iou_reg = torch.tensor(0.0, device=DEV)

                if fpn_feats is not None and proposals is not None:
                    fpn_keys = sorted(fpn_feats.keys(), key=int)
                    all_props = []
                    for p in proposals:
                        all_props.append(p)
                    if all_props:
                        prop_boxes = torch.cat(all_props, dim=0)[:256]
                        P = prop_boxes.shape[0]
                        if P > 0:
                            # FPN-level ROI Align
                            w = prop_boxes[:, 2] - prop_boxes[:, 0]
                            h = prop_boxes[:, 3] - prop_boxes[:, 1]
                            area = (w * h).clamp_min(1)
                            lvl = torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).long().clamp(2, 5)
                            roi14_list = []
                            for i in range(P):
                                ki = min(len(fpn_keys) - 1, max(0, lvl[i].item() - 2))
                                feat = fpn_feats[fpn_keys[ki]]
                                bx = prop_boxes[i:i + 1]
                                ri = torch.cat([torch.zeros(1, 1, device=DEV), bx], dim=1)
                                scale = 1.0 / (2 ** (int(fpn_keys[ki]) + 2))
                                r14 = torchvision.ops.roi_align(feat, ri, output_size=ROI_SIZE, spatial_scale=scale)
                                roi14_list.append(r14)

                            if roi14_list:
                                roi14 = torch.cat(roi14_list, dim=0)

                                # IoU targets for these proposals
                                iou_targets = compute_iou_targets(proposals, targets_t)
                                iou_targets = iou_targets[:P]

                                # Predict quality from spectral features
                                quality_pred = spectral_head(roi14)  # (P,)

                                # Auxiliary IoU regression loss
                                iou_reg = F.mse_loss(quality_pred, iou_targets)

                                # Quality-weighted bbox loss (like 2.31 but learned)
                                spec_loss = (quality_pred * box_reg).mean()

                                avg_q = quality_pred.mean().item()
                                avg_iou = iou_targets.mean().item()

                loss = det_loss + alpha * spec_loss + BETA * iou_reg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_spec += spec_loss.item()
                total_iou_reg += iou_reg.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} "
                  f"spec={total_spec:.3f} iou_r={total_iou_reg:.3f} q={avg_q:.4f} iou_t={avg_iou:.4f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_fpn.remove(); hk_rpn.remove()

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.49 Learned Spectral Reward Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
