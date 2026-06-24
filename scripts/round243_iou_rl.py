"""Plan 2.43: Verify IoU as RL reward is infeasible.

Freeze backbone+RPN → box_head outputs μ,σ → sample M deltas →
decode → IoU with matched GT → REINFORCE policy gradient.
"""
import sys, json, subprocess, math
from pathlib import Path
import torch
import torch.nn.functional as F
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
M_SAMPLES = 8
ALPHAS = [0.01, 0.1, 0.5]
EPOCHS = 5


def decode_boxes(proposals, deltas):
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights
    pred_ctr_x = deltas[:, 0] * widths + ctr_x
    pred_ctr_y = deltas[:, 1] * heights + ctr_y
    pred_w = torch.exp(deltas[:, 2]) * widths
    pred_h = torch.exp(deltas[:, 3]) * heights
    refined = torch.zeros_like(deltas)
    refined[:, 0] = pred_ctr_x - 0.5 * pred_w
    refined[:, 1] = pred_ctr_y - 0.5 * pred_h
    refined[:, 2] = pred_ctr_x + 0.5 * pred_w
    refined[:, 3] = pred_ctr_y + 0.5 * pred_h
    return refined.clamp(min=0)


def gaussian_log_prob(deltas, mu, sigma):
    eps = (deltas - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    return -0.5 * (eps.pow(2) + 2 * torch.log(sigma.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


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
        if isinstance(part, torch.nn.Module):
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
        run_name = f"round243_iourl_a{alpha}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        freeze_except(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])

        train_loader, val_loader = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0

        proposal_cache = {}
        roi_cache = {}
        matched_gt = {}

        def rpn_hook(module, inp, out):
            proposal_cache["p"] = out[0]

        def roi_hook(module, inp):
            roi_cache["x"] = inp[0]

        hk_rpn = model.rpn.register_forward_hook(rpn_hook)
        hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_det, total_rl = 0.0, 0.0
            avg_iou, avg_adv = 0.0, 0.0
            pos_count = 0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images_dev = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                proposal_cache.clear()
                roi_cache.clear()

                ld = model(images_dev, targets_t)
                det_loss = sum(ld.values())

                roi_feats = roi_cache.get("x")
                proposals = proposal_cache.get("p")
                rl_loss = torch.tensor(0.0, device=DEV)

                if roi_feats is not None and proposals is not None and roi_feats.shape[0] > 0:
                    N = roi_feats.shape[0]
                    box_ft = model.roi_heads.box_head(roi_feats)
                    mu = model.roi_heads.box_predictor.bbox_pred(box_ft)[:, -4:]  # (N, 4)
                    sigma_val = 0.1
                    sigma = torch.full_like(mu, sigma_val)

                    # Sample M deltas per proposal
                    eps = torch.randn(N, M_SAMPLES, 4, device=DEV)
                    deltas = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps  # (N, M, 4)

                    # Decode to image coordinates
                    proposals_cat = torch.cat(proposals, dim=0)
                    N_prop = min(N, proposals_cat.shape[0])
                    N = N_prop
                    mu = mu[:N]
                    deltas = deltas[:N]
                    sigma = sigma[:N]  # align sigma with truncated N

                    all_deltas = deltas.reshape(N * M_SAMPLES, 4)
                    props_expanded = proposals_cat[:N].unsqueeze(1).expand(-1, M_SAMPLES, -1).reshape(N * M_SAMPLES, 4)
                    all_boxes = decode_boxes(props_expanded, all_deltas)

                    # Match proposals to GT, compute IoU per delta
                    iou_matrix = torch.zeros(N, M_SAMPLES, device=DEV)
                    is_pos = torch.zeros(N, dtype=torch.bool, device=DEV)

                    for img_i, props_img in enumerate(proposals):
                        if props_img.shape[0] == 0:
                            continue
                        gt_boxes = targets_t[img_i]["boxes"]
                        if len(gt_boxes) == 0:
                            continue

                        # Find which proposals belong to this image
                        # proposals_cat is concatenated in order; find start/end for this image
                        n_before = sum(p.shape[0] for p in proposals[:img_i])
                        n_this = props_img.shape[0]
                        if n_before + n_this > N:
                            n_this = max(0, N - n_before)
                        if n_this <= 0:
                            continue

                        # IoU of each proposal's M decoded boxes with all GT boxes
                        for pi in range(n_before, n_before + n_this):
                            pi_local = pi - n_before
                            base_box = props_img[pi_local]
                            sampled_boxes = all_boxes[pi * M_SAMPLES:(pi + 1) * M_SAMPLES]
                            ious_with_gt = box_iou(sampled_boxes, gt_boxes)  # (M, G)
                            best_iou_per_delta, _ = ious_with_gt.max(dim=1)  # (M,)
                            iou_matrix[pi] = best_iou_per_delta
                            is_pos[pi] = best_iou_per_delta.max() > 0.3

                    # Compute log-probs (with grad)
                    log_probs = gaussian_log_prob(deltas, mu, sigma)  # (N, M)

                    pos_mask = is_pos
                    if pos_mask.sum() > 0:
                        iou_pos = iou_matrix[pos_mask]  # (P, M)
                        lp_pos = log_probs[pos_mask]  # (P, M)
                        P = iou_pos.shape[0]

                        # REINFORCE: advantage = IoU - baseline (mean IoU of all samples)
                        baseline = iou_pos.mean()  # batch-level baseline
                        advantage = iou_pos - baseline  # (P, M)
                        rl_loss = -(advantage.detach() * lp_pos).mean()

                        avg_iou = iou_pos.mean().item()
                        avg_adv = advantage.abs().mean().item()
                        pos_count = int(pos_mask.sum())

                loss = det_loss + alpha * rl_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_rl += rl_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} rl={total_rl:.3f} "
                  f"iou={avg_iou:.4f} adv_std={avg_adv:.4f} pos={pos_count}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_rpn.remove(); hk_roi.remove()

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.43 IoU-RL Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
