"""Plan 2.39: SPR — NaN fixes + robust GT ROI handling."""
import sys, json, subprocess, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import roi_align, box_iou

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
ALPHAS = [0.1, 0.5]
EPOCHS = 5
PATCH_SIZE = 7


# ─── Proposal Refiner ─────────────────────────────────────────
class ProposalRefiner(nn.Module):
    """Lightweight MLP: ROI features → (μ, logσ) for bbox delta sampling."""

    def __init__(self, roi_channels=256, hidden=256, num_samples=8):
        super().__init__()
        self.M = num_samples
        self.fc1 = nn.Linear(roi_channels * 7 * 7, hidden)
        self.fc_mu = nn.Linear(hidden, 4)
        self.fc_logsigma = nn.Linear(hidden, 4)

    def forward(self, roi_feats):
        """roi_feats: (N, C, 7, 7) from box_roi_pool → (μ, logσ, log_probs) per sample."""
        N = roi_feats.size(0)
        x = roi_feats.flatten(1)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)                     # (N, 4)
        sigma = F.softplus(self.fc_logsigma(x)) + 1e-6  # (N, 4)

        # Sample M deltas per proposal
        mu_rep = mu.unsqueeze(1).expand(-1, self.M, -1).reshape(N * self.M, 4)
        sig_rep = sigma.unsqueeze(1).expand(-1, self.M, -1).reshape(N * self.M, 4)
        eps = torch.randn_like(mu_rep)
        deltas = mu_rep + sig_rep * eps

        # Log probability per sample (Gaussian)
        log_prob = -0.5 * (eps.pow(2) + 2 * torch.log(sig_rep) + math.log(2 * math.pi)).sum(dim=-1)

        return mu_rep, sig_rep, deltas, log_prob  # all (N*M, ...)


def decode_boxes(boxes, deltas):
    """boxes, deltas: (N, 4) xyxy → decoded xyxy."""
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
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


# ─── Fourier Reward ───────────────────────────────────────────
def fourier_reward(pred_roi, gt_roi):
    """Per-box FFT comparison: pred ROI vs GT ROI spectrum."""
    K = pred_roi.shape[0]
    pred_gray = pred_roi.mean(dim=1)
    gt_gray = gt_roi.mean(dim=1)
    A_pred = torch.log1p(torch.fft.fft2(pred_gray.float()).abs().clamp(min=1e-6))
    A_gt = torch.log1p(torch.fft.fft2(gt_gray.float()).abs().clamp(min=1e-6))

    freq = torch.fft.fftfreq(PATCH_SIZE, device=pred_roi.device)
    Y, X = torch.meshgrid(freq, freq, indexing='ij')
    radius = torch.sqrt(X**2 + Y**2)
    weights = [(radius <= 1.0, 0.5), ((radius > 1.0) & (radius <= 2.5), 0.2), (radius > 2.5, 0.3)]

    reward = torch.zeros(K, device=pred_roi.device)
    for mask, w in weights:
        diff = (A_pred[:, mask] - A_gt[:, mask]).abs().mean(dim=-1)
        reward -= w * diff
    return reward


# ─── Dataset & Model ──────────────────────────────────────────
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


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        out = model([img.to(DEV) for img in images])
        preds.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


# ─── Main ─────────────────────────────────────────────────────
def main():
    all_r = []

    for alpha in ALPHAS:
        run_name = f"round239_spr_a{alpha}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        # Freeze backbone + FPN + RPN + box_head
        freeze_all(model)
        # Only SPR is trainable
        spr = ProposalRefiner(roi_channels=256, num_samples=M_SAMPLES).to(DEV)

        train_loader, val_loader = build_loaders()
        opt = torch.optim.Adam(spr.parameters(), lr=0.001)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0
        reward_baseline = None

        # Get model components
        backbone = model.backbone
        rpn = model.rpn
        roi_heads = model.roi_heads
        box_roi_pool = roi_heads.box_roi_pool
        box_head = roi_heads.box_head
        box_predictor = roi_heads.box_predictor
        transform = model.transform

        for epoch in range(1, EPOCHS + 1):
            model.train()
            spr.train()
            total_rl, total_det = 0.0, 0.0
            avg_rw = 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                orig_shapes = [(img.shape[-2], img.shape[-1]) for img in images]

                # 1. Standard detection forward for stability loss
                ld = model(images, targets_t)
                det_loss = sum(ld.values())

                # 2. SPR: RPN proposals → SPR refine → spectral reward → REINFORCE
                images_t, _ = transform(images, None)
                fpn_feat = backbone(images_t.tensors)
                with torch.no_grad():
                    proposals, _ = rpn(images_t, fpn_feat, targets_t)

                # Skip if too few proposals
                total_props = sum(p.shape[0] for p in proposals)
                if total_props < 4:
                    loss = det_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total_det += det_loss.item()
                    continue

                # Box pool → SPR
                roi_feats_for_spr = box_roi_pool(fpn_feat, proposals, images_t.image_sizes)
                mu_rep, sig_rep, deltas, log_probs = spr(roi_feats_for_spr)

                # Decode refined boxes
                props_cat = torch.cat(proposals, dim=0).repeat_interleave(M_SAMPLES, dim=0)
                refined_boxes = decode_boxes(props_cat, deltas)

                # Clip to image bounds
                N_total = refined_boxes.shape[0]
                refined_list = []
                n_per_img = [p.shape[0] * M_SAMPLES for p in proposals]
                idx = 0
                for img_i, n in enumerate(n_per_img):
                    rb = refined_boxes[idx:idx + n].clone()
                    rb[:, 0].clamp_(0, orig_shapes[img_i][1])
                    rb[:, 2].clamp_(0, orig_shapes[img_i][1])
                    rb[:, 1].clamp_(0, orig_shapes[img_i][0])
                    rb[:, 3].clamp_(0, orig_shapes[img_i][0])
                    refined_list.append(rb)
                    idx += n

                # Per-delta independent ROI Align → FFT reward
                refined_cat = torch.cat(refined_list, dim=0)
                roi_refined = box_roi_pool(fpn_feat, refined_list, images_t.image_sizes)

                # Build GT ROI features: match each base proposal to GT, replicate M times
                gt_roi_list = []
                for img_i, props_img in enumerate(proposals):
                    tgt = targets_t[img_i]
                    n_p = props_img.shape[0]
                    if len(tgt["boxes"]) > 0:
                        ious = box_iou(props_img, tgt["boxes"])
                        best_iou, best_idx = ious.max(dim=1)
                        for j in range(n_p):
                            if best_iou[j] > 0.3:
                                gt_box = tgt["boxes"][best_idx[j]:best_idx[j] + 1]
                                # Replicate M times
                                for _ in range(M_SAMPLES):
                                    gt_roi_list.append(gt_box)
                            else:
                                for _ in range(M_SAMPLES):
                                    gt_roi_list.append(torch.tensor([[0, 0, 1, 1]], device=DEV, dtype=torch.float32))
                    else:
                        for _ in range(n_p * M_SAMPLES):
                            gt_roi_list.append(torch.tensor([[0, 0, 1, 1]], device=DEV, dtype=torch.float32))

                gt_boxes_t = torch.cat(gt_roi_list, dim=0)
                roi_gt = box_roi_pool(fpn_feat, [gt_boxes_t], images_t.image_sizes)

                # Fourier reward per delta
                rewards = fourier_reward(roi_refined, roi_gt)

                # Greedy baseline (no noise)
                with torch.no_grad():
                    greedy_deltas = mu_rep  # μ only, no noise
                    greedy_boxes = decode_boxes(props_cat, greedy_deltas)
                    greedy_list = []
                    idx = 0
                    for img_i, n in enumerate(n_per_img):
                        gb = greedy_boxes[idx:idx + n].clone()
                        gb[:, 0].clamp_(0, orig_shapes[img_i][1])
                        gb[:, 2].clamp_(0, orig_shapes[img_i][1])
                        gb[:, 1].clamp_(0, orig_shapes[img_i][0])
                        gb[:, 3].clamp_(0, orig_shapes[img_i][0])
                        greedy_list.append(gb)
                        idx += n
                    roi_greedy = box_roi_pool(fpn_feat, greedy_list, images_t.image_sizes)
                    baseline_rewards = fourier_reward(roi_greedy, roi_gt)

                # REINFORCE: per-delta advantage
                advantage = rewards - baseline_rewards.detach()
                rl_loss = -(advantage.detach() * log_probs).mean()
                avg_rw = rewards.mean().item()

                # NaN guard — use only log_probs for grad if reward is NaN
                if torch.isnan(rl_loss) or torch.isinf(rl_loss):
                    rl_loss = log_probs.mean() * 0.0  # keeps grad through log_probs

                loss = det_loss.detach() + alpha * rl_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_rl += rl_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} rl={total_rl:.3f} r={avg_rw:.4f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.38 Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
