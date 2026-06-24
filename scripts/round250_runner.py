"""Plan 2.50: X-DPO (eXplainable DPO) — 3 white-box metrics + Pareto dominance.

Metrics: q_edge (boundary ring energy), q_smooth (internal homogeneity), q_overlap (IoU).
Preference: wins at least 2/3 → chosen. Only valid pairs contribute to DPO loss.
"""
import sys, json, subprocess, math, copy
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
K_SAMPLES = 2  # pairwise for DPO
BETAS = [0.5, 1.0]
EPOCHS = 10
PIXEL_SIZE = 64


def edge_boundary_quality(patches):
    """Sobel response in 8px boundary ring / total edge energy."""
    gray = patches.float().mean(dim=1)
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]
    edge = torch.sqrt(gx[:, :-1, :].pow(2) + gy[:, :, :-1].pow(2) + 1e-6)  # (N,63,63)
    total = edge.flatten(1).sum(dim=1).clamp_min(1e-6)
    b = 8
    ring = torch.zeros_like(edge)
    ring[:, :b, :] = 1; ring[:, -b:, :] = 1
    ring[:, :, :b] = 1; ring[:, :, -b:] = 1
    boundary = (edge * ring).flatten(1).sum(dim=1)
    return (boundary / total).clamp(0, 1)


def internal_smoothness(patches):
    """1 - Var(patch) uses variance as texture proxy. Low var = smooth."""
    gray = patches.float().mean(dim=1)
    var_p = gray.flatten(1).var(dim=1)
    return (1.0 - 1.0 / (1.0 + var_p * 100)).clamp(0, 1)


def gaussian_log_prob(deltas, mu, sigma):
    eps = (deltas - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    return -0.5 * (eps.pow(2) + 2 * torch.log(sigma.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


def decode_boxes(proposals, deltas):
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths; ctr_y = proposals[:, 1] + 0.5 * heights
    pred_ctr_x = deltas[:, 0] * widths + ctr_x
    pred_ctr_y = deltas[:, 1] * heights + ctr_y
    pred_w = torch.exp(deltas[:, 2]) * widths; pred_h = torch.exp(deltas[:, 3]) * heights
    refined = torch.zeros_like(deltas)
    refined[:, 0] = pred_ctr_x - 0.5 * pred_w; refined[:, 1] = pred_ctr_y - 0.5 * pred_h
    refined[:, 2] = pred_ctr_x + 0.5 * pred_w; refined[:, 3] = pred_ctr_y + 0.5 * pred_h
    return refined.clamp(min=0)


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

    for beta in BETAS:
        run_name = f"round250_xdpo_b{beta}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        ref_model = copy.deepcopy(model)
        freeze_except(ref_model, [])
        ref_model.eval()

        freeze_except(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])

        train_loader, val_loader = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0

        proposal_cache = {}
        roi_cache = {}

        def rpn_hook(module, inp, out):
            proposal_cache["p"] = out[0]
        def roi_hook(module, inp):
            roi_cache["x"] = inp[0]

        hk_rpn = model.rpn.register_forward_hook(rpn_hook)
        hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_det, total_dpo = 0.0, 0.0
            valid_pairs = 0; total_pairs = 0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images_dev = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                proposal_cache.clear(); roi_cache.clear()

                ld = model(images_dev, targets_t)
                det_loss = sum(ld.values())

                roi_feats = roi_cache.get("x")
                proposals = proposal_cache.get("p")
                dpo_loss = torch.tensor(0.0, device=DEV)

                if roi_feats is not None and proposals is not None and roi_feats.shape[0] > 0:
                    N = roi_feats.shape[0]
                    box_ft = model.roi_heads.box_head(roi_feats)
                    mu = model.roi_heads.box_predictor.bbox_pred(box_ft)[:, -4:]
                    sigma = torch.full_like(mu, 0.1)

                    eps = torch.randn(N, K_SAMPLES, 4, device=DEV)
                    deltas = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps

                    log_probs = gaussian_log_prob(deltas, mu, sigma)
                    with torch.no_grad():
                        ref_ft = ref_model.roi_heads.box_head(roi_feats)
                        ref_mu = ref_model.roi_heads.box_predictor.bbox_pred(ref_ft)[:, -4:]
                        ref_sigma = torch.full_like(ref_mu, 0.1)
                    ref_deltas = deltas.detach()
                    log_probs_ref = gaussian_log_prob(ref_deltas, ref_mu, ref_sigma)

                    proposals_cat = torch.cat(proposals, dim=0)
                    N = min(N, proposals_cat.shape[0])
                    mu = mu[:N]; deltas = deltas[:N]
                    log_probs = log_probs[:N]; log_probs_ref = log_probs_ref[:N]
                    sigma = sigma[:N]

                    ad = deltas.reshape(N * K_SAMPLES, 4)
                    pe = proposals_cat[:N].unsqueeze(1).expand(-1, K_SAMPLES, -1).reshape(N * K_SAMPLES, 4)
                    all_boxes = decode_boxes(pe, ad)

                    npi = [p.shape[0] for p in proposals]
                    ii = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(npi)], dim=0)[:N]

                    patches_list = []
                    for idx in range(min(N * K_SAMPLES, 256)):
                        pi = min(idx // K_SAMPLES, N - 1)
                        img_i = ii[pi].item(); img = images[img_i]; box = all_boxes[idx]
                        x1, y1 = max(0, int(box[0].round().item())), max(0, int(box[1].round().item()))
                        x2, y2 = min(img.shape[-1], max(x1 + 1, int(box[2].round().item()))), min(img.shape[-2], max(y1 + 1, int(box[3].round().item())))
                        crop = img[:, y1:y2, x1:x2]
                        if crop.shape[-1] >= 4 and crop.shape[-2] >= 4:
                            crop = F.interpolate(crop.unsqueeze(0).float(), size=(PIXEL_SIZE, PIXEL_SIZE), mode='bilinear', align_corners=False).squeeze(0)
                            patches_list.append(crop)
                        else:
                            patches_list.append(torch.zeros(3, PIXEL_SIZE, PIXEL_SIZE))

                    if patches_list:
                        pb = torch.stack(patches_list).to(DEV)
                        q_edge = edge_boundary_quality(pb)
                        q_smooth = internal_smoothness(pb)

                        # IoU overlap per sampled box
                        Kt = N * K_SAMPLES; qov = torch.zeros(Kt, device=DEV)
                        pim = []; nb = 0
                        for ip, p in enumerate(proposals):
                            for _ in range(p.shape[0]):
                                if nb < N: pim.append(ip)
                                nb += 1
                        pim = pim[:N]
                        for pi in range(N):
                            gt_boxes = targets_t[pim[pi]]["boxes"]
                            if len(gt_boxes) > 0:
                                ious = box_iou(all_boxes[pi * K_SAMPLES:(pi + 1) * K_SAMPLES], gt_boxes)
                                qov[pi * K_SAMPLES:(pi + 1) * K_SAMPLES] = ious.max(dim=1).values

                        qp_edge = torch.zeros(Kt, device=DEV); qp_edge[:len(q_edge)] = q_edge
                        qp_smooth = torch.zeros(Kt, device=DEV); qp_smooth[:len(q_smooth)] = q_smooth

                        qe = qp_edge.view(N, K_SAMPLES); qs = qp_smooth.view(N, K_SAMPLES); qo = qov.view(N, K_SAMPLES)

                        # Pareto preference: 2/3 wins
                        wins_0 = (qe[:, 0] > qe[:, 1]).float() + (qs[:, 0] > qs[:, 1]).float() + (qo[:, 0] > qo[:, 1]).float()
                        wins_1 = 3.0 - wins_0
                        chosen_mask = wins_0 >= 2  # sample 0 wins
                        rejected_mask = wins_1 >= 2  # sample 1 wins
                        valid = chosen_mask | rejected_mask

                        if valid.sum() > 0:
                            lp_c = torch.where(chosen_mask & valid, log_probs[:, 0], log_probs[:, 1])
                            lp_r = torch.where(chosen_mask & valid, log_probs[:, 1], log_probs[:, 0])
                            lp_ref_c = torch.where(chosen_mask & valid, log_probs_ref[:, 0], log_probs_ref[:, 1])
                            lp_ref_r = torch.where(chosen_mask & valid, log_probs_ref[:, 1], log_probs_ref[:, 0])
                            ratio = lp_c - lp_ref_c - lp_r + lp_ref_r
                            dpo_loss = -F.logsigmoid(beta * ratio[valid]).mean()
                            valid_pairs += valid.sum().item()
                        total_pairs += N

                loss = det_loss + dpo_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_dpo += dpo_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} dpo={total_dpo:.3f} valid={valid_pairs}/{total_pairs}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_rpn.remove(); hk_roi.remove()

        ep_m.update({"run_name": run_name, "beta": beta, "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE b{beta}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.50 X-DPO Results")
    for r in all_r:
        print(f"  b{r['beta']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
