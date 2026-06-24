"""Diagnostic: MRI G-field modulation for displacement-sensitive quality.

Key: patch → multiply by e^{jGx} → FFT → phase encodes displacement.
With G=32, 1px shift → π rad phase difference (2π·G·1/N = 2π·32/64 = π).
"""
import sys, torch, torch.nn.functional as F, numpy as np
from torchvision.ops import box_iou
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.seed import set_seed
set_seed(42)
DEV = "cuda"; M = 8; PIX = 64; MAX_P = 200; G = 32.0

def gfield_quality(patches):
    """G-field modulated phase consistency across multiple G values.

    Multiply patch by e^{jGx}, FFT, measure phase at the G-frequency.
    For a shifted patch, this phase changes linearly with displacement.
    """
    gray = patches.float().mean(dim=1)  # (N, 64, 64)
    N, H, W = gray.shape
    x = torch.arange(W, device=patches.device).float()  # (64,)
    y = torch.arange(H, device=patches.device).float()  # (64,)

    # Multiple G values: measure phase at each
    G_vals = torch.tensor([4.0, 8.0, 16.0, 32.0, 48.0], device=patches.device)
    all_phase = []

    for g in G_vals:
        # Modulate: multiply pixel values by complex exponential in x-direction
        carrier = torch.exp(2j * np.pi * g * x / W)  # (64,) complex
        modulated = gray.float() * carrier.unsqueeze(0).unsqueeze(1)  # (N, 64, 64) complex

        # FFT on complex-valued modulated signal
        fft = torch.fft.fft2(modulated)
        # The G-th bin corresponds to the carrier frequency
        g_bin = int(g.item()) % W
        bin_val = fft[:, g_bin % H, g_bin]  # (N,) complex
        phase = torch.angle(bin_val)
        all_phase.append(phase.unsqueeze(1))

    phases = torch.cat(all_phase, dim=1)  # (N, 5)

    # Quality: low variance of phase across G values = structured content
    # Random noise → random phases at different G → high variance
    # Structured edges → consistent phase response → low variance
    phase_consistency = 1.0 - phases.var(dim=1) / (np.pi**2 / 3)  # normalize by uniform variance
    return phase_consistency.clamp(0, 1)


def pixel_fft_quality(patches):
    gray = patches.float().mean(dim=1)
    fft = torch.fft.fft2(gray).abs()
    mag_flat = fft.flatten(1)
    total = mag_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf = mag_flat[:, mag_flat.shape[1] // 2:].sum(dim=1) / total.squeeze(1)
    mag_norm = mag_flat / total
    entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum(dim=1)
    max_e = torch.log(torch.tensor(float(mag_flat.shape[1]), device=patches.device))
    e_norm = 1.0 - entropy / max_e
    pha_var = torch.angle(torch.fft.fft2(gray) + 1e-6).flatten(1).std(dim=1).clamp_max(1.0)
    quality = 0.3 * hf + 0.4 * e_norm + 0.3 * (1.0 - pha_var)
    return quality.clamp(0.0, 1.0)


def decode_boxes(proposals, deltas):
    w = proposals[:,2] - proposals[:,0]; h = proposals[:,3] - proposals[:,1]
    cx = proposals[:,0] + 0.5*w; cy = proposals[:,1] + 0.5*h
    px = deltas[:,0]*w + cx - 0.5*torch.exp(deltas[:,2])*w
    py = deltas[:,1]*h + cy - 0.5*torch.exp(deltas[:,3])*h
    return torch.stack([px, py, deltas[:,0]*w+cx+0.5*torch.exp(deltas[:,2])*w,
                        deltas[:,1]*h+cy+0.5*torch.exp(deltas[:,3])*h], dim=1).clamp(min=0)

cfg = {"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                 "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                 "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
model = build_detector(cfg).to(DEV)
ckpt = torch.load("runs/round227_v1_baseline_20ep/checkpoint_best.pth", map_location=DEV)
model.load_state_dict(ckpt["model"]); model.eval()
loaders = build_penn_fudan_loaders({
    "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
    "train": {"batch_size": 2}})

# --- Test 1: synthetic shift sensitivity ---
print("=== Test 1: Synthetic shift sensitivity ===")
test_patch = torch.randn(3, 64, 64)
for shift in [0, 1, 2, 3, 5]:
    shifted = torch.roll(test_patch, shifts=shift, dims=-1)
    pb = torch.stack([test_patch, shifted]).to(DEV)
    q = gfield_quality(pb)
    print(f"  shift={shift}px:  q_original={q[0]:.4f}  q_shifted={q[1]:.4f}  delta={abs(q[1]-q[0]):.4f}")

# --- Test 2: real data correlation with IoU ---
print("\n=== Test 2: G-field quality vs IoU on real proposals ===")
pc = {}; rc = {}
model.rpn.register_forward_hook(lambda m,i,o: pc.update({"p": o[0]}))
model.roi_heads.box_head.register_forward_pre_hook(lambda m,i: rc.update({"x": i[0]}))

all_gq, all_pq, all_i = [], [], []

for images, targets in loaders[0]:
    pc.clear(); rc.clear()
    model([img.to(DEV) for img in images], [{k: v.to(DEV) for k, v in t.items()} for t in targets])
    rf = rc.get("x"); pr = pc.get("p")
    if rf is None or rf.shape[0] == 0: continue
    N = rf.shape[0]; bf = model.roi_heads.box_head(rf)
    mu = model.roi_heads.box_predictor.bbox_pred(bf)[:, -4:]
    pc_ = torch.cat(pr, dim=0); N = min(N, pc_.shape[0], MAX_P); mu = mu[:N]
    eps = torch.randn(N, M, 4, device=DEV)
    deltas = mu.unsqueeze(1) + 0.1 * eps
    ad = deltas.reshape(N*M, 4)
    pe = pc_[:N].unsqueeze(1).expand(-1, M, -1).reshape(N*M, 4)
    boxes = decode_boxes(pe, ad)
    npi = [p.shape[0] for p in pr]
    ii = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(npi)], dim=0)[:N]
    patches = []; Kt = N * M; sl = min(Kt, 256)
    for idx in range(sl):
        pi_idx = min(idx // M, N - 1); img_i = ii[pi_idx].item(); img = images[img_i]; b = boxes[idx]
        x1, y1 = max(0, int(b[0].round().item())), max(0, int(b[1].round().item()))
        x2, y2 = min(img.shape[-1], max(x1+1, int(b[2].round().item()))), min(img.shape[-2], max(y1+1, int(b[3].round().item())))
        crop = img[:, y1:y2, x1:x2]
        patches.append(F.interpolate(crop.unsqueeze(0).float(), size=(PIX, PIX), mode='bilinear', align_corners=False).squeeze(0) if crop.shape[-1]>=4 and crop.shape[-2]>=4 else torch.zeros(3, PIX, PIX))
    pb = torch.stack(patches).to(DEV)
    gq = gfield_quality(pb)
    pq = pixel_fft_quality(pb)
    gqp = torch.zeros(Kt, device=DEV); gqp[:sl] = gq; gqm = gqp.view(N, M)[:N]
    pqp = torch.zeros(Kt, device=DEV); pqp[:sl] = pq; pqm = pqp.view(N, M)[:N]
    pim = []; nb = 0
    for ip, p in enumerate(pr):
        for _ in range(p.shape[0]):
            if nb < N: pim.append(ip)
            nb += 1
    pim = pim[:N]; im = torch.zeros(N, M)
    for pi in range(N):
        gt_ = targets[pim[pi]]["boxes"].to(DEV)
        if len(gt_) > 0: ious = box_iou(boxes[pi*M:(pi+1)*M], gt_); im[pi] = ious.max(dim=1).values
    for pi in range(N):
        all_gq.extend(gqm[pi].tolist()); all_pq.extend(pqm[pi].tolist()); all_i.extend(im[pi].tolist())
    if len(all_gq) > 3000: break

gqa = np.array(all_gq); pqa = np.array(all_pq); ia = np.array(all_i)
print(f"\n  Total pairwise: {len(gqa)}")
print(f"  {'':20s}  {'r(IoU)':>8s}  {'q[IoU<0.2]':>10s}  {'q[IoU>0.4]':>10s}  {'top1':>6s}")
print(f"  {'G-field':20s}  {np.corrcoef(gqa, ia)[0,1]:8.4f}  {gqa[ia<0.2].mean():10.4f}  {gqa[ia>0.4].mean():10.4f}  {np.mean([gqa.reshape(-1,M)[i].argmax()==ia.reshape(-1,M)[i].argmax() for i in range(min(len(gqa)//M, 200))]):6.3f}")
print(f"  {'Pixel FFT':20s}  {np.corrcoef(pqa, ia)[0,1]:8.4f}  {pqa[ia<0.2].mean():10.4f}  {pqa[ia>0.4].mean():10.4f}  {np.mean([pqa.reshape(-1,M)[i].argmax()==ia.reshape(-1,M)[i].argmax() for i in range(min(len(pqa)//M, 200))]):6.3f}")
print(f"  G-field quality: mean={gqa.mean():.4f}  std={gqa.std():.4f}  range=[{gqa.min():.4f}, {gqa.max():.4f}]")
