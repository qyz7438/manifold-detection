"""Gradient diagnosis for MicroAFM: why mag_scale and phase_scale stay at zero.

Hypothesis: residual_scale provides a clean, direct gradient path through
  loss -> output -> x + s_res * residual
  while mag_scale/phase_scale require gradients through
  loss -> output -> x + s_res * residual -> freq_out -> iRFFT -> complex -> gate -> scale
  The FFT-path gradients are weaker/noisier and the optimizer starves them.

Tests:
  1. Gradient magnitude comparison: |dL/d(residual_scale)| vs |dL/d(mag_scale)| vs |dL/d(phase_scale)|
  2. Does gradient exist at all for mag_scale/phase_scale when scales are 0?
  3. What does the gate output distribution look like before scaling?
  4. MPLSeg-style gate: would it get gradient?
"""

import torch
from torch import nn


def build_afm(channels=256, residual_mode="current"):
    """Replicate the AFMBlock from micro_afm.py."""
    from spectral_detection_posttrain.models.micro_afm import AFMBlock
    return AFMBlock(channels=channels, residual_mode=residual_mode)


def build_mplseg_style_afm(in_ch=256, mid_ch=256):
    """Build MPLSeg-style AFM for comparison."""
    return MPLSegStyleAFM(in_ch, mid_ch)


class MPLSegStyleAFM(nn.Module):
    """Minimal MPLSeg-style AFM: double-sigmoid gate, no learnable scales, no residual."""

    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.mp = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch // 4, 1, bias=False),
            nn.InstanceNorm2d(mid_ch // 4),
            nn.Sigmoid(),
            nn.Conv2d(mid_ch // 4, mid_ch // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch // 4),
            nn.Sigmoid(),
            nn.Conv2d(mid_ch // 4, mid_ch, 1, bias=False),
            nn.InstanceNorm2d(mid_ch),
            nn.Sigmoid(),
        )
        self.pa = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch // 4, 1, bias=False),
            nn.InstanceNorm2d(mid_ch // 4),
            nn.Tanh(),
            nn.Conv2d(mid_ch // 4, mid_ch // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch // 4),
            nn.Tanh(),
            nn.Conv2d(mid_ch // 4, mid_ch, 1, bias=False),
            nn.InstanceNorm2d(mid_ch),
            nn.Tanh(),
        )
        self.eps = 1e-3
        # Kaiming init
        for ly in [self.mp, self.pa]:
            for m in ly:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0)

    def forward(self, x):
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + self.eps)

        mag = mag * (1 - self.mp(torch.sigmoid(torch.log(mag + self.eps))))
        pha = pha + self.pa(pha)

        fr = mag * torch.exp(1j * pha)
        output = torch.fft.irfft2(fr, norm="ortho")
        output = torch.relu(output)
        return output


def diagnose_gradients(afm, x, label="AFM"):
    """Run one backward pass and report gradient norms for all parameters."""
    x_grad = x.detach().clone().requires_grad_(True)

    # Use a simple loss: MSE between output and a random target
    # This simulates detection loss flowing through the AFM
    target = torch.randn_like(x_grad) * 0.1

    output = afm(x_grad)
    if output.shape != target.shape:
        target = target[:, :output.shape[1], :output.shape[2], :output.shape[3]]

    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    results = {"label": label, "loss": float(loss.item())}

    for name, param in afm.named_parameters():
        if param.grad is not None:
            grad_norm = float(param.grad.norm().item())
            param_norm = float(param.norm().item())
            results[f"{name}_grad_norm"] = grad_norm
            results[f"{name}_param_norm"] = param_norm
        else:
            results[f"{name}_grad_norm"] = 0.0
            results[f"{name}_param_norm"] = float(param.norm().item())

    results["x_grad_norm"] = float(x_grad.grad.norm().item()) if x_grad.grad is not None else 0.0
    return results


def diagnose_forward_distribution(afm, x, label="AFM"):
    """Check forward pass: what does the gate output look like?"""
    with torch.no_grad():
        # For our AFM, we need to hook into the intermediate values
        output = afm(x)

    results = {
        "label": label,
        "output_mean": float(output.mean().item()),
        "output_std": float(output.std().item()),
        "output_min": float(output.min().item()),
        "output_max": float(output.max().item()),
        "output_rel_diff": float((output - x).abs().mean().item()) if output.shape == x.shape else None,
    }
    return results


def check_afm_in_detector_context():
    """Verify gradient flows through AFM when inserted in a real detector."""
    from spectral_detection_posttrain.models import build_detector

    config = {
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": True,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
            "afm_channels": 256,
            "afm_type": "identity",
            "afm_residual_mode": "current",
        }
    }

    model = build_detector(config)
    # Find the AFM scales
    if hasattr(model.roi_heads.box_head, "afm"):
        afm = model.roi_heads.box_head.afm
        print("\n=== Detector AFM scales (before training) ===")
        for key in ["mag_scale", "phase_scale", "residual_scale"]:
            if hasattr(afm, key):
                val = getattr(afm, key)
                print(f"  {key}: {float(val.item()):.6f} (requires_grad={val.requires_grad})")

        # Run one forward/backward with a dummy detection batch
        print("\n=== Running detector forward/backward ===")
        x = torch.randn(2, 3, 320, 320)
        targets = [
            {"boxes": torch.tensor([[50.0, 60.0, 200.0, 250.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[30.0, 40.0, 180.0, 220.0]]), "labels": torch.tensor([1])},
        ]
        model.train()
        loss_dict = model([x[0], x[1]], targets)  # Pass individual images
        loss = sum(loss_dict.values())

        # Check AFM gradients before backward
        grads_before = {}
        for key in ["mag_scale", "phase_scale", "residual_scale"]:
            if hasattr(afm, key):
                p = getattr(afm, key)
                grads_before[key] = p.grad.item() if p.grad is not None else None

        loss.backward()

        print("\n=== AFM scale gradients after detector loss backward ===")
        for key in ["mag_scale", "phase_scale", "residual_scale"]:
            if hasattr(afm, key):
                p = getattr(afm, key)
                grad_before = grads_before[key]
                grad_after = p.grad.item() if p.grad is not None else None
                print(f"  {key}: grad={grad_after} (before={grad_before})")
    else:
        print("No AFM found in detector")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # Test 1: Gradient comparison on standalone AFM
    print("\n" + "=" * 60)
    print("Test 1: Standalone AFM gradient comparison")
    print("=" * 60)

    x = torch.randn(2, 256, 16, 16, device=device)

    # Our current AFM (current residual mode)
    afm_ours = build_afm(channels=256, residual_mode="current").to(device)
    results_ours = diagnose_gradients(afm_ours, x, "our_current")
    print(f"\nOur AFM (current residual):")
    for k, v in sorted(results_ours.items()):
        if "grad" in k or "scale" in k:
            print(f"  {k}: {v}")

    # Our AFM (delta residual mode)
    afm_delta = build_afm(channels=256, residual_mode="delta").to(device)
    results_delta = diagnose_gradients(afm_delta, x, "our_delta")
    print(f"\nOur AFM (delta residual):")
    for k, v in sorted(results_delta.items()):
        if "grad" in k or "scale" in k:
            print(f"  {k}: {v}")

    # MPLSeg-style AFM
    afm_mplseg = build_mplseg_style_afm(in_ch=256, mid_ch=256).to(device)
    results_mplseg = diagnose_gradients(afm_mplseg, x, "mplseg_style")
    print(f"\nMPLSeg-style AFM:")
    for k, v in sorted(results_mplseg.items()):
        if "grad" in k or "conv" in k:
            print(f"  {k}: {v}")

    # Test 2: Gradient ratio analysis
    print("\n" + "=" * 60)
    print("Test 2: Gradient ratio (how much weaker is mag_scale gradient?)")
    print("=" * 60)

    mag_grad = abs(results_ours.get("mag_scale_grad_norm", 0))
    res_grad = abs(results_ours.get("residual_scale_grad_norm", 1))
    if res_grad > 0:
        print(f"  |grad(mag_scale)| / |grad(residual_scale)| = {mag_grad / res_grad:.2e}")
    else:
        print("  residual_scale_grad is zero — cannot compute ratio")

    # Test 3: Forward distribution
    print("\n" + "=" * 60)
    print("Test 3: Forward pass output distribution")
    print("=" * 60)

    for name, afm in [("our_current", afm_ours), ("our_delta", afm_delta), ("mplseg_style", afm_mplseg)]:
        results = diagnose_forward_distribution(afm, x, name)
        print(f"\n{name}:")
        for k, v in results.items():
            if k != "label" and v is not None:
                print(f"  {k}: {v}")

    # Test 4: Detector context
    print("\n" + "=" * 60)
    print("Test 4: AFM in real detector context")
    print("=" * 60)
    check_afm_in_detector_context()


if __name__ == "__main__":
    main()
