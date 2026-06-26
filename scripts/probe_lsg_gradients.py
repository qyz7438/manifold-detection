"""Gradient and spectrum analysis probe for LearnedSpectralGate (LSG v1).

Reports:
  - strict identity at init (output - input norm)
  - gradient flow through FFT/iFFT
  - magnitude gate statistics and radial profile
  - frequency-position selectivity (low / mid / high frequency response)
"""

import math
import torch
import numpy as np
from spectral_detection_posttrain.methods.detection.pbg import LearnedSpectralGate


def _bin_by_radius(gate, n_bins=4):
    """Return mean gate per radial frequency bin."""
    h, w = gate.shape[-2], gate.shape[-1]
    fy = torch.fft.fftfreq(h, device=gate.device, dtype=gate.dtype).view(h, 1)
    fx = torch.fft.rfftfreq((w - 1) * 2, device=gate.device, dtype=gate.dtype).view(1, w)
    r = torch.sqrt(fx ** 2 + fy ** 2)
    r = r / (r.max() + 1e-6)

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=gate.device)
    means = []
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.any():
            means.append(gate[..., mask].mean().item())
        else:
            means.append(float("nan"))
    return means


def probe_lsg(channels=256, h=14, w=14, use_radius=True, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lsg = LearnedSpectralGate(channels, alpha_init=0.1, use_radius=use_radius).to(device)
    lsg.eval()

    x = torch.randn(2, channels, h, w, device=device, requires_grad=True)
    y = lsg(x)

    identity_error = (y - x).abs().max().item()

    loss = (y ** 2).mean()
    loss.backward()

    print(f"\n=== LSG v1 probe (use_radius={use_radius}, device={device}) ===")
    print(f"input shape       : {tuple(x.shape)}")
    print(f"output shape      : {tuple(y.shape)}")
    print(f"max |y - x|       : {identity_error:.2e}  (should be ~0 at init)")
    print(f"input grad ok?    : {x.grad is not None and x.grad.abs().max().item() > 0}")
    print(f"alpha value       : {lsg.alpha.item():.4f}")
    print(f"alpha grad        : {lsg.alpha.grad.item():.4f}")

    with torch.no_grad():
        X = torch.fft.rfft2(x, norm="ortho")
        mag = torch.log1p(torch.abs(X))
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (
            mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        )
        if use_radius:
            r = lsg._radius_map(mag.size(-2), mag.size(-1), mag.device, mag.dtype)
            r = r.expand(mag.size(0), 1, mag.size(-2), mag.size(-1))
            gate_input = torch.cat([mag, r], dim=1)
        else:
            gate_input = mag
        delta = torch.tanh(lsg.mag_gate(gate_input))
        gate = 1.0 + delta

    print(f"\ngate stats:")
    print(f"  mean : {gate.mean().item():.4f}")
    print(f"  std  : {gate.std().item():.4f}")
    print(f"  min  : {gate.min().item():.4f}")
    print(f"  max  : {gate.max().item():.4f}")

    print(f"\nradial profile (mean gate per bin):")
    radial = _bin_by_radius(gate)
    for i, m in enumerate(radial):
        print(f"  bin {i}: {m:.4f}")

    # Entropy-like measure: how far from uniform?
    flat = gate.view(-1)
    uniform_std = flat.std().item()
    print(f"\ngate std (non-uniformity signal): {uniform_std:.4f}")
    if uniform_std < 0.01:
        print("WARNING: gate is nearly uniform -> no frequency selectivity!")


if __name__ == "__main__":
    probe_lsg(use_radius=False)
    probe_lsg(use_radius=True)
