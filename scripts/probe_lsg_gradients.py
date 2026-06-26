"""Gradient and spectrum analysis probe for LearnedSpectralGate (LSG).

Reports:
  - gradient flow through FFT/iFFT
  - magnitude gate statistics (mean, std, range)
  - frequency selectivity: gate variance across frequency bins
"""

import torch
import numpy as np
from spectral_detection_posttrain.methods.detection.pbg import LearnedSpectralGate


def probe_lsg(channels=256, h=14, w=14, use_phase=False, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lsg = LearnedSpectralGate(channels, alpha_init=1.0, use_phase=use_phase).to(device)
    lsg.train()

    x = torch.randn(2, channels, h, w, device=device, requires_grad=True)
    y = lsg(x)
    loss = (y ** 2).mean()
    loss.backward()

    print(f"\n=== LSG probe (use_phase={use_phase}, device={device}) ===")
    print(f"input shape  : {tuple(x.shape)}")
    print(f"output shape : {tuple(y.shape)}")
    print(f"input grad?  : {x.grad is not None and x.grad.abs().max().item() > 0}")
    print(f"alpha grad   : {lsg.alpha.grad.item():.4f}")

    with torch.no_grad():
        X = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(X)
        gate = 1.0 + torch.tanh(lsg.mag_gate(mag))

    print(f"\nmag_gate stats:")
    print(f"  mean : {gate.mean().item():.4f}")
    print(f"  std  : {gate.std().item():.4f}")
    print(f"  min  : {gate.min().item():.4f}")
    print(f"  max  : {gate.max().item():.4f}")

    # Frequency selectivity: std across frequency bins per channel/sample
    gate_per_freq = gate.view(gate.size(0), channels, -1)
    freq_std = gate_per_freq.std(dim=-1)
    print(f"\nper-frequency std:")
    print(f"  mean : {freq_std.mean().item():.4f}")
    print(f"  std  : {freq_std.std().item():.4f}")

    # Compare low vs high frequency response
    # rfft2 layout: DC at (0,0), high freq at far edges
    low_freq_mask = torch.zeros_like(gate[0, 0], dtype=torch.bool)
    low_freq_mask[:h//4, :w//4] = True
    high_freq_mask = ~low_freq_mask

    low_resp = gate[:, :, low_freq_mask].mean().item()
    high_resp = gate[:, :, high_freq_mask].mean().item()
    print(f"\nlow-freq  gate mean : {low_resp:.4f}")
    print(f"high-freq gate mean : {high_resp:.4f}")
    print(f"low/high ratio      : {low_resp / (high_resp + 1e-12):.3f}")


if __name__ == "__main__":
    probe_lsg(use_phase=False)
    probe_lsg(use_phase=True)
