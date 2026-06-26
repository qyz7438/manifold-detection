"""Statistical analysis of FSBG phase boundary output.

Measures:
  - output magnitude distribution
  - edge selectivity (response at known edges vs flat regions)
  - correlation with spatial gradient magnitude
  - signal-to-noise ratio
  - effect of phase_mask modes

Usage:
    PYTHONPATH=. python scripts/analyze_phase_boundary.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from spectral_detection_posttrain.methods.detection.pbg import FrequencySpatialBoundaryGate


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradient magnitude with Sobel filters."""
    gx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=x.dtype, device=x.device)
    gy = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=x.dtype, device=x.device)
    c = x.size(1)
    gx = gx.repeat(c, 1, 1, 1)
    gy = gy.repeat(c, 1, 1, 1)
    dx = F.conv2d(x, gx, padding=1, groups=c)
    dy = F.conv2d(x, gy, padding=1, groups=c)
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-12)


def make_edge_image(b: int, c: int, h: int, w: int, device: str = "cpu"):
    """Synthetic image with vertical/horizontal step edges and noise."""
    x = torch.zeros(b, c, h, w, device=device)
    x[:, :, h // 4:, :] += 1.0
    x[:, :, :, w // 3:] += 0.7
    # Add smooth blobs
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device), indexing='ij')
    blob = torch.sin(4 * np.pi * xx) * torch.cos(3 * np.pi * yy)
    x += blob.unsqueeze(0).unsqueeze(0) * 0.3
    # Add noise
    x += torch.randn_like(x) * 0.05
    return x


def make_random_image(b: int, c: int, h: int, w: int, device: str = "cpu"):
    return torch.randn(b, c, h, w, device=device)


def analyze_phase_boundary(
    channels: int = 256,
    h: int = 32,
    w: int = 32,
    alpha_init: float = 1e-2,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Phase boundary statistical analysis (device={device}) ===")

    for mask in ["soft", "hard", "none"]:
        print(f"\n--- phase_mask='{mask}' ---")
        fsbg = FrequencySpatialBoundaryGate(channels, alpha_init=alpha_init, phase_mask=mask).to(device)
        fsbg.eval()

        for name, x in [("edge_image", make_edge_image(4, channels, h, w, device)),
                        ("random_noise", make_random_image(4, channels, h, w, device))]:
            with torch.no_grad():
                spatial = fsbg.spatial_edge(x)
                phase = fsbg._phase_boundary(x)
                gate = torch.tanh(spatial + phase + fsbg.fusion_bias)

            grad_mag = sobel_magnitude(x).mean(dim=1, keepdim=True)

            # Flatten spatial dims
            pflat = phase.view(phase.size(0), -1)
            gflat = gate.view(gate.size(0), -1)
            gradflat = grad_mag.view(grad_mag.size(0), -1)
            sflat = spatial.view(spatial.size(0), -1)

            # Correlations per sample
            corr_p_g = []
            corr_p_s = []
            for i in range(phase.size(0)):
                cp = torch.corrcoef(torch.stack([pflat[i], gradflat[i]]))
                corr_p_g.append(float(cp[0, 1]) if not torch.isnan(cp[0, 1]) else 0.0)
                cs = torch.corrcoef(torch.stack([pflat[i], sflat[i]]))
                corr_p_s.append(float(cs[0, 1]) if not torch.isnan(cs[0, 1]) else 0.0)

            print(f"  [{name}]")
            print(f"    phase mean={phase.mean().item():.4f}  std={phase.std().item():.4f}  max={phase.max().item():.4f}")
            print(f"    gate  mean={gate.mean().item():.4f}  std={gate.std().item():.4f}  max={gate.max().item():.4f}")
            print(f"    spatial mean={spatial.mean().item():.4f}  std={spatial.std().item():.4f}")
            print(f"    corr(phase, sobel_grad) = {np.mean(corr_p_g):.3f} ± {np.std(corr_p_g):.3f}")
            print(f"    corr(phase, spatial)    = {np.mean(corr_p_s):.3f} ± {np.std(corr_p_s):.3f}")

            # Edge selectivity: response on top-10% gradient pixels vs bottom-50%
            top_mask = gradflat > gradflat.quantile(0.90)
            bot_mask = gradflat < gradflat.quantile(0.50)
            top_resp = pflat[top_mask].mean().item()
            bot_resp = pflat[bot_mask].mean().item()
            selectivity = top_resp / (bot_resp + 1e-12)
            print(f"    edge selectivity (top10%/bot50%) = {selectivity:.3f}")

    print("\n=== Interpretation ===")
    print("- High corr(phase, sobel_grad) / high selectivity → phase boundary aligns with real edges.")
    print("- Low correlation / selectivity ≈ 1 → phase boundary is no better than random.")


if __name__ == "__main__":
    analyze_phase_boundary(channels=256, h=32, w=32)
