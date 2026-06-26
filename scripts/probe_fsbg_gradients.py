"""Gradient probe for FrequencySpatialBoundaryGate (FSBG v2).

Attaches forward/backward hooks to every tensor of interest and reports:
  - whether gradients flow (non-zero)
  - gradient magnitude relative to the tensor norm
  - the contribution of the phase vs spatial branch

Usage:
    PYTHONPATH=. python scripts/probe_fsbg_gradients.py
"""

import torch
import torch.nn as nn
from spectral_detection_posttrain.methods.detection.pbg import FrequencySpatialBoundaryGate


def _mag(t: torch.Tensor) -> float:
    return float(t.detach().abs().mean())


def _ratio(grad: torch.Tensor, value: torch.Tensor) -> float:
    """Relative gradient magnitude."""
    g = grad.detach()
    v = value.detach()
    return float(g.abs().mean() / (v.abs().mean() + 1e-12))


class Probe:
    def __init__(self, name: str):
        self.name = name
        self.input = None
        self.output = None
        self.grad_input = None
        self.grad_output = None

    def register(self, module: nn.Module):
        module.register_forward_hook(self._fwd_hook)
        module.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.input = inp[0] if isinstance(inp, tuple) else inp
        self.output = out[0] if isinstance(out, tuple) else out

    def _bwd_hook(self, module, grad_input, grad_output):
        self.grad_input = grad_input[0] if isinstance(grad_input, tuple) else grad_input
        self.grad_output = grad_output[0] if isinstance(grad_output, tuple) else grad_output


def probe_fsbg_unit(
    channels: int = 256,
    h: int = 14,
    w: int = 14,
    alpha_init: float = 1e-2,
    phase_mask: str = "soft",
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fsbg = FrequencySpatialBoundaryGate(channels, alpha_init=alpha_init, phase_mask=phase_mask).to(device)
    fsbg.train()

    probes = {
        "fsbg": Probe("FSBG output"),
        "spatial": Probe("spatial_edge"),
        "phase": Probe("phase_encoder"),
    }
    probes["fsbg"].register(fsbg)
    probes["spatial"].register(fsbg.spatial_edge)
    probes["phase"].register(fsbg.phase_encoder)

    x = torch.randn(2, channels, h, w, device=device, requires_grad=True)
    y = fsbg(x)
    loss = (y ** 2).mean()
    loss.backward()

    print(f"\n=== FSBG v2 gradient probe ({phase_mask=}, {alpha_init=}, device={device}) ===")
    print(f"input  shape : {tuple(x.shape)}")
    print(f"output shape : {tuple(y.shape)}")
    print(f"loss         : {loss.item():.6f}")

    # Input / output gradient flow
    print("\n--- tensor gradients ---")
    print(f"{'tensor':20s} | grad? | grad_mag | rel_grad")
    for name, t, g in [
        ("input", x, x.grad),
        ("output", y, y.grad),
    ]:
        has_grad = g is not None and g.abs().max().item() > 0
        print(f"{name:20s} | {str(has_grad):5s} | {_mag(g) if has_grad else 0.0:8.2e} | {_ratio(g, t) if has_grad else 0.0:8.2e}")

    # Submodule probes
    print("\n--- submodule gradient flow ---")
    print(f"{'module':20s} | out_grad? | in_grad? | out_rel_grad | in_rel_grad")
    for key in ["fsbg", "spatial", "phase"]:
        p = probes[key]
        out_has = p.grad_output is not None and p.grad_output.abs().max().item() > 0
        in_has = p.grad_input is not None and p.grad_input.abs().max().item() > 0
        out_rel = _ratio(p.grad_output, p.output) if out_has else 0.0
        in_rel = _ratio(p.grad_input, p.input) if in_has else 0.0
        print(f"{key:20s} | {str(out_has):9s} | {str(in_has):8s} | {out_rel:12.2e} | {in_rel:11.2e}")

    # Parameter gradients
    print("\n--- parameter gradients ---")
    print(f"{'param':30s} | grad? | grad_mag | rel_grad | shape")
    for name, param in fsbg.named_parameters():
        has_grad = param.grad is not None and param.grad.abs().max().item() > 0
        rel = _ratio(param.grad, param) if has_grad else 0.0
        print(f"{name:30s} | {str(has_grad):5s} | {_mag(param.grad) if has_grad else 0.0:8.2e} | {rel:8.2e} | {tuple(param.shape)}")

    # Phase path isolation: zero out each branch and compare gradients
    print("\n--- branch ablation (gradient contribution) ---")
    x2 = x.detach().requires_grad_(True)
    y_spatial_only = fsbg.spatial_edge(x2)
    y_phase_only = fsbg._phase_boundary(x2)

    loss_s = (y_spatial_only ** 2).mean()
    loss_s.backward()
    grad_spatial = x2.grad.clone()
    x2.grad.zero_()

    loss_p = (y_phase_only ** 2).mean()
    loss_p.backward()
    grad_phase = x2.grad.clone()

    print(f"spatial branch grad_mag : {_mag(grad_spatial):.2e}")
    print(f"phase   branch grad_mag : {_mag(grad_phase):.2e}")
    ratio = _mag(grad_phase) / (_mag(grad_spatial) + 1e-12)
    print(f"phase / spatial ratio   : {ratio:.3f}")

    if ratio < 0.01:
        print("WARNING: phase branch gradient is negligible compared to spatial branch!")
    if not has_grad:
        print("WARNING: alpha parameter has no gradient - FSBG is not learnable!")

    return fsbg


if __name__ == "__main__":
    for mask in ["soft", "hard", "none"]:
        probe_fsbg_unit(channels=256, h=14, w=14, alpha_init=1e-2, phase_mask=mask)
