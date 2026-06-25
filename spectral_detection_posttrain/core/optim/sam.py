r"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

SAM (Foret et al., 2020) replaces a single gradient step with a two-step
procedure that seeks a flat minimum:

1. Compute the gradient at the current weights.
2. Take a small adversarial step in the direction of that gradient to a
   nearby "sharp" point.
3. Compute the gradient at the perturbed point and use it for the actual
   parameter update.

This wrapper can be used with any base optimizer (e.g. SGD, AdamW).
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper around a base optimizer.

    Args:
        params: iterable of parameters to optimize.
        base_optimizer: an initialized PyTorch optimizer (e.g. ``SGD``,
            ``AdamW``). SAM will use this optimizer for the actual updates.
        rho: radius of the adversarial perturbation neighborhood.
    """

    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        rho: float = 0.05,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Compute the global gradient norm across all parameter groups."""
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(2))
        return torch.norm(torch.stack(norms), 2) if norms else torch.tensor(0.0)

    @torch.no_grad()
    def _first_step(self) -> None:
        """Apply the adversarial weight perturbation."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + eps"

    @torch.no_grad()
    def _restore_original_params(self, original_params: list[torch.Tensor]) -> None:
        """Restore parameters to their pre-perturbation values."""
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.copy_(original_params[idx])
                idx += 1

    def step(self, closure=None) -> float | None:
        """Perform a SAM optimization step.

        ``closure`` is called twice: first at ``w`` to obtain the perturbation
        direction, then at ``w + eps`` to obtain the update gradient.
        """
        if closure is None:
            raise ValueError("SAM requires a closure")

        closure = torch.enable_grad()(closure)

        # First forward-backward at the original weights.
        loss = closure()

        # Store original parameters before perturbation.
        original_params = [
            p.clone() for group in self.param_groups for p in group["params"]
        ]

        # Perturb weights to the adversarial point.
        self._first_step()

        # Second forward-backward at the perturbed weights.
        loss = closure()

        # Restore original parameters so the base optimizer updates from w,
        # using the gradient computed at w + eps.
        self._restore_original_params(original_params)
        self.base_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)
