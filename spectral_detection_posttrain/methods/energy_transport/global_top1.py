"""Image-level listwise no-op versus one bounded detector action."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.set_policy import NMSAwareSetPolicyHead


@dataclass(frozen=True)
class GlobalTop1Output:
    action_logits: torch.Tensor
    noop_logit: torch.Tensor
    conflict_stats: torch.Tensor


@dataclass(frozen=True)
class FlattenedGlobalLogits:
    logits: torch.Tensor
    proposal_indices: torch.Tensor
    candidate_indices: torch.Tensor


@dataclass(frozen=True)
class GlobalTop1Target:
    is_noop: bool
    proposal_index: int
    candidate_index: int
    delta_u: float


@dataclass(frozen=True)
class GlobalTop1Selection:
    is_noop: bool
    proposal_index: int
    candidate_index: int
    selection_logit: float
    box_delta: torch.Tensor


class GlobalTop1PolicyHead(nn.Module):
    """Score one global no-op against every observable proposal-action pair."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        candidate_deltas: torch.Tensor,
        hidden_dim: int = 96,
        spatial_size: int = 7,
        energy_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.proposal_policy = NMSAwareSetPolicyHead(
            in_channels=in_channels,
            num_classes=num_classes,
            candidate_deltas=candidate_deltas,
            hidden_dim=hidden_dim,
            spatial_size=spatial_size,
            energy_weight=energy_weight,
        )
        for parameter in self.proposal_policy.move_head.parameters():
            parameter.requires_grad_(False)
        self.noop_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        spatial_features: torch.Tensor,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        image_size: tuple[int, int] | torch.Tensor,
        observable_mask: torch.Tensor,
    ) -> GlobalTop1Output:
        output = self.proposal_policy(
            spatial_features,
            class_logits,
            labels,
            scores,
            boxes,
            image_size,
        )
        if observable_mask.shape != (output.action_logits.shape[0],):
            raise ValueError("observable_mask must have shape (N,)")
        observable_noop = output.action_logits[
            observable_mask.to(device=output.action_logits.device).bool(), 0
        ]
        if observable_noop.numel():
            noop_logit = torch.logsumexp(observable_noop, dim=0) - math.log(observable_noop.numel())
            noop_logit = noop_logit + self.noop_bias
        else:
            noop_logit = self.noop_bias
        return GlobalTop1Output(
            action_logits=output.action_logits,
            noop_logit=noop_logit,
            conflict_stats=output.conflict_stats,
        )


def flatten_observable_action_logits(
    output: GlobalTop1Output,
    observable_mask: torch.Tensor,
) -> FlattenedGlobalLogits:
    if output.action_logits.ndim != 2 or output.action_logits.shape[1] < 2:
        raise ValueError("action_logits must have shape (N, K) with K >= 2")
    count, candidates = output.action_logits.shape
    if observable_mask.shape != (count,):
        raise ValueError("observable_mask must have shape (N,)")
    rows = torch.nonzero(
        observable_mask.to(device=output.action_logits.device).bool(), as_tuple=False
    ).flatten()
    proposal_indices = torch.cat(
        (
            torch.full((1,), -1, dtype=torch.long, device=rows.device),
            rows.repeat_interleave(candidates - 1),
        )
    )
    candidate_indices = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=rows.device),
            torch.arange(1, candidates, device=rows.device).repeat(rows.numel()),
        )
    )
    logits = torch.cat((output.noop_logit.reshape(1), output.action_logits[rows, 1:].reshape(-1)))
    return FlattenedGlobalLogits(logits, proposal_indices, candidate_indices)


def build_global_top1_target(
    delta_u: torch.Tensor,
    observable_mask: torch.Tensor,
    min_delta_u: float = 0.0,
) -> GlobalTop1Target:
    if delta_u.ndim != 2 or delta_u.shape[1] < 2:
        raise ValueError("delta_u must have shape (N, K) with K >= 2")
    if observable_mask.shape != (delta_u.shape[0],):
        raise ValueError("observable_mask must have shape (N,)")
    if not math.isfinite(float(min_delta_u)):
        raise ValueError("min_delta_u must be finite")
    rows = torch.nonzero(observable_mask.to(device=delta_u.device).bool(), as_tuple=False).flatten()
    if rows.numel() == 0:
        return GlobalTop1Target(True, -1, 0, 0.0)
    utilities = delta_u[rows, 1:]
    if not torch.isfinite(utilities).all():
        raise ValueError("observable non-noop delta_u values must be finite")
    flat_index = int(utilities.reshape(-1).argmax().item())
    candidate_count = utilities.shape[1]
    row_offset, candidate_offset = divmod(flat_index, candidate_count)
    best_delta = float(utilities[row_offset, candidate_offset].item())
    if best_delta <= float(min_delta_u):
        return GlobalTop1Target(True, -1, 0, 0.0)
    return GlobalTop1Target(
        False,
        int(rows[row_offset].item()),
        int(candidate_offset + 1),
        best_delta,
    )


def global_top1_loss(
    output: GlobalTop1Output,
    observable_mask: torch.Tensor,
    target: GlobalTop1Target,
) -> dict[str, torch.Tensor]:
    flattened = flatten_observable_action_logits(output, observable_mask)
    if target.is_noop:
        target_index = 0
    else:
        matches = flattened.proposal_indices.eq(target.proposal_index) & flattened.candidate_indices.eq(
            target.candidate_index
        )
        if int(matches.sum().item()) != 1:
            raise ValueError("target action is not in the detector-observable candidate set")
        target_index = int(torch.nonzero(matches, as_tuple=False).flatten()[0].item())
    target_tensor = torch.tensor([target_index], dtype=torch.long, device=flattened.logits.device)
    loss = F.cross_entropy(flattened.logits[None, :], target_tensor)
    prediction = int(flattened.logits.argmax().item())
    return {
        "loss_total": loss,
        "accuracy": loss.detach().new_tensor(float(prediction == target_index)),
        "predicted_noop": loss.detach().new_tensor(float(prediction == 0)),
    }


def select_global_top1_action(
    output: GlobalTop1Output,
    candidate_deltas: torch.Tensor,
    observable_mask: torch.Tensor,
) -> GlobalTop1Selection:
    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4:
        raise ValueError("candidate_deltas must have shape (K, 4)")
    if candidate_deltas.shape[0] != output.action_logits.shape[1]:
        raise ValueError("candidate_deltas must match action_logits")
    if candidate_deltas[0].count_nonzero().item() != 0:
        raise ValueError("candidate index 0 must be no-op")
    flattened = flatten_observable_action_logits(output, observable_mask)
    selected_index = int(flattened.logits.argmax().item())
    proposal_index = int(flattened.proposal_indices[selected_index].item())
    candidate_index = int(flattened.candidate_indices[selected_index].item())
    box_delta = output.action_logits.new_zeros((output.action_logits.shape[0], 4))
    is_noop = selected_index == 0
    if not is_noop:
        box_delta[proposal_index] = candidate_deltas[candidate_index].to(
            device=box_delta.device, dtype=box_delta.dtype
        )
    return GlobalTop1Selection(
        is_noop=is_noop,
        proposal_index=proposal_index,
        candidate_index=candidate_index,
        selection_logit=float(flattened.logits[selected_index].detach().item()),
        box_delta=box_delta,
    )
