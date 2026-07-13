"""Global post-NMS suppress/no-op policy over the native kept set."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.energy_transport.policy.global_top1 import GlobalTop1Output
from spectral_detection_posttrain.methods.energy_transport.policy.set_policy import class_aware_conflict_statistics


@dataclass(frozen=True)
class PostNMSSuppression:
    is_noop: bool
    detection_index: int
    selection_logit: float


def build_post_nms_detection_features(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    *,
    num_classes: int = 11,
) -> torch.Tensor:
    """Build detector-only node and kept-set relation features."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    count = boxes.shape[0]
    if scores.shape != (count,) or labels.shape != (count,):
        raise ValueError("scores and labels must have shape (N,)")
    if num_classes <= 1 or (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError("labels must fit num_classes")
    size = torch.as_tensor(image_size, dtype=boxes.dtype, device=boxes.device).flatten()
    if size.numel() != 2 or (size <= 0).any():
        raise ValueError("image_size must contain positive height and width")
    height, width = size[0], size[1]
    scale = torch.stack((width, height, width, height))
    normalized_boxes = boxes / scale.clamp_min(1.0)
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6) / width.clamp_min(1.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6) / height.clamp_min(1.0)
    log_area = torch.log((widths * heights).clamp_min(1e-8))[:, None]
    log_aspect = torch.log((widths / heights).clamp_min(1e-8))[:, None]
    label_code = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=boxes.dtype)
    conflicts = class_aware_conflict_statistics(boxes, labels, scores, image_size)
    return torch.cat((scores[:, None].to(boxes.dtype), normalized_boxes, log_area, log_aspect, label_code, conflicts), dim=1)


class PostNMSSuppressPolicyHead(nn.Module):
    """Permutation-equivariant image-level suppress/no-op scorer."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.local_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.noop_head = nn.Linear(2 * self.hidden_dim, 1)
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)
        nn.init.zeros_(self.noop_head.weight)
        nn.init.zeros_(self.noop_head.bias)

    def forward(self, features: torch.Tensor, observable_mask: torch.Tensor) -> GlobalTop1Output:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape (N, input_dim)")
        count = features.shape[0]
        if observable_mask.shape != (count,):
            raise ValueError("observable_mask must have shape (N,)")
        local = self.local_encoder(features)
        visible = observable_mask.to(device=features.device).bool()
        if visible.any():
            set_mean = local[visible].mean(dim=0)
            set_max = local[visible].max(dim=0).values
        else:
            set_mean = local.new_zeros((self.hidden_dim,))
            set_max = local.new_zeros((self.hidden_dim,))
        action_context = torch.cat(
            (
                local,
                set_mean[None, :].expand(count, -1),
                set_max[None, :].expand(count, -1),
            ),
            dim=1,
        )
        suppress_logits = self.action_head(action_context).squeeze(1)
        suppress_logits = torch.where(visible, suppress_logits, suppress_logits.new_full((count,), -1e9))
        action_logits = torch.stack((suppress_logits.new_zeros((count,)), suppress_logits), dim=1)
        noop_logit = self.noop_head(torch.cat((set_mean, set_max), dim=0)).squeeze()
        return GlobalTop1Output(action_logits, noop_logit, features.new_zeros((count, 4)))


def select_post_nms_suppression(
    output: GlobalTop1Output,
    observable_mask: torch.Tensor,
    *,
    allow_noop: bool = True,
) -> PostNMSSuppression:
    count = output.action_logits.shape[0]
    if output.action_logits.shape != (count, 2) or observable_mask.shape != (count,):
        raise ValueError("output and observable_mask must align")
    rows = torch.nonzero(observable_mask.to(device=output.action_logits.device).bool(), as_tuple=False).flatten()
    if rows.numel() == 0:
        return PostNMSSuppression(True, -1, float(output.noop_logit.item()))
    logits = output.action_logits[rows, 1]
    position = int(logits.argmax().item())
    row = int(rows[position].item())
    value = float(logits[position].item())
    if allow_noop and float(output.noop_logit.item()) >= value:
        return PostNMSSuppression(True, -1, float(output.noop_logit.item()))
    return PostNMSSuppression(False, row, value)


def suppress_detection(prediction: dict[str, torch.Tensor], detection_index: int) -> dict[str, torch.Tensor]:
    count = int(prediction["boxes"].shape[0])
    if detection_index < 0 or detection_index >= count:
        raise IndexError("detection_index out of range")
    keep = torch.ones(count, dtype=torch.bool, device=prediction["boxes"].device)
    keep[detection_index] = False
    return {
        key: value[keep] if torch.is_tensor(value) and value.shape[:1] == (count,) else value
        for key, value in prediction.items()
    }
