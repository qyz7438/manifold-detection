from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualPrototypeHead(nn.Module):
    """Residual prototype-based classification head for Faster R-CNN.

    Instead of replacing the pretrained predictor, this head keeps the original
    classification layer and adds a prototype-based residual term:

        cls_logits = standard_logits + gamma * proto_logits

    This makes the module safe to insert into a pretrained detector: when
    ``gamma`` is initialised near zero the behaviour is close to the original
    predictor, and the prototype term is learned gradually.

    Foreground prototypes are initialised from the pretrained classifier
    weights (L2-normalised).  Background is represented by multiple prototypes
    because "background" in detection is not a single semantic class.

    Args:
        in_features: dimension of the box-head output feature.
        num_classes: number of detection classes including background.
        temperature: initial temperature for cosine similarity.
        learnable_temp: whether to learn the log-temperature.
        num_background_prototypes: number of background prototypes.
        gamma_init: initial scale of the prototype residual term.
        old_predictor: pretrained ``FastRCNNPredictor`` used to initialise
            the standard layer and the prototypes.  If ``None``, random
            initialisation is used.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        temperature: float = 0.3,
        learnable_temp: bool = False,
        num_background_prototypes: int = 4,
        gamma_init: float = 0.0,
        old_predictor: nn.Module | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_background_prototypes = num_background_prototypes

        # Standard classifier and bbox regressor.  When a pretrained predictor
        # is supplied we copy its weights; otherwise we initialise from scratch.
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

        if old_predictor is not None:
            with torch.no_grad():
                self.cls_score.weight.copy_(old_predictor.cls_score.weight)
                self.cls_score.bias.copy_(old_predictor.cls_score.bias)
                self.bbox_pred.weight.copy_(old_predictor.bbox_pred.weight)
                self.bbox_pred.bias.copy_(old_predictor.bbox_pred.bias)

        # Foreground prototypes: one per foreground class.
        num_fg = num_classes - 1
        self.fg_prototypes = nn.Parameter(torch.randn(num_fg, in_features))

        # Background prototypes: multiple vectors to capture diverse background.
        self.bg_prototypes = nn.Parameter(torch.randn(num_background_prototypes, in_features))

        if old_predictor is not None and num_fg > 0:
            # Initialise foreground prototypes from the pretrained classifier
            # weights of the foreground classes (class indices 1..C-1).
            with torch.no_grad():
                old_weight = old_predictor.cls_score.weight  # (num_classes, in_features)
                fg_weight = old_weight[1:num_classes]  # (num_fg, in_features)
                self.fg_prototypes.copy_(F.normalize(fg_weight, p=2, dim=1))

        nn.init.xavier_uniform_(self.bg_prototypes)

        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / max(temperature, 1e-6)))
        else:
            self.register_buffer("logit_scale", torch.ones([]) * math.log(1.0 / max(temperature, 1e-6)))

        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def _proto_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute prototype-based logits.

        Foreground classes use their dedicated prototypes.  Background score is
        aggregated from multiple background prototypes via logsumexp.
        """
        x_norm = F.normalize(x, p=2, dim=1)
        fg_norm = F.normalize(self.fg_prototypes, p=2, dim=1)
        bg_norm = F.normalize(self.bg_prototypes, p=2, dim=1)

        scale = torch.exp(self.logit_scale).clamp_max(100.0)

        # Foreground similarities: (B, num_fg)
        fg_sim = torch.matmul(x_norm, fg_norm.t()) * scale

        # Background similarity: aggregate multiple prototypes.
        bg_sim = torch.matmul(x_norm, bg_norm.t()) * scale  # (B, K)
        bg_score = torch.logsumexp(bg_sim, dim=1, keepdim=True)  # (B, 1)

        # Class 0 is background, classes 1..C-1 are foreground.
        proto_logits = torch.cat([bg_score, fg_sim], dim=1)
        return proto_logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        standard_logits = self.cls_score(x)
        proto_logits = self._proto_logits(x)
        cls_logits = standard_logits + self.gamma * proto_logits
        bbox_deltas = self.bbox_pred(x)
        return cls_logits, bbox_deltas


# Backward-compatible alias for legacy code / configs.
PrototypeAwareHead = ResidualPrototypeHead
