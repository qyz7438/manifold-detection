"""Small action adapters for semantic segmentation post-training.

Each adapter is zero-initialized so that policy == baseline at the start of
post-training.  Only the adapter parameters are trainable; the baseline FCN is
frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitCorrectionAdapter(nn.Module):
    """Outputs a per-pixel logit delta given backbone features."""

    def __init__(
        self,
        in_ch: int = 2048,
        num_classes: int = 2,
        hidden_ch: int = 64,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, num_classes, kernel_size=1, bias=True),
        )
        # Zero-init last layer so policy == baseline at start.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FeatureRefinementAdapter(nn.Module):
    """Outputs a residual feature map to be added to backbone features."""

    def __init__(
        self,
        in_ch: int = 2048,
        hidden_ch: int = 64,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_ch, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CalibrationAdapter(nn.Module):
    """Per-class temperature and bias for logit calibration.

    Temperature is parameterized as exp(log_temp) so it stays positive.
    Initialized to identity: temp=1, bias=0.
    """

    def __init__(self, num_classes: int = 2, init_temp: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.log_temp = nn.Parameter(torch.full((num_classes,), float(init_temp)).log())
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); temp/bias: (C,)
        temp = self.log_temp.exp().view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return logits / temp + bias


class SegActionPolicy(nn.Module):
    """Wraps a frozen FCN baseline and a trainable action adapter.

    Returns both the policy logits and (optionally) the backbone features needed
    for NC/flatness diagnostics and feature-level rewards.
    """

    def __init__(
        self,
        base_model: nn.Module,
        action_type: str,
        num_classes: int = 2,
        in_ch: int = 2048,
        adapter_kwargs: dict | None = None,
    ):
        super().__init__()
        if action_type not in {"logit", "feature", "calibration", "none"}:
            raise ValueError(f"Unknown action_type: {action_type}")

        self.base_model = base_model
        self.action_type = action_type
        self.num_classes = num_classes

        # Freeze baseline.
        for p in self.base_model.parameters():
            p.requires_grad = False

        adapter_kwargs = adapter_kwargs or {}
        if action_type == "logit":
            self.adapter = LogitCorrectionAdapter(
                in_ch=in_ch, num_classes=num_classes, **adapter_kwargs
            )
        elif action_type == "feature":
            self.adapter = FeatureRefinementAdapter(in_ch=in_ch, **adapter_kwargs)
        elif action_type == "calibration":
            self.adapter = CalibrationAdapter(num_classes=num_classes, **adapter_kwargs)
        elif action_type == "none":
            self.adapter = nn.Identity()

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> dict[str, torch.Tensor]:
        features = self.base_model.backbone(x)["out"]

        if self.action_type == "logit":
            with torch.no_grad():
                base_logits_low = self.base_model.classifier(features)
            delta = self.adapter(features)
            policy_logits_low = base_logits_low + self.adapter.scale * delta
            policy_logits = F.interpolate(
                policy_logits_low, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        elif self.action_type == "feature":
            delta = self.adapter(features)
            refined = features + self.adapter.scale * delta
            policy_logits = self.base_model.classifier(refined)
            policy_logits = F.interpolate(
                policy_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        elif self.action_type == "calibration":
            with torch.no_grad():
                base_logits = self.base_model.classifier(features)
                base_logits = F.interpolate(
                    base_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            policy_logits = self.adapter(base_logits)

        elif self.action_type == "none":
            policy_logits = self.base_model.classifier(features)
            policy_logits = F.interpolate(
                policy_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        out: dict[str, torch.Tensor] = {"out": policy_logits}
        if return_features:
            out["features"] = features
        return out

    def baseline_forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> dict[str, torch.Tensor]:
        """Run the frozen baseline without the adapter."""
        with torch.no_grad():
            features = self.base_model.backbone(x)["out"]
            logits = self.base_model.classifier(features)
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        out: dict[str, torch.Tensor] = {"out": logits}
        if return_features:
            out["features"] = features
        return out


def install_seg_adapter(
    base_model: nn.Module,
    action_type: str,
    num_classes: int = 2,
    in_ch: int = 2048,
    adapter_kwargs: dict | None = None,
) -> SegActionPolicy:
    """Wrap a torchvision FCN-style model with a trainable action adapter."""
    return SegActionPolicy(
        base_model=base_model,
        action_type=action_type,
        num_classes=num_classes,
        in_ch=in_ch,
        adapter_kwargs=adapter_kwargs,
    )
