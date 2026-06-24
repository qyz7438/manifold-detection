r"""Low-dimensional-preserving ROI box heads for Faster R-CNN.

The default torchvision ``TwoMLPHead`` flattens the 256x7x7 ROI feature into a
12544-dim vector and then compresses it back to 1024 via two fully-connected
layers.  This expansion can scatter the compact manifold structure that exists
in the spatial ROI map.  The heads below avoid the large flatten expansion and
keep the representation low-dimensional throughout the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.core.models.bottleneck_box_head import (
    BottleneckTwoMLPHead,
)


class ConvLowDimBoxHead(nn.Module):
    """Spatial conv reduction + adaptive pooling + small MLP bottleneck.

    Path: 256 x 7 x 7 -> conv -> 128 x 7 x 7 -> global avg pool -> 128 -> MLP -> 1024
    """

    def __init__(
        self,
        in_channels: int = 256,
        representation_size: int = 1024,
        conv_channels: int = 128,
        bottleneck_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.representation_size = representation_size
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            nn.Linear(conv_channels, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, representation_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N x C x 7 x 7
        x = self.conv_reduce(x)        # N x conv_channels x 7 x 7
        x = self.global_pool(x)        # N x conv_channels x 1 x 1
        x = x.flatten(start_dim=1)     # N x conv_channels
        return self.bottleneck(x)      # N x representation_size


class BottleneckBoxHead(nn.Module):
    """Two-MLP head with a low-rank bottleneck in the middle.

    Path: 256*7*7 -> fc6 -> 1024 -> down -> rank -> up -> 1024 (+ optional skip)

    The bottleneck forces the head to pass information through a low-rank
    subspace, which can help preserve the compact manifold structure of the
    ROI feature.
    """

    def __init__(
        self,
        in_channels: int = 256,
        representation_size: int = 1024,
        rank: int = 128,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.representation_size = representation_size
        self.rank = rank
        self.use_skip = use_skip
        self.fc6 = nn.Linear(in_channels * 7 * 7, representation_size)
        self.bottleneck_down = nn.Linear(representation_size, rank)
        self.bottleneck_up = nn.Linear(rank, representation_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        h = F.relu(self.fc6(x))
        z = F.relu(self.bottleneck_down(h))
        out = self.bottleneck_up(z)
        if self.use_skip:
            out = out + h
        return F.relu(out)

    def init_from_twomlp(self, twomlp_head: nn.Module) -> None:
        """Copy compatible weights from a torchvision TwoMLPHead.

        Only ``fc6`` can be copied directly; the bottleneck is randomly
        initialized.  The skip connection keeps the initial output close to the
        original head.
        """
        source_fc6 = getattr(twomlp_head, "fc6", None)
        source_fc7 = getattr(twomlp_head, "fc7", None)
        if source_fc6 is None:
            inner = getattr(twomlp_head, "head", None)
            if inner is not None:
                source_fc6 = getattr(inner, "fc6", None)
                source_fc7 = getattr(inner, "fc7", None)
        if source_fc6 is not None:
            self.fc6.weight.data.copy_(source_fc6.weight.data)
            self.fc6.bias.data.copy_(source_fc6.bias.data)
        # If fc7 exists, initialize bottleneck_up to approximate fc7 via SVD.
        if source_fc7 is not None and self.rank < self.representation_size:
            with torch.no_grad():
                u, s, vh = torch.linalg.svd(source_fc7.weight.data, full_matrices=False)
                r = min(self.rank, s.shape[0])
                self.bottleneck_down.weight.data.copy_(vh[:r, :])
                self.bottleneck_down.bias.data.zero_()
                self.bottleneck_up.weight.data.copy_(
                    u[:, :r] @ torch.diag(s[:r])
                )
                self.bottleneck_up.bias.data.copy_(source_fc7.bias.data)


class AttentionPoolBoxHead(nn.Module):
    """Learned spatial attention pooling over the ROI map.

    Path: 256 x 7 x 7 -> spatial attention -> weighted sum over spatial locations
          -> 256 -> linear -> 1024

    The attention map focuses on informative spatial locations without
    flattening the whole ROI map.
    """

    def __init__(
        self,
        in_channels: int = 256,
        representation_size: int = 1024,
        attention_channels: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.representation_size = representation_size
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, attention_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, 1, 1),
        )
        self.proj = nn.Linear(in_channels, representation_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N x C x H x W
        n, c, h, w = x.shape
        attn = self.attention(x)               # N x 1 x H x W
        attn = attn.view(n, -1)
        attn = F.softmax(attn, dim=-1)         # spatial softmax
        x_view = x.view(n, c, h * w)           # N x C x HW
        pooled = (x_view * attn.unsqueeze(1)).sum(dim=-1)  # N x C
        return F.relu(self.proj(pooled))


def replace_box_head(
    model: nn.Module,
    head_type: str,
    *,
    representation_size: int | None = None,
    rank: int = 128,
    conv_channels: int = 128,
    bottleneck_dim: int = 512,
    attention_channels: int = 64,
    copy_compatible_weights: bool = True,
) -> nn.Module:
    """Replace ``model.roi_heads.box_head`` with a low-dim-preserving variant.

    Args:
        model: a Faster R-CNN style detector.
        head_type: one of ``"original"``, ``"conv_lowdim"``, ``"bottleneck"``,
            ``"attention_pool"``.
        representation_size: output dim of the new head.  If None, inferred from
            the existing box_predictor input dim.
        rank: rank for ``bottleneck`` head.
        conv_channels: channels for ``conv_lowdim`` head.
        bottleneck_dim: hidden dim for ``conv_lowdim`` head.
        attention_channels: intermediate channels for ``attention_pool`` head.
        copy_compatible_weights: for ``bottleneck``, try to initialize fc6 and
            the bottleneck from the original TwoMLPHead via SVD.

    Returns:
        The modified model.
    """
    head_type = head_type.replace("-", "_")
    if head_type == "original":
        return model

    if representation_size is None:
        representation_size = int(model.roi_heads.box_predictor.cls_score.in_features)

    # Try to infer ROI map channel count from the first conv/pool layer.
    original_head = model.roi_heads.box_head
    in_channels = 256  # default for Faster R-CNN FPN
    if hasattr(original_head, "fc6"):
        in_channels = original_head.fc6.in_features // (7 * 7)
    else:
        inner = getattr(original_head, "head", None)
        if inner is not None and hasattr(inner, "fc6"):
            in_channels = inner.fc6.in_features // (7 * 7)

    if head_type == "conv_lowdim":
        new_head = ConvLowDimBoxHead(
            in_channels=in_channels,
            representation_size=representation_size,
            conv_channels=conv_channels,
            bottleneck_dim=bottleneck_dim,
        )
    elif head_type == "bottleneck":
        new_head = BottleneckBoxHead(
            in_channels=in_channels,
            representation_size=representation_size,
            rank=rank,
            use_skip=True,
        )
        if copy_compatible_weights:
            new_head.init_from_twomlp(original_head)
    elif head_type == "bottleneck_twomlp":
        new_head = BottleneckTwoMLPHead(
            in_channels=in_channels,
            bottleneck_channels=conv_channels,
            representation_size=representation_size,
            grid_size=7,
        )
    elif head_type == "attention_pool":
        new_head = AttentionPoolBoxHead(
            in_channels=in_channels,
            representation_size=representation_size,
            attention_channels=attention_channels,
        )
    else:
        raise ValueError(f"Unknown box_head type: {head_type}")

    device = next(original_head.parameters()).device
    dtype = next(original_head.parameters()).dtype
    new_head = new_head.to(device=device, dtype=dtype)
    model.roi_heads.box_head = new_head
    return model


def get_box_head_type(model: nn.Module) -> str:
    """Return a string tag for the current box_head class."""
    box_head_tags = {
        ConvLowDimBoxHead: "conv_lowdim",
        BottleneckBoxHead: "bottleneck",
        BottleneckTwoMLPHead: "bottleneck_twomlp",
        AttentionPoolBoxHead: "attention_pool",
    }
    head = model.roi_heads.box_head
    for cls, tag in box_head_tags.items():
        if isinstance(head, cls):
            return tag
    return "original"
