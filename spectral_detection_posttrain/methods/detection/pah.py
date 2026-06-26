import torch.nn as nn


class PrototypeAwareHead(nn.Module):
    """No-op stub for the Prototype-Aware Head."""

    def __init__(self, in_features: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, z):
        raise NotImplementedError("PAH stub: implement classification/regression")
