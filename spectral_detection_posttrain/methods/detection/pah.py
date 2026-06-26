import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAwareHead(nn.Module):
    """Prototype-based classification + regression head for Faster R-CNN.

    Classification is performed by cosine similarity between the L2-normalized
    embedding and learned class prototypes (including background).  Regression
    uses a small per-class linear branch.
    """

    def __init__(self, in_features: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

    def forward(self, x: torch.Tensor):
        x = F.normalize(x, p=2, dim=1)
        protos = F.normalize(self.prototypes, p=2, dim=1)
        cls_score = (x @ protos.t()) / self.temperature
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred
