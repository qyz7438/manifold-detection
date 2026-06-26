import torch

from spectral_detection_posttrain.methods.detection.pah import PrototypeAwareHead


def test_pah_output_shapes():
    head = PrototypeAwareHead(1024, 3)
    x = torch.randn(8, 1024)
    cls, reg = head(x)
    assert cls.shape == (8, 3)
    assert reg.shape == (8, 12)


def test_pah_gradient_flow():
    head = PrototypeAwareHead(512, 2)
    x = torch.randn(4, 512, requires_grad=True)
    cls, reg = head(x)
    loss = cls.sum() + reg.sum()
    loss.backward()
    assert x.grad is not None
    assert head.prototypes.grad is not None
    assert head.bbox_pred.weight.grad is not None
