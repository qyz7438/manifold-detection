import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.detection.pah import ResidualPrototypeHead


def test_pah_shape():
    m = ResidualPrototypeHead(1024, num_classes=2, num_background_prototypes=4)
    x = torch.randn(8, 1024)
    logits, bbox = m(x)
    assert logits.shape == (8, 2)
    assert bbox.shape == (8, 8)


def test_pah_residual_at_init():
    # gamma_init=0 -> cls_logits equals standard logits.
    m = ResidualPrototypeHead(1024, num_classes=2, gamma_init=0.0)
    x = torch.randn(8, 1024)
    logits, _ = m(x)
    standard_logits = m.cls_score(x)
    assert torch.allclose(logits, standard_logits, atol=1e-6)


def test_pah_initialises_from_old_predictor():
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    old = FastRCNNPredictor(1024, 2)
    m = ResidualPrototypeHead(1024, num_classes=2, old_predictor=old, gamma_init=0.0)
    assert torch.allclose(m.cls_score.weight, old.cls_score.weight, atol=1e-6)
    assert torch.allclose(m.bbox_pred.weight, old.bbox_pred.weight, atol=1e-6)

    x = torch.randn(4, 1024)
    logits, bbox = m(x)
    assert logits.shape == (4, 2)
    assert bbox.shape == (4, 8)


def test_pah_learnable_temperature():
    m = ResidualPrototypeHead(1024, num_classes=2, learnable_temp=True)
    assert isinstance(m.logit_scale, nn.Parameter)
