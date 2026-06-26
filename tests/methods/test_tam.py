import torch

from spectral_detection_posttrain.methods.detection.tam import TaskAlignedManifold


def test_tam_identity_at_init():
    m = TaskAlignedManifold(1024, latent_dim=256)
    x = torch.randn(4, 1024)
    y = m(x)
    assert y.shape == x.shape
    # Decoder is zero-initialised and scale starts at 0, so output equals input.
    assert torch.allclose(y, x, atol=1e-6)


def test_tam_spectral_quality_loss():
    m = TaskAlignedManifold(1024, latent_dim=64, use_spectral_quality=True)
    x = torch.randn(4, 1024)
    roi = torch.randn(4, 256, 7, 7)
    y = m(x, roi_feature=roi)
    loss = m.get_aux_loss()
    assert y.shape == x.shape
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_tam_contrastive_loss():
    m = TaskAlignedManifold(1024, latent_dim=64, use_contrastive=True, num_classes=2)
    x = torch.randn(4, 1024)
    labels = torch.tensor([0, 1, 1, 0])
    y = m(x, labels=labels)
    loss = m.get_aux_loss()
    assert y.shape == x.shape
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
