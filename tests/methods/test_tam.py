import torch

from spectral_detection_posttrain.methods.detection.tam import TaskAlignedManifold


def test_tam_identity_at_init():
    m = TaskAlignedManifold(1024, 256)
    z = torch.randn(4, 1024)
    out = m(z)
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-6)


def test_tam_gradient_flow():
    m = TaskAlignedManifold(512, 128)
    z = torch.randn(4, 512, requires_grad=True)
    out = m(z)
    loss = out.sum()
    loss.backward()
    assert z.grad is not None
    assert any(p.grad is not None for p in m.parameters())
