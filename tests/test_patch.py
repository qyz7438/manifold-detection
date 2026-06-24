import torch

from mfvpt.transforms.patch import add_patch, add_random_patch


def test_random_patch_changes_copy_without_mutating_input():
    torch.manual_seed(0)
    x = torch.zeros(2, 3, 32, 32)
    original = x.clone()
    out = add_random_patch(x, patch_size=8)
    assert out.shape == x.shape
    assert torch.equal(x, original)
    assert not torch.equal(out, original)


def test_patch_types_run():
    torch.manual_seed(0)
    x = torch.zeros(2, 3, 32, 32)
    for patch_type in ["random", "checkerboard", "qr_like"]:
        out = add_patch(x, patch_type=patch_type, patch_size=8)
        assert out.shape == x.shape
        assert not torch.equal(out, x)


def test_invalid_patch_size_raises():
    x = torch.zeros(1, 3, 16, 16)
    try:
        add_random_patch(x, patch_size=16)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
