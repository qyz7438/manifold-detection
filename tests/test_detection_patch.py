import torch

from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch


def test_detection_patch_changes_image_without_mutating_input():
    torch.manual_seed(0)
    image = torch.zeros(3, 64, 64)
    target = {"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]])}
    original = image.clone()
    patched = add_detection_patch(image, target, placement="object", patch_type="random", patch_size=16)
    assert patched.shape == image.shape
    assert torch.equal(image, original)
    assert not torch.equal(patched, original)


def test_detection_patch_checkerboard_and_qr_like():
    image = torch.zeros(3, 64, 64)
    target = {"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]])}
    for patch_type in ["checkerboard", "qr_like"]:
        patched = add_detection_patch(image, target, placement="edge", patch_type=patch_type, patch_size=16)
        assert patched.shape == image.shape
        assert not torch.equal(patched, image)


def test_object_inside_object_edge_and_near_object_patch_modes_change_image():
    image = torch.zeros((3, 64, 64), dtype=torch.float32)
    target = {"boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]]), "labels": torch.tensor([1])}

    for placement in ["object_inside", "object_edge", "near_object"]:
        patched = add_detection_patch(image, target, placement=placement, patch_type="checkerboard", patch_size=12)
        assert patched.shape == image.shape
        assert torch.sum(torch.abs(patched - image)) > 0
