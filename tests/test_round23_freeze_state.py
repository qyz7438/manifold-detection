import torch

from spectral_detection_posttrain.models.build_detector import (
    set_detector_eval_except_trainable,
)


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return x


def test_set_detector_eval_except_trainable_keeps_batchnorm_eval():
    model = TinyModule()
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True

    set_detector_eval_except_trainable(model)

    assert model.training is False
    assert model.bn.training is False
    assert model.head.training is True
