from __future__ import annotations

from typing import Literal

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TrainMode = Literal["baseline", "standard_aug", "fourier_aug", "mfvpt_posttrain"]


def _train_transform(image_size: int, train_mode: str) -> transforms.Compose:
    if train_mode == "standard_aug":
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def _val_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


def _limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_cifar100_loaders(
    config: dict,
    train_mode: TrainMode,
    limit_train: int | None = None,
    limit_val: int | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    root = data_cfg.get("root", "./data")
    image_size = int(data_cfg.get("image_size", 224))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_set = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=_train_transform(image_size, train_mode),
    )
    val_set = datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=_val_transform(image_size),
    )
    train_set = _limit_dataset(train_set, limit_train)
    val_set = _limit_dataset(val_set, limit_val)

    bs = int(batch_size or config["train"].get("batch_size", 32))
    return (
        DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
