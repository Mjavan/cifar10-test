from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Type
from torch.utils.data import random_split

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    val_fraction: float = 0.1
    seed: int = 42


def build_transforms(train: bool) -> Callable:
    """CIFAR-10 standard-ish transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])


def get_datasets(
    *,
    root: str,
    dataset_cls: Type[torch.utils.data.Dataset] = datasets.CIFAR10,
    download: bool = True,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_ds = dataset_cls(root=root, train=True, download=download, transform=build_transforms(train=True))
    test_ds = dataset_cls(root=root, train=False, download=download, transform=build_transforms(train=False))
    return train_ds, test_ds


def get_dataloaders(
    cfg: DataConfig,
    *,
    dataset_cls: Type[torch.utils.data.Dataset] = datasets.CIFAR10,
    download: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[DataLoader, DataLoader]:
    full_train, test_ds = get_datasets(root=cfg.root, dataset_cls=dataset_cls, download=download)

     # Split train -> train/val
    val_size = int(len(full_train) * cfg.val_fraction)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders(DataConfig())

    print(len(train_loader.dataset))
    print(len(val_loader.dataset))














