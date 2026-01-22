import pytest
import torch
from torchvision import datasets

from src.model import build_model
from src.utils import set_seed


class FakeCIFAR10(datasets.FakeData):
    """Drop-in replacement for CIFAR10 for testing (no download needed)."""
    def __init__(self, root, train, download, transform=None):
        size = 64 if train else 32
        super().__init__(
            size=size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture()
def model(device):
    set_seed(0)
    return build_model(num_classes=10).to(device)
