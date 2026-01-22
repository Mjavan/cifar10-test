import pytest
import torch
from torchvision import datasets, transforms


@pytest.fixture(scope="session")
def cifar_transform():
    # Keep it deterministic & simple for unit tests
    return transforms.Compose([
        transforms.ToTensor(),  # -> float32 in [0, 1], shape [C,H,W]
    ])


def test_cifar10_sample_shape_dtype_range(cifar_transform):
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=cifar_transform)

    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (3, 32, 32)  # CIFAR-10 is 32x32 RGB

    # Range check: ToTensor scales uint8 [0..255] to float [0..1]
    assert torch.isfinite(x).all()
    assert 0.0 <= float(x.min()) <= 1.0
    assert 0.0 <= float(x.max()) <= 1.0

    assert isinstance(y, int)
    assert 0 <= y <= 9


def test_cifar10_len_nonzero(cifar_transform):
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=cifar_transform)
    assert len(ds) == 50_000
