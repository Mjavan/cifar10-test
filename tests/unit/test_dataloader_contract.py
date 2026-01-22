import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def test_dataloader_batch_contract():
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    x, y = next(iter(loader))

    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (8, 3, 32, 32)
    assert y.shape == (8,)
    # Labels should be integer type for nn.CrossEntropyLoss
    assert y.dtype in (torch.int64, torch.long)
