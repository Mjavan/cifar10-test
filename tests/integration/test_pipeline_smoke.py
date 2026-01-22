import torch

from src.data import DataConfig, get_dataloaders
from src.model import build_model
from src.train import train_one_epoch, evaluate

# Reuse FakeCIFAR10 from tests/conftest.py (the one we made earlier)
from conftest import FakeCIFAR10


def test_end_to_end_pipeline_smoke_cpu():
    device = torch.device("cpu")

    cfg = DataConfig(root="data", batch_size=8, num_workers=0, pin_memory=False)
    train_loader,val_loader, test_loader = get_dataloaders(cfg, dataset_cls=FakeCIFAR10, download=False)

    model = build_model(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_metrics = train_one_epoch(model, train_loader, optimizer, device)
    test_metrics = evaluate(model, test_loader, device)

    assert "loss" in train_metrics and "acc" in train_metrics
    assert "loss" in test_metrics and "acc" in test_metrics

    assert 0.0 <= train_metrics["acc"] <= 1.0
    assert 0.0 <= test_metrics["acc"] <= 1.0
    assert train_metrics["loss"] >= 0.0
    assert test_metrics["loss"] >= 0.0