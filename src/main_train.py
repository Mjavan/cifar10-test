from __future__ import annotations

import torch

# --- path bootstrap (must be at top) ---
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))  # folder that contains src/

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------

from src.data import DataConfig, get_dataloaders
from src.model import build_model
from src.train import TrainConfig, train_one_epoch, evaluate
from src.utils import *


def main() -> None:
    set_seed(0)
    device = get_device(prefer_cuda=True)

    data_cfg = DataConfig(root="data", batch_size=128, num_workers=2)
    train_loader, val_loader, test_loader = get_dataloaders(data_cfg, download=True)

    model = build_model(num_classes=10).to(device)

    train_cfg = TrainConfig(lr=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg.lr,
        momentum=train_cfg.momentum,
        weight_decay=train_cfg.weight_decay,
    )

    for epoch in range(1, 6):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        te = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={tr['loss']:.4f} train_acc={tr['acc']:.3f} "
              f"test_loss={te['loss']:.4f} test_acc={te['acc']:.3f}")

        save_checkpoint(
            f"checkpoints/ckpt_epoch{epoch}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=epoch * len(train_loader),
        )


if __name__ == "__main__":
    main()
