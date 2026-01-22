from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import AverageMeter


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_fn: nn.Module | None = None,
) -> Dict[str, float]:
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise ValueError("Loss is not finite. Check your data/model.")

        loss.backward()
        optimizer.step()

        batch_size = int(x.size(0))
        loss_meter = loss_meter.update(loss.item(), n=batch_size)
        acc_meter = acc_meter.update(accuracy_top1(logits.detach(), y), n=batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_fn: nn.Module | None = None,
) -> Dict[str, float]:
    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        batch_size = int(x.size(0))
        loss_meter = loss_meter.update(loss.item(), n=batch_size)
        acc_meter = acc_meter.update(accuracy_top1(logits, y), n=batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}

