from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    x = x.to(device)
    logits = model(x)
    return logits


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = predict_logits(model, x, device)
    return logits.argmax(dim=1)


@torch.no_grad()
def predict_proba(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = predict_logits(model, x, device)
    return torch.softmax(logits, dim=1)
