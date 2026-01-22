from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make runs reproducible (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism settings (may have perf impact)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: str | os.PathLike,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | os.PathLike,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return {
        "epoch": int(payload.get("epoch", 0)),
        "step": int(payload.get("step", 0)),
        "extra": payload.get("extra", {}),
    }


@dataclass(frozen=True)
class AverageMeter:
    """Tiny helper for tracking mean over batches."""
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> "AverageMeter":
        return AverageMeter(self.total + float(value) * int(n), self.count + int(n))

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
