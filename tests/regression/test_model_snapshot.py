import torch
from pathlib import Path

from src.model import build_model
from src.utils import set_seed


def test_model_logits_snapshot():
    assets = Path(__file__).resolve().parents[1] / "assets"
    x = torch.load(assets / "snapshot_input.pt", map_location="cpu")
    expected_logits = torch.load(assets / "snapshot_logits.pt", map_location="cpu")

    set_seed(0)
    model = build_model(num_classes=10)
    model.eval()

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == expected_logits.shape
    assert torch.allclose(logits, expected_logits, atol=1e-6, rtol=0)
