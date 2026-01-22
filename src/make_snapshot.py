import torch
from pathlib import Path

from src.model import build_model
from src.utils import set_seed

def main():
    set_seed(0)
    model = build_model(num_classes=10)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    out_dir = Path("tests/assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(x, out_dir / "snapshot_input.pt")
    torch.save(logits, out_dir / "snapshot_logits.pt")
    print("Saved:", out_dir / "snapshot_input.pt", "and", out_dir / "snapshot_logits.pt")

if __name__ == "__main__":
    main()
