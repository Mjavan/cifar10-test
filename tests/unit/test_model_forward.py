import torch


def test_model_forward_shape():
    from src.model import build_model

    model = build_model(num_classes=10)
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()
