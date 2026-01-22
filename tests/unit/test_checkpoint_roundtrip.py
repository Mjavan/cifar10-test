import torch

def test_checkpoint_roundtrip(tmp_path):
    from src.model import build_model
    from src.utils import save_checkpoint, load_checkpoint

    torch.manual_seed(0)
    model = build_model(num_classes=10)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model=model, optimizer=optim, step=123)

    model2 = build_model(num_classes=10)
    optim2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    meta = load_checkpoint(ckpt_path, model=model2, optimizer=optim2)

    assert meta["step"] == 123

    # Compare a forward pass output
    x = torch.randn(2, 3, 32, 32)
    model.eval(); model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2, atol=0, rtol=0)
