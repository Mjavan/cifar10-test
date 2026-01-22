import torch

from src.model import build_model
from src.utils import save_checkpoint, load_checkpoint
from src.infer import predict_proba


def test_checkpoint_then_infer_roundtrip(tmp_path):
    device = torch.device("cpu")
    torch.manual_seed(0)

    model1 = build_model(num_classes=10).to(device)
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)

    x = torch.randn(4, 3, 32, 32)

    # Run one optimizer step to make state non-trivial
    y = torch.randint(0, 10, (4,))
    loss = torch.nn.CrossEntropyLoss()(model1(x), y)
    opt1.zero_grad()
    loss.backward()
    opt1.step()

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model=model1, optimizer=opt1, epoch=1, step=10)

    model2 = build_model(num_classes=10).to(device)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    meta = load_checkpoint(ckpt, model=model2, optimizer=opt2, map_location=device)

    assert meta["epoch"] == 1
    assert meta["step"] == 10

    proba1 = predict_proba(model1, x, device)
    proba2 = predict_proba(model2, x, device)

    assert torch.allclose(proba1, proba2, atol=0, rtol=0)
