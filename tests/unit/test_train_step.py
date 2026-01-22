import torch
import torch.nn as nn


def test_single_train_step_runs_and_updates_params():
    from src.model import build_model

    torch.manual_seed(0)
    model = build_model(num_classes=10)
    model.train()

    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    # snapshot params
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    logits = model(x)
    loss = loss_fn(logits, y)
    assert torch.isfinite(loss).all()

    optim.zero_grad()
    loss.backward()
    optim.step()

    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    # At least one parameter tensor should change
    any_changed = any((b != a).any().item() for b, a in zip(before, after))
    assert any_changed
