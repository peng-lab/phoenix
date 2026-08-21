import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from phoenix.trainers.mixer_trainer import WarmupCosineAnnealingLR, move_to  # noqa: E402


def test_warmup_cosine_annealing_lr_warms_up_then_anneals():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=5,
        total_steps=10,
        start_lr=0.0,
        max_lr=1e-2,
        final_lr=1e-4,
    )

    lrs = []
    for _ in range(10):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # linear warmup: strictly increasing up to the peak at warmup_steps
    assert lrs[0] < lrs[4]
    assert lrs[4] == pytest.approx(1e-2, rel=1e-2)
    # cosine annealing afterwards: strictly decreasing towards final_lr
    assert lrs[-1] < lrs[4]
    assert lrs[-1] == pytest.approx(1e-4, rel=1e-2)


def test_move_to_moves_nested_tensor_structures():
    payload = {
        "a": torch.zeros(2),
        "b": [torch.zeros(2), torch.zeros(2, dtype=torch.float64)],
        "c": None,
    }
    moved = move_to(payload, torch.device("cpu"))

    assert moved["a"].device.type == "cpu"
    assert moved["b"][0].device.type == "cpu"
    # float64 tensors are downcast to float32 along the way
    assert moved["b"][1].dtype == torch.float32
    assert moved["c"] is None


def test_move_to_rejects_unsupported_types():
    with pytest.raises(TypeError):
        move_to(object(), torch.device("cpu"))
