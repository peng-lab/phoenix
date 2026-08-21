import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from phoenix.helpers.inference import FlowPipeline, run_flow  # noqa: E402
from phoenix.models.flow_simple import FlowTransformerConfig, FlowTransformerModel  # noqa: E402


@pytest.fixture
def tiny_model():
    cfg = FlowTransformerConfig(
        d_genes=4,
        d_image=16,
        d_model=16,
        d_cross=16,
        n_heads=2,
        n_layers=1,
        n_classes=0,
        checkpoint=False,
    )
    return FlowTransformerModel(cfg, None).eval()


def test_run_flow_integrates_to_the_target_shape(tiny_model):
    torch.manual_seed(0)
    x_0 = torch.randn(1, 6, 4)
    c = torch.randn(1, 5, 16)

    out = run_flow(tiny_model, x_0, t_0=0.0, t_1=1.0, c=c, y=None, atol=1e-1, rtol=1e-1)

    assert out.shape == x_0.shape


def test_flow_pipeline_requires_stats(tiny_model):
    with pytest.raises(ValueError, match="stats"):
        FlowPipeline(model=tiny_model, stats=None)


def test_flow_pipeline_stores_mean_and_std(tiny_model):
    stats = {"mean": np.array([1.0, 2.0]), "std": np.array([0.5, 0.5])}
    pipeline = FlowPipeline(model=tiny_model, stats=stats)
    np.testing.assert_array_equal(pipeline.mean, stats["mean"])
    np.testing.assert_array_equal(pipeline.std, stats["std"])


@pytest.fixture
def pipeline_model():
    # `__call__` builds its noise with a trailing dim of 1, so the pipeline path needs
    # d_genes=1 -- unlike `tiny_model`, which is shaped for run_flow's 4-wide x_0.
    cfg = FlowTransformerConfig(
        d_genes=1,
        d_image=16,
        d_model=16,
        d_cross=16,
        n_heads=2,
        n_layers=1,
        n_classes=0,
        checkpoint=False,
    )
    return FlowTransformerModel(cfg, None).eval()


def test_flow_pipeline_runs_on_cpu(pipeline_model):
    """
    FlowPipeline must run end-to-end without a GPU.

    `vision_forward` is a pass-through when no vision model is set, so the loader can
    yield already-extracted features directly.
    """
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    n_samples, n_genes = 4, 3
    feats = torch.randn(n_samples, 5, 16)
    coords = torch.zeros(n_samples, 2)
    # batch_size=1 would let the squeeze() inside __call__ collapse the batch dim
    loader = DataLoader(TensorDataset(feats, coords), batch_size=2)

    stats = {"mean": np.zeros(n_genes), "std": np.ones(n_genes)}
    pipeline = FlowPipeline(model=pipeline_model, stats=stats, atol=1e-1, rtol=1e-1)
    assert pipeline.device.type == "cpu"

    gex_pred, coords_list = pipeline(["A", "B", "C"], loader)

    assert gex_pred.shape == (n_samples, n_genes)
    assert np.isfinite(gex_pred).all()
    assert sum(len(c) for c in coords_list) == n_samples
