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
