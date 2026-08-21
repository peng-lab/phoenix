"""
Shape/forward-pass tests, converted from the `if __name__ == "__main__"` smoke
blocks that used to live in each model module.
"""

import pytest

torch = pytest.importorskip("torch")

from phoenix.models.flow_simple import FlowTransformerConfig, FlowTransformerModel  # noqa: E402


def test_flow_transformer_model_forward_and_backward():
    """Mirrors the flow_simple.py __main__ block, at CPU-friendly size."""
    torch.manual_seed(0)
    cfg = FlowTransformerConfig(
        d_genes=8,
        d_image=32,
        d_model=32,
        d_cross=32,
        n_heads=4,
        n_layers=2,
        n_classes=0,
        checkpoint=False,
    )
    model = FlowTransformerModel(cfg, None)
    optimizer = torch.optim.AdamW(model.parameters())

    x = torch.rand(1, 8, 8)
    t = torch.rand(x.shape[0])
    c = torch.rand(1, 8, 32)

    output = model(x, t, c)
    assert output.shape == x.shape

    loss = (output**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # a second forward pass after the optimizer step should still produce a
    # finite, correctly-shaped output
    output = model(x, t, c)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_flow_transformer_model_requires_conditioning():
    cfg = FlowTransformerConfig(d_genes=4, d_image=8, d_model=8, d_cross=8, n_heads=2, n_layers=1, n_classes=0)
    model = FlowTransformerModel(cfg, None)
    x = torch.rand(1, 4, 4)
    t = torch.rand(1)

    with pytest.raises(AssertionError):
        model(x, t, c=None)


# ---------------------------------------------------------------------------
# Optimized variants: need apex/flash-attn/xformers, which are deliberately not
# pip-installable (see pyproject.toml), plus flash-attn specifically requires a
# CUDA runtime. These are skipped in any environment lacking that stack -- which
# is the expected default -- but will run wherever it's actually available.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash-attn requires a CUDA device")
def test_flow_transformer_model_llama3_forward():
    pytest.importorskip("apex")
    pytest.importorskip("flash_attn")
    pytest.importorskip("xformers")
    from phoenix.models.flow_llama3 import FlowTransformerConfig as LlamaConfig
    from phoenix.models.flow_llama3 import FlowTransformerModel as LlamaModel

    cfg = LlamaConfig(d_genes=8, d_image=32, d_model=32, d_cross=32, n_heads=4, n_layers=2, n_classes=0)
    model = LlamaModel(cfg, None).cuda()

    x = torch.rand(1, 8, 8).cuda()
    t = torch.rand(1).cuda()
    c = torch.rand(1, 8, 32).cuda()

    output = model(x, t, c)
    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="apex/xformers fused ops require a CUDA device")
def test_mixer_autoencoder_roundtrip():
    pytest.importorskip("apex")
    pytest.importorskip("xformers")
    pytest.importorskip("vector_quantize_pytorch")
    from phoenix.models.mlp_mixer_ae import MixerAutoencoder

    model = MixerAutoencoder(
        d_input=1,
        d_tokens=4,
        n_tokens=[64, 16, 4],
        n_layers=2,
        ffn_mult=4,
        ffn_bias=True,
        levels=[8, 5, 5, 5],
        checkpoint=False,
    ).cuda()

    inputs = torch.rand(1, 64, 1).cuda()
    outputs, indices = model(inputs, return_indices=True)
    assert outputs.shape == inputs.shape
    assert indices is not None
