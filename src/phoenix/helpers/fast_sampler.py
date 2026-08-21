"""
Fast flow inference that caches the conditioning K/V projections.

The conditioning tensor `c` is constant across ODE steps, so its key/value
projections in every cross-attention block can be computed once and reused,
instead of being recomputed at every solver step.
"""

import torch
from zuko.utils import odeint

# -------------------------------------------------------------------------------


class OptimizedFlow:
    """
    Wraps a flow transformer model, caching the conditioning K/V projections.

    Parameters
    ----------
    flow_model
        Flow transformer model (``flow_llama3`` or ``flow_simple``) to wrap.
    """

    def __init__(self, flow_model):
        self.m = flow_model
        self._kv = None

    @torch.no_grad()
    def prep(self, c):
        """
        Precompute the K/V projections of the conditioning tensor `c`.

        `c` is constant in the ODE, so its K/V projections can be cached once.
        """
        m = self.m
        self._kv = None  # drop any previous cache

        if c.dim() == 4:  # if c is an image, extract image features
            c = m.vision_forward(c)

        # mirror the model's conditioning path dtype (fp32)
        pc = torch.arange(c.size(1), device=c.device)
        c = m.c_projection(m.c_norm(c))  # c is the feature vector
        c = c + m.pc_embedding(pc)
        for layer in m.layers:  # conditioning blocks
            c = layer(c)

        c_bsz, c_sql, _ = c.shape
        kv = []

        # mirror the attention blocks' dtype (bf16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block in m.blocks:
                attn = block.xattn
                xk = attn.weight_k(c)
                xv = attn.weight_v(c)
                xk = xk.view(c_bsz, c_sql, attn.n_heads, attn.d_head)
                xv = xv.view(c_bsz, c_sql, attn.n_heads, attn.d_head)
                xk = attn.norm_k(xk)
                kv.append((xk, xv))

        self._kv = kv  # c isn't needed anymore

    @torch.no_grad()
    def velocity(self, t_scalar, x, y=None, device=None):
        """Evaluate the velocity field at time `t_scalar`, reusing the cached K/V."""
        m = self.m

        t = torch.full((x.shape[0],), t_scalar, device=device or x.device)

        px = torch.arange(x.size(1), device=x.device)
        x = m.x_projection(x)
        x = x + m.px_embedding(px)

        t = m.t_embedding(t)
        if m.cfg.n_classes > 0 and y is not None:
            t = t + m.y_embedding(y, m.training)

        if self._kv is None:
            raise RuntimeError("call prep(c) before velocity()")
        for block, kv in zip(m.blocks, self._kv, strict=True):
            x = block(x, t, None, kv)

        return m.head(x, t)


# -------------------------------------------------------------------------------


@torch.no_grad()
def run_fast_flow(flow_model, x_0, t_0, t_1, c, y, atol, rtol, device="cpu"):
    """
    Integrate the velocity field, caching the conditioning K/V projections.

    Drop-in replacement for ``run_flow`` that computes the K/V projections of
    `c` once instead of recomputing them at every ODE solver step.
    """
    flow_model.eval()
    phi = flow_model.parameters()
    opt = OptimizedFlow(flow_model)
    opt.prep(c)

    def f(t: float, x: torch.Tensor):
        return opt.velocity(t, x, y=y, device=device)

    return odeint(f, x_0, t_0, t_1, phi=phi, atol=atol, rtol=rtol)
