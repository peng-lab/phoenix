"""
Flow matching inference based on ODE solver.

© Peng Lab / Helmholtz Munich
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from zuko.utils import odeint

# -------------------------------------------------------------------------------


@torch.no_grad()
def run_flow(flow_model, x_0, t_0, t_1, c, y, atol, rtol, device="cpu"):
    """
    Integrate the velocity field given by the flow model.

    Parameters
    ----------
    flow_model
        Flow transformer model providing the velocity field; put into eval mode
        before integration.
    x_0
        Initial state (noise) to integrate from.
    t_0
        Start time of the integration.
    t_1
        End time of the integration.
    c
        Conditioning tensor (e.g. image features) passed to `flow_model` at each step.
    y
        Optional class-conditioning tensor passed to `flow_model` at each step.
    atol
        Absolute tolerance passed to the ODE solver.
    rtol
        Relative tolerance passed to the ODE solver.
    device
        Device the per-step time tensor is created on.

    Returns
    -------
    The integrated state at `t_1`.
    """
    flow_model.eval()
    phi = flow_model.parameters()

    def f(t: float, x: torch.Tensor):
        t_vec = torch.full((x.shape[0],), t, device=device)
        return flow_model(x, t_vec, c, y)

    return odeint(f, x_0, t_0, t_1, phi=phi, atol=atol, rtol=rtol)


# -------------------------------------------------------------------------------


class FlowPipeline:
    """
    Wraps a flow matching model with its sampling logic.

    Parameters
    ----------
    model
        Flow transformer model to sample from.
    stats
        Dict with ``"mean"`` and ``"std"`` entries used to de-normalize predicted
        gene expression back to its original scale.
    t_0
        Start time of the flow integration.
    t_1
        End time of the flow integration.
    atol
        Absolute tolerance passed to the ODE solver.
    rtol
        Relative tolerance passed to the ODE solver.
    """

    def __init__(
        self,
        model: nn.Module,
        stats: dict | None = None,
        t_0: float = 0.0,
        t_1: float = 1.0,
        atol: float = 1e-1,
        rtol: float = 1e-1,
    ):
        self.model = model
        self.t_0 = t_0
        self.t_1 = t_1
        self.atol = atol
        self.rtol = rtol
        if stats is None:
            raise ValueError("FlowPipeline requires `stats` with 'mean' and 'std' entries")
        self.mean, self.std = stats["mean"], stats["std"]
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def __call__(self, gene_list: list, dataloader: DataLoader):
        """
        Run the flow model on every batch in the dataloader.

        Encodes each batch's images with `model.vision_forward`, samples gene
        expression from Gaussian noise by integrating the flow, and de-normalizes
        the result using the pipeline's stored `mean`/`std`.

        Parameters
        ----------
        gene_list
            Genes to sample; only its length is used, to size the initial noise.
        dataloader
            Batches of ``(image, coords)`` to run inference on.

        Returns
        -------
        Tuple of the concatenated, de-normalized predicted gene expression array
        and the list of per-batch coordinate tensors.
        """
        self.model.eval()
        device = self.device

        pred_list, coords_list = [], []
        for batch in tqdm(dataloader, desc="Flow sampling"):
            image, coords = batch[0].cuda(), batch[1]
            # nn.Module's __getattr__ stub can't see `vision_forward`, a method
            # specific to FlowTransformerModel; declaring `model: nn.Module`
            # keeps this pipeline decoupled from a specific model implementation.
            feats = self.model.vision_forward(image)  # type: ignore[operator]
            noise = torch.randn(image.size(0), len(gene_list), 1).cuda()

            gex_pred = run_flow(
                flow_model=self.model,
                x_0=noise.float(),
                t_0=self.t_0,
                t_1=self.t_1,
                c=feats,
                y=None,
                atol=self.atol,
                rtol=self.rtol,
                device=device,
            )

            gex_pred = gex_pred.float().squeeze().detach().cpu().numpy()
            pred_list.append(gex_pred)
            coords_list.append(coords)

        gex_pred = np.concatenate(pred_list, axis=0)
        gex_pred = np.clip(gex_pred, 0, None)
        gex_pred = gex_pred * self.std + self.mean

        return gex_pred, coords_list
