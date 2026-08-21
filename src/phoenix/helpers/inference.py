"""
Flow matching inference based on ODE solver
© Peng Lab / Helmholtz Munich
"""

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from zuko.utils import odeint

#-------------------------------------------------------------------------------


@torch.no_grad()
def run_flow(flow_model, x_0, t_0, t_1, c, y, atol, rtol, device='cpu'):
    """
    Integrate the velocity field given by the flow model.
    """
    flow_model.eval()
    phi = flow_model.parameters()

    def f(t: float, x: torch.tensor):
        t = torch.full((x.shape[0], ), t, device=device)
        return flow_model(x, t, c, y)

    return odeint(f, x_0, t_0, t_1, phi=phi, atol=atol, rtol=rtol)


#-------------------------------------------------------------------------------


class FlowPipeline:
    """
    Wraps a flow matching model with its sampling logic.
    """
    def __init__(
        self,
        model: nn.Module,
        stats: dict = None,
        t_0: float = 0.0,
        t_1: float = 1.0,
        atol: float = 1e-1,
        rtol: float = 1e-1
    ):
        self.model = model
        self.t_0 = t_0
        self.t_1 = t_1
        self.atol = atol
        self.rtol = rtol
        self.mean, self.std = stats["mean"], stats["std"]
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def __call__(self, gene_list: list, dataloader: DataLoader):
        """
        Run the flow model on every batch in the dataloader.
        """
        self.model.eval()
        device = self.device

        pred_list, coords_list = [], []
        for batch in tqdm(dataloader, desc='Flow sampling'):
            image, coords = batch[0].cuda(), batch[1]
            feats = self.model.vision_forward(image)
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
