""" 
Lightning class for classical autoencoder
© Peng Lab / Helmholtz Munich
"""

import yaml

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim

from torch import Tensor as Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

#-------------------------------------------------------------------------------


def move_to(object: Union[Tensor, Dict, List, None], device: torch.device):
    # move single tensor to device
    if torch.is_tensor(object):
        if object.dtype == torch.float64:
            object = object.float()
        return object.to(device)
    elif object is None:
        return None
    # move list of tensors to device
    elif isinstance(object, list):
        object = [move_to(v, device) for v in object]
        return object
    # move dict of tensors to device
    elif isinstance(object, dict):
        for key, value in object.items():
            # move single value tensor to device
            if isinstance(value, Tensor):
                object[key] = move_to(value, device)
            # move list of value tensors to device
            elif isinstance(value, list):
                object[key] = [move_to(v, device) for v in value]
        return object
    else:
        raise TypeError("Invalid type for move_to")


#-------------------------------------------------------------------------------


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        start_lr: float,
        max_lr: float,
        final_lr: float = 0,
        last_step: int = -1,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.final_lr = final_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [
                self.start_lr + (self.max_lr - self.start_lr) *
                (self._step_count) / self.warmup_steps for base_lr in self.base_lrs
            ]
        else:
            t = self._step_count - self.warmup_steps
            T = self.total_steps - self.warmup_steps
            return [
                self.final_lr + (self.max_lr - self.final_lr) *
                (1 + torch.cos(torch.tensor((t / T) * torch.pi)).item()) / 2
                for base_lr in self.base_lrs
            ]


#-------------------------------------------------------------------------------


class MixerTrainer(pl.LightningModule):
    """
    The lightning class for autoencoder based on MLP-Mixer.
    """
    def __init__(self, cfg, mixer_model):
        super().__init__()
        #self.save_hyperparameters()
        self.mixer_model = mixer_model

        self.warmup_steps = cfg.warmup_steps
        self.total_steps = cfg.total_steps
        self.betas = cfg.betas
        self.start_lr = cfg.start_lr
        self.max_lr = cfg.max_lr
        self.final_lr = cfg.final_lr
        self.weight_decay = cfg.weight_decay

    def on_train_start(self):
        """
        Initialize the optimizer on re-start.
        """
        self.optimizers(use_pl_optimizer=False).param_groups[0]["lr"]
        self.optimizers(use_pl_optimizer=False).param_groups[0]["weight_decay"]
        self.optimizers().param_groups = (self.optimizers()._optimizer.param_groups)

    def on_train_batch_end(self, *_):
        """
        Update the scheduler after every step.
        """
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_optimizers(self):
        """
        Configurate the optimizer and scheduler.
        """
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.max_lr,
            betas=self.betas,
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            max_lr=self.max_lr,
            start_lr=self.start_lr,
            final_lr=self.final_lr,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def shared_step(self, batch: Tensor):
        """
        Compute the loss in a shared forward pass.
        """
        targets = batch.squeeze().unsqueeze(-1)
        recons = self.mixer_model(targets)
        return F.mse_loss(recons.squeeze(), targets.squeeze(), reduction='mean')

    def training_step(self, batch: Tensor, _):
        """
        Make one training step.
        """
        loss = self.shared_step(batch)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, _):
        """
        Make one validation step.
        """
        loss = self.shared_step(batch)
        self.log("loss/valid", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Tensor, _):
        """
        Make one test step.
        """
        loss = self.shared_step(batch)
        self.log("loss/test", loss, prog_bar=True)
        return loss


#-------------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    '''
    The lightning trainer configuration class.
    '''
    warmup_steps: int = 0
    total_steps: int = 0
    betas: Tuple[float, float] = (0.9, 0.95)
    start_lr: float = 0.0
    max_lr: float = 1e-4
    final_lr: float = 1e-5
    weight_decay: float = 0.1

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
            return cls(**config_data)

    def save_yaml(self, file_path: str):
        with open(file_path, 'w') as file:
            yaml.safe_dump(self.__dict__, file)
