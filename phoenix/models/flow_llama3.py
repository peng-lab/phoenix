"""
Flow matching model based on Transformer
© Manuel Tran / Helmholtz Munich
"""

import math
import yaml
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor as Tensor
from torch.utils.checkpoint import checkpoint

from apex.normalization import FusedRMSNorm
from flash_attn import flash_attn_func
from xformers.ops import SwiGLU

#--------------------------------------------------------------------------------


def modulate(x: Tensor, shift: Tensor, scale: Tensor):
    """
    Multiply the input x by 1 + scale and translate by shift.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def requires_grad(model: nn.Module, flag: bool = True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#--------------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """
    The scalar timestep to vector embedding layer.
    """
    def __init__(self, hidden_size: int, freq_emb_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=freq_emb_size,
                out_features=hidden_size,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=False,
            ),
        )
        self.freq_emb_size = freq_emb_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: Tensor):
        t_freq = self.timestep_embedding(t, self.freq_emb_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    The class label to vector embedding layer.
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_embeddings=num_classes + use_cfg_embedding,
            embedding_dim=hidden_size,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: Tensor, force_drop_ids: bool = None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(
                labels.shape[0], device=labels.device
            ) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: Tensor, train: bool, force_drop_ids: bool = None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#--------------------------------------------------------------------------------


class SwiGLUFFNFused(nn.Module):
    """
    The position-wise feed-forward neural network.
    """
    def __init__(
        self,
        d_input: int = 1024,
        d_output: Optional[int] = None,
        ffn_mult: int = 4,
        ffn_bias: bool = True,
    ):
        super().__init__()
        d_hidden = ffn_mult * d_input
        d_hidden = int(2 * d_hidden / 3)
        d_hidden = (d_hidden // 64) * 64

        self.swiglu = SwiGLU(
            in_features=d_input,
            hidden_features=d_hidden,
            out_features=d_output,
            bias=ffn_bias,
            _pack_weights=False,
        )

    def forward(self, x: Tensor):
        return self.swiglu(x)


class FlashAttention(nn.Module):
    """
    The scaled dot-product multi-head attention.
    """
    def __init__(
        self,
        d_model: int = 1024,
        d_cross: Optional[int] = None,
        n_heads: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        d_cross = d_model if d_cross is None else d_cross

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5

        self.weight_q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )
        self.weight_k = nn.Linear(
            in_features=d_cross,
            out_features=d_model,
            bias=qkv_bias,
        )
        self.weight_v = nn.Linear(
            in_features=d_cross,
            out_features=d_model,
            bias=qkv_bias,
        )
        self.weight_o = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=qkv_bias,
        )
        self.norm_q = FusedRMSNorm(
            normalized_shape=self.d_head,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.norm_k = FusedRMSNorm(
            normalized_shape=self.d_head,
            eps=1e-5,
            elementwise_affine=True,
        )

        self.attn_drop = attn_drop
        self.dropout = nn.Dropout(proj_drop)

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, x: Tensor, c: Tensor = None):
        x_bsz, x_sql, _ = x.shape
        if c is not None:
            c_bsz, c_sql, _ = c.shape
        else:
            c_bsz, c_sql, _ = x.shape

        xq = self.weight_q(x)  # inputs serve as query
        xk = self.weight_k(c if c is not None else x)
        xv = self.weight_v(c if c is not None else x)

        xq = xq.view(x_bsz, x_sql, self.n_heads, self.d_head)
        xk = xk.view(c_bsz, c_sql, self.n_heads, self.d_head)
        xv = xv.view(c_bsz, c_sql, self.n_heads, self.d_head)

        xq = self.norm_q(xq)
        xk = self.norm_k(xk)

        xo = flash_attn_func(
            xq,
            xk,
            xv,
            dropout_p=self.attn_drop,
            softmax_scale=self.scale,
            causal=False,
            window_size=(-1, -1),
            return_attn_probs=False,
        )

        xo = xo.contiguous().view(x_bsz, x_sql, -1)
        return self.dropout(self.weight_o(xo.float()))


#--------------------------------------------------------------------------------


class ClassicalTransformerBlock(nn.Module):
    """
    The classical transformer block.
    """
    def __init__(self, cfg):
        super().__init__()
        self.mha = FlashAttention(
            d_model=cfg.d_cross,
            d_cross=None,
            n_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop
        )
        self.mlp = SwiGLUFFNFused(
            d_input=cfg.d_cross,
            d_output=None,
            ffn_mult=cfg.ffn_mult,
            ffn_bias=cfg.ffn_bias,
        )
        self.norm_1 = FusedRMSNorm(
            normalized_shape=cfg.d_cross,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.norm_2 = FusedRMSNorm(
            normalized_shape=cfg.d_cross,
            eps=1e-5,
            elementwise_affine=True,
        )

    def forward(self, x: Tensor):
        x = x + self.mha(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class FlowTransformerBlock(nn.Module):
    """
    The flow matching transformer block.
    """
    def __init__(self, cfg):
        super().__init__()
        self.zattn = FlashAttention(
            d_model=cfg.d_model,
            d_cross=None,
            n_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop
        )
        self.xattn = FlashAttention(
            d_model=cfg.d_model,
            d_cross=cfg.d_cross,
            n_heads=cfg.n_heads,
            qkv_bias=cfg.qkv_bias,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop
        )
        self.mlp = SwiGLUFFNFused(
            d_input=cfg.d_model,
            d_output=None,
            ffn_mult=cfg.ffn_mult,
            ffn_bias=cfg.ffn_bias,
        )
        self.norm_1 = FusedRMSNorm(
            normalized_shape=cfg.d_model,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.norm_2 = FusedRMSNorm(
            normalized_shape=cfg.d_model,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.norm_3 = FusedRMSNorm(
            normalized_shape=cfg.d_model,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.adaLN_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=cfg.d_model,
                out_features=3 * cfg.d_model,
                bias=False,
            ),
        )
        #self.adaLN_2 = nn.Sequential(
        #    nn.SiLU(),
        #    nn.Linear(
        #        in_features=cfg.d_model,
        #        out_features=3 * cfg.d_model,
        #        bias=False,
        #    ),
        #)
        self.adaLN_3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=cfg.d_model,
                out_features=3 * cfg.d_model,
                bias=False,
            ),
        )

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None):
        shift_zattn, scale_zattn, gate_zattn = self.adaLN_1(t).chunk(3, dim=1)
        #shift_xattn, scale_xattn, gate_xattn = self.adaLN_2(t).chunk(3, dim=1)
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_3(t).chunk(3, dim=1)

        r = self.zattn(modulate(self.norm_1(x), shift_zattn, scale_zattn))
        x = x + gate_zattn.unsqueeze(1) * r

        r = self.xattn(self.norm_2(x), c)
        x = x + r

        r = self.mlp(modulate(self.norm_3(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * r

        return x


class FlowTransformerHead(nn.Module):
    """
    The flow matching transformer head.
    """
    def __init__(
        self,
        in_features: int = 1024,
        out_features: int = 65536,
        bias: bool = True,
    ):
        super().__init__()
        self.norm = FusedRMSNorm(
            normalized_shape=in_features,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=in_features,
                out_features=2 * in_features,
                bias=bias,
            )
        )

    def forward(self, x: Tensor, t: Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


#--------------------------------------------------------------------------------


class FlowTransformerModel(nn.Module):
    """
    The flow matching transformer model.
    """
    def __init__(self, cfg, vision_model=None):
        super().__init__()
        self.cfg = cfg
        self.checkpoint = cfg.checkpoint

        if vision_model:
            requires_grad(vision_model, False)
            self.vision_model = vision_model.eval()
        else:
            self.vision_model = None

        self.x_projection = SwiGLUFFNFused(
            d_input=cfg.d_genes,
            d_output=cfg.d_model,
            ffn_mult=cfg.ffn_mult,
            ffn_bias=cfg.ffn_bias,
        )
        self.c_norm = FusedRMSNorm(
            normalized_shape=cfg.d_image,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.c_projection = SwiGLUFFNFused(
            d_input=cfg.d_image,
            d_output=cfg.d_cross,
            ffn_mult=cfg.ffn_mult,
            ffn_bias=cfg.ffn_bias,
        )
        self.px_embedding = nn.Embedding(
            num_embeddings=1024,
            embedding_dim=cfg.d_model,
        )
        self.pc_embedding = nn.Embedding(
            num_embeddings=1024,
            embedding_dim=cfg.d_cross,
        )
        self.t_embedding = TimestepEmbedder(
            hidden_size=cfg.d_model,
            freq_emb_size=256,
        )
        self.y_embedding = LabelEmbedder(
            num_classes=cfg.n_classes,
            hidden_size=cfg.d_model,
            dropout_prob=cfg.cls_drop
        )
        self.layers = nn.ModuleList([])
        for _ in range(2):
            self.layers.append(ClassicalTransformerBlock(cfg))
        self.blocks = nn.ModuleList([])
        for _ in range(cfg.n_layers):
            self.blocks.append(FlowTransformerBlock(cfg))
        self.head = FlowTransformerHead(
            in_features=cfg.d_model,
            out_features=cfg.d_genes,
            bias=False,
        )

    def _custom(self, layer, x, t, c):
        return layer(x, t, c)

    def vision_forward(self, c: Tensor):
        # extract image features from vision model
        if self.vision_model:
            with torch.no_grad():
                #c = self.vision_model.forward_features(c)
                c = self.vision_model(c)
        return c

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None, y: Tensor = None):
        # extract or use pre-extracted image features
        if c.dim() == 4:
            c = self.vision_forward(c)

        # project genes and add position embedding
        px = torch.arange(x.size(1), device=x.device)
        x = self.x_projection(x)
        x = x + self.px_embedding(px)

        # project image and add position embedding
        pc = torch.arange(c.size(1), device=x.device)
        c = self.c_projection(self.c_norm(c))
        c = c + self.pc_embedding(pc)

        # adapt conditionings for the task at hand
        for layer in self.layers:
            if self.checkpoint:
                c = checkpoint(layer, c, use_reentrant=True)
            else:
                c = layer(c)  # use gradient checkpointing

        # embed time and labels, then add them
        t = self.t_embedding(t)
        if y is not None:
            t = t + self.y_embedding(y, self.training)

        # process tokens in the transformer blocks
        for block in self.blocks:
            if self.checkpoint:
                x = checkpoint(
                    self._custom,
                    block,
                    x,
                    t,
                    c,
                    use_reentrant=True,
                )
            else:
                x = block(x, t, c)

        # apply the output projection heads
        return self.head(x, t)


#--------------------------------------------------------------------------------


@dataclass
class FlowTransformerConfig:
    '''
    The flow matching model configuration.
    '''
    d_genes: int = 8
    d_image: int = 1536
    d_model: int = 1024
    d_cross: int = 1024
    n_heads: int = 16
    n_layers: int = 24
    qkv_bias: bool = False
    ffn_bias: bool = False
    ffn_mult: int = 4
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    n_classes: int = 16
    cls_drop: float = 0.1

    checkpoint: bool = True

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
            return cls(**config_data)

    def save_yaml(self, file_path: str):
        with open(file_path, 'w') as file:
            yaml.safe_dump(self.__dict__, file)


#--------------------------------------------------------------------------------

if __name__ == "__main__":

    model = FlowTransformerModel(FlowTransformerConfig(), None).cuda()
    print('Model:', sum(p.numel() for p in model.parameters()) / 1e+6)

    x = torch.rand(1, 64, 8).cuda()
    t = torch.rand(x.shape[0]).cuda()
    c = torch.rand(1, 256, 1536).cuda()

    output = model(x, t, c)
    print(output.size())
