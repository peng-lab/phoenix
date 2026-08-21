"""
Autoencoder model based on MLP-Mixer
© Peng Lab / Helmholtz Munich
"""

import torch
import torch.nn as nn
from apex.normalization import FusedRMSNorm
from torch import Tensor as Tensor
from torch.utils.checkpoint import checkpoint
from vector_quantize_pytorch import FSQ
from xformers.ops import SwiGLU

# --------------------------------------------------------------------------------


class SwiGLUFFNFused(nn.Module):
    """
    The position-wise feed-forward neural network, backed by `xformers`' fused SwiGLU.

    Parameters
    ----------
    d_input
        Input feature dimension.
    d_output
        Output feature dimension.
    ffn_mult
        Hidden-dimension multiplier, before rounding to a multiple of 2.
    ffn_bias
        Whether the internal linear layers use a bias term.
    """

    def __init__(
        self,
        d_input: int = 1024,
        d_output: int = 1024,
        ffn_mult: int = 4,
        ffn_bias: bool = True,
    ):
        super().__init__()
        d_hidden = ffn_mult * d_input
        d_hidden = int(2 * d_hidden / 3)
        d_hidden = (d_hidden // 2) * 2

        self.swiglu = SwiGLU(
            in_features=d_input,
            hidden_features=d_hidden,
            out_features=d_output,
            bias=ffn_bias,
            _pack_weights=False,
        )

    def forward(self, x: Tensor):
        """
        Apply the gated feed-forward network.

        Parameters
        ----------
        x
            Input tensor of shape ``(..., d_input)``.

        Returns
        -------
        Output tensor of shape ``(..., d_output)``.
        """
        return self.swiglu(x)


# --------------------------------------------------------------------------------


class MixerBlock(nn.Module):
    """
    The multilayer perceptron mixer block.

    Mixes information across tokens (`token_mixer`) and, after a residual linear
    projection to the output token count, across channels (`channel_mixer`).

    Parameters
    ----------
    n_tokens_input
        Number of input tokens.
    n_tokens_output
        Number of output tokens.
    n_channels_input
        Input channel dimension.
    n_channels_output
        Output channel dimension.
    ffn_mult
        Hidden-dimension multiplier used by the internal feed-forward networks.
    ffn_bias
        Whether the internal linear layers use a bias term.
    """

    def __init__(
        self,
        n_tokens_input: int = 1024,
        n_tokens_output: int = 64,
        n_channels_input: int = 512,
        n_channels_output: int = 512,
        ffn_mult: int = 4,
        ffn_bias: bool = True,
    ):
        super().__init__()
        self.token_mixer = SwiGLUFFNFused(
            d_input=n_tokens_input,
            d_output=n_tokens_output,
            ffn_mult=ffn_mult,
            ffn_bias=ffn_bias,
        )
        self.channel_mixer = SwiGLUFFNFused(
            d_input=n_channels_input,
            d_output=n_channels_output,
            ffn_mult=ffn_mult,
            ffn_bias=ffn_bias,
        )
        self.linear = nn.Linear(
            in_features=n_tokens_input,
            out_features=n_tokens_output,
            bias=ffn_bias,
        )
        self.norm_1 = FusedRMSNorm(
            normalized_shape=n_channels_input,
            eps=1e-05,
            elementwise_affine=True,
        )
        self.norm_2 = FusedRMSNorm(
            normalized_shape=n_channels_input,
            eps=1e-05,
            elementwise_affine=True,
        )

    def forward(self, x: Tensor):
        """
        Mix `x` across tokens, then across channels.

        Parameters
        ----------
        x
            Input tensor, shape ``(batch, n_tokens_input, n_channels_input)``.

        Returns
        -------
        Output tensor, shape ``(batch, n_tokens_output, n_channels_output)``.
        """
        r = self.token_mixer(self.norm_1(x).transpose(-1, -2)).transpose(-1, -2)
        h = self.linear(x.transpose(-1, -2)).transpose(-1, -2) + r
        r = self.channel_mixer(self.norm_2(h))
        o = h + r
        return o


# --------------------------------------------------------------------------------


class MixerEncoder(nn.Module):
    """
    The multilayer perceptron mixer encoder model.

    Parameters
    ----------
    d_input
        Input feature dimension.
    d_tokens
        Channel dimension used throughout the mixer blocks.
    n_tokens
        Token counts at each stage; `n_tokens[i]` is the input token count and
        `n_tokens[i + 1]` the output token count of mixer block `i`. Defaults to
        ``[1024, 256, 64]`` when `None`.
    n_layers
        Number of `MixerBlock` layers; must be less than ``len(n_tokens)``.
    ffn_mult
        Hidden-dimension multiplier used by the internal feed-forward networks.
    ffn_bias
        Whether the internal linear layers use a bias term.
    """

    def __init__(
        self,
        d_input: int = 1,
        d_tokens: int = 512,
        n_tokens: list | None = None,
        n_layers: int = 2,
        ffn_mult: int = 4,
        ffn_bias: bool = True,
    ):
        super().__init__()
        if n_tokens is None:
            n_tokens = [1024, 256, 64]
        self.position_embedding = nn.Embedding(
            num_embeddings=n_tokens[0],
            embedding_dim=d_input,
        )
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Linear(
                in_features=d_input,
                out_features=d_tokens,
                bias=True,
            )
        )
        for i in range(n_layers):
            self.encoder.append(
                MixerBlock(
                    n_tokens_input=n_tokens[i],
                    n_tokens_output=n_tokens[i + 1],
                    n_channels_input=d_tokens,
                    n_channels_output=d_tokens,
                    ffn_mult=ffn_mult,
                    ffn_bias=ffn_bias,
                )
            )

    def forward(self, x: Tensor):
        """
        Encode `x` through the position embedding and mixer blocks.

        Parameters
        ----------
        x
            Input tensor, shape ``(batch, n_tokens[0], d_input)``.

        Returns
        -------
        Encoded tensor, shape ``(batch, n_tokens[n_layers], d_tokens)``.
        """
        p = torch.arange(x.size(1)).to(x.device)
        x = x + self.position_embedding(p)
        for encoder_block in self.encoder:
            x = encoder_block(x)
        return x


class MixerDecoder(nn.Module):
    """
    The multilayer perceptron mixer decoder model.

    Mirrors `MixerEncoder`, ending with a `SwiGLUFFNFused` projection back to a
    single channel.

    Parameters
    ----------
    d_input
        Output feature dimension.
    d_tokens
        Channel dimension used throughout the mixer blocks.
    n_tokens
        Token counts at each stage, from smallest to largest. Defaults to
        ``[64, 256, 1024]`` when `None`.
    n_layers
        Number of `MixerBlock` layers; must be less than ``len(n_tokens)``.
    ffn_mult
        Hidden-dimension multiplier used by the internal feed-forward networks.
    ffn_bias
        Whether the internal linear layers use a bias term.
    """

    def __init__(
        self,
        d_input: int = 1,
        d_tokens: int = 512,
        n_tokens: list | None = None,
        n_layers: int = 2,
        ffn_mult: int = 4,
        ffn_bias: bool = True,
    ):
        super().__init__()
        if n_tokens is None:
            n_tokens = [64, 256, 1024]
        self.position_embedding = nn.Embedding(
            num_embeddings=n_tokens[0],
            embedding_dim=d_input,
        )
        self.decoder = nn.ModuleList()
        self.decoder.append(
            nn.Linear(
                in_features=d_input,
                out_features=d_tokens,
                bias=True,
            )
        )
        for i in range(n_layers):
            self.decoder.append(
                MixerBlock(
                    n_tokens_input=n_tokens[i],
                    n_tokens_output=n_tokens[i + 1],
                    n_channels_input=d_tokens,
                    n_channels_output=d_tokens,
                    ffn_mult=ffn_mult,
                    ffn_bias=ffn_bias,
                )
            )
        self.decoder.append(
            SwiGLUFFNFused(
                d_input=d_tokens,
                d_output=1,
                ffn_mult=ffn_mult,
                ffn_bias=ffn_bias,
            )
        )

    def forward(self, x: Tensor):
        """
        Decode `x` through the position embedding, mixer blocks, and output projection.

        Parameters
        ----------
        x
            Input tensor, shape ``(batch, n_tokens[0], d_input)``.

        Returns
        -------
        Decoded tensor, shape ``(batch, n_tokens[-1], 1)``.
        """
        p = torch.arange(x.size(1)).to(x.device)
        x = x + self.position_embedding(p)
        for decoder_block in self.decoder:
            x = decoder_block(x)
        return x


class MixerAutoencoder(nn.Module):
    """
    The multilayer perceptron mixer autoencoder model.

    Combines a `MixerEncoder`-style encoder with an optional finite scalar
    quantization (`FSQ`) bottleneck, and a matching decoder.

    Parameters
    ----------
    d_input
        Input/output feature dimension.
    d_tokens
        Channel dimension used throughout the mixer blocks.
    n_tokens
        Token counts at each encoder stage, largest to smallest; the decoder
        uses them in reverse. Defaults to ``[1024, 256, 64]`` when `None`.
    n_layers
        Number of `MixerBlock` layers per encoder/decoder stack.
    ffn_mult
        Hidden-dimension multiplier used by the internal feed-forward networks.
    ffn_bias
        Whether the internal linear layers use a bias term.
    levels
        FSQ quantization levels per dimension; quantization is disabled when
        `None` or empty. Defaults to ``[8, 5, 5, 5]`` when not overridden.
    checkpoint
        Whether to use gradient checkpointing in `encode`/`decode`.
    """

    def __init__(
        self,
        d_input: int = 1,
        d_tokens: int = 512,
        n_tokens: list | None = None,
        n_layers: int = 2,
        ffn_mult: int = 4,
        ffn_bias: bool = True,
        levels: list | None = None,
        checkpoint: bool = False,
    ):
        super().__init__()
        if n_tokens is None:
            n_tokens = [1024, 256, 64]
        if levels is None:
            levels = [8, 5, 5, 5]
        self.checkpoint = checkpoint

        self.position_embedding = nn.Embedding(
            num_embeddings=n_tokens[0],
            embedding_dim=d_input,
        )

        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Linear(
                in_features=d_input,
                out_features=d_tokens,
                bias=True,
            )
        )
        for i in range(n_layers):
            self.encoder.append(
                MixerBlock(
                    n_tokens_input=n_tokens[i],
                    n_tokens_output=n_tokens[i + 1],
                    n_channels_input=d_tokens,
                    n_channels_output=d_tokens,
                    ffn_mult=ffn_mult,
                    ffn_bias=ffn_bias,
                )
            )

        self.norm = nn.LayerNorm(
            normalized_shape=d_tokens,
            eps=1e-5,
            elementwise_affine=False,
            bias=False,
        )
        if levels:
            self.quantizer = FSQ(levels=levels)
        else:
            self.quantizer = None

        self.decoder = nn.ModuleList()
        for i in range(n_layers):
            self.decoder.append(
                MixerBlock(
                    n_tokens_input=n_tokens[-i - 1],
                    n_tokens_output=n_tokens[-i - 2],
                    n_channels_input=d_tokens,
                    n_channels_output=d_tokens,
                    ffn_mult=ffn_mult,
                    ffn_bias=ffn_bias,
                )
            )
        self.decoder.append(
            SwiGLUFFNFused(
                d_input=d_tokens,
                d_output=d_input,
                ffn_mult=4,
                ffn_bias=True,
            )
        )

    def encode(self, x: Tensor):
        """
        Encode `x`, quantizing the result when an `FSQ` bottleneck is configured.

        Parameters
        ----------
        x
            Input tensor, shape ``(batch, n_tokens[0], d_input)``.

        Returns
        -------
        Either the encoded tensor, or a tuple of the quantized tensor and its
        FSQ indices when quantization is enabled.
        """
        p = torch.arange(x.size(1)).to(x.device)
        x = x + self.position_embedding(p)

        for encoder_block in self.encoder:
            if self.checkpoint:
                x = checkpoint(encoder_block, x, use_reentrant=True)
            else:
                x = encoder_block(x)  # use gradient checkpointing

        x = self.norm(x)

        if self.quantizer:
            q, indices = self.quantizer(x)
            x = q.type(x.dtype)
            return x, indices

        return x

    def decode(self, x: Tensor):
        """
        Decode `x` back to input (channel) space.

        Parameters
        ----------
        x
            Encoded tensor, as returned by `encode`.

        Returns
        -------
        Decoded tensor, shape ``(batch, n_tokens[0], d_input)``.
        """
        for decoder_block in self.decoder:
            if self.checkpoint:
                x = checkpoint(decoder_block, x, use_reentrant=True)
            else:
                x = decoder_block(x)  # use gradient checkpointing

        return x

    def forward(self, x: Tensor, return_indices: bool = False):
        """
        Encode and decode `x`, optionally also returning the FSQ indices.

        Parameters
        ----------
        x
            Input tensor, shape ``(batch, n_tokens[0], d_input)``.
        return_indices
            Whether to also return the FSQ quantization indices; requires an
            `FSQ` bottleneck to be configured.

        Returns
        -------
        The reconstructed tensor, or a tuple of the reconstructed tensor and FSQ
        indices when `return_indices` is `True`.
        """
        # Pass through encoder
        if self.quantizer:
            x, indices = self.encode(x)
        else:
            x = self.encode(x)

        # Pass through decoder
        x = self.decode(x)

        if return_indices:
            return x, indices

        return x


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    encoder_block = MixerBlock(
        n_tokens_input=1024,
        n_tokens_output=64,
        n_channels_input=512,
        n_channels_output=512,
        ffn_mult=4,
        ffn_bias=True,
    ).cuda()
    inputs = torch.rand(1, 1024, 512).cuda()
    print(encoder_block(inputs).shape)

    decoder_block = MixerBlock(
        n_tokens_input=64,
        n_tokens_output=1024,
        n_channels_input=512,
        n_channels_output=512,
        ffn_mult=4,
        ffn_bias=True,
    ).cuda()
    inputs = torch.rand(1, 64, 512).cuda()
    print(decoder_block(inputs).shape)

    encoder_model = MixerEncoder(
        d_input=1,
        d_tokens=256,
        n_tokens=[1370, 1024, 768, 512],
        n_layers=3,
        ffn_mult=4,
        ffn_bias=True,
    ).cuda()
    inputs = torch.rand(1, 1370, 1).cuda()
    print(encoder_model(inputs).shape)

    decoder_model = MixerDecoder(
        d_input=1,
        d_tokens=256,
        n_tokens=[512, 768, 1024, 1370],
        n_layers=3,
        ffn_mult=4,
        ffn_bias=True,
    ).cuda()
    inputs = torch.rand(1, 512, 1).cuda()
    print(decoder_model(inputs).shape)

    autoencoder_model = MixerAutoencoder(
        d_input=1,
        d_tokens=4,
        n_tokens=[1024, 256, 64],
        n_layers=2,
        ffn_mult=4,
        ffn_bias=True,
        levels=[8, 5, 5, 5],
        checkpoint=True,
    ).cuda()
    inputs = torch.rand(1, 1024, 1).cuda()
    outputs, indices = autoencoder_model(inputs, return_indices=True)
    print(outputs.shape, indices.shape)

    parameters = sum(p.numel() for p in autoencoder_model.parameters())
    print(f"MLP-Mixer model size: {parameters / 1e6:.3f} Million")
