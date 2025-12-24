# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
from typing import Optional, Tuple, List
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import PatchEmbed
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed


def replicate(x: torch.Tensor, num: int = 1) -> torch.Tensor:
    x = x.unsqueeze(1)  # [B, H] -> [B, 1, H]
    x = x.repeat([1, num, 1])  # [B, S, H]
    return x


@lru_cache(maxsize=8)
def prepare_latent_image_ids(height, width):
    """
    copy from diffusers.pipelines.flux.pipeline_flux

    :param height: latent height
    :param width: latent width
    """
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids


@lru_cache(maxsize=8)
def prepare_cond_ids(seq_length: int):
    """
    refer diffusers.pipelines.flux.pipeline_flux

    :param seq_length:
    """
    cond_ids = torch.zeros(seq_length, 3)

    return cond_ids


def create_axes_dim(dim: int, num_head: int) -> List[int]:
    assert dim % num_head == 0
    dim_per_head = dim // num_head

    dim_t = 16
    dim_h = dim_w = (dim_per_head - dim_t) // 2
    assert dim_t + dim_h + dim_w == dim_per_head

    return [dim_t, dim_h, dim_w]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, 256)
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        embeddings = self.mlp(embeddings)
        return embeddings


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = torch.float32
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection. (copy from timm repo)

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            fused_attn: bool = True
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            fused_attn: Whether to use fused attention
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            rotary_emb: Tuple[torch.Tensor],
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rotary_emb(q, rotary_emb, sequence_dim=2)
        k = apply_rotary_emb(k, rotary_emb, sequence_dim=2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                scale=self.scale,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """
    copy from https://github.com/hustvl/LightningDiT/blob/main/models/swiglu_ffn.py
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class DiTBlock(nn.Module):
    """
    A DiT block with SwiGLU.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3)
        self.mlp = SwiGLUFFN(in_features=hidden_size, hidden_features=mlp_hidden_dim)

    def forward(self, x, rotary_emb):
        x = x + self.attn(self.norm1(x), rotary_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        fused_attn=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.axes_dim = create_axes_dim(hidden_size, num_heads)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.h_embedder = TimestepEmbedder(hidden_size)
        self.w_embedder = TimestepEmbedder(hidden_size)
        self.interval_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = LabelEmbedder(num_classes, hidden_size)
        cond_seq = 4 + 4 + 4 + 2 + 2 + 8
        self.learnable_embed = nn.Parameter(torch.zeros(1, cond_seq, hidden_size), requires_grad=True)
        self.pos_embed = FluxPosEmbed(1000, self.axes_dim)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, fused_attn=fused_attn) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                fan_in = module.weight.size(1)
                torch.nn.init.normal_(module.weight, std=math.sqrt(0.1) / math.sqrt(fan_in))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.c_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        for module in [self.t_embedder, self.h_embedder, self.w_embedder, self.interval_embedder, self.c_embedder]:
            nn.init.normal_(module.mlp[0].weight, std=0.02)
            nn.init.constant_(module.mlp[0].bias, 0)
            nn.init.normal_(module.mlp[2].weight, std=0.02)
            nn.init.constant_(module.mlp[2].bias, 0)

        # Initial state of a residual block is always identity mapping:
        for block in self.blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.attn.proj.bias, 0)
            nn.init.constant_(block.mlp.w3.weight, 0)
            nn.init.constant_(block.mlp.w3.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, h, w, t_min, t_max, c, img_ids, cond_ids):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        c: (N,) tensor of class labels
        """
        x_tokens = self.x_embedder(x)             # (N, T, D), where T = H * W / patch_size ** 2
        x_seq = x_tokens.shape[1]
        t_embed = self.t_embedder(t)                 # (N, D)
        h_embed = self.h_embedder(h)                 # (N, D)
        w_embed = self.w_embedder(w)                 # (N, D)
        t_min_embed = self.interval_embedder(t_min)  # (N, D)
        t_max_embed = self.interval_embedder(t_max)  # (N, D)
        c_embed = self.c_embedder(c)                 # (N, D)
        cond_for_final_layer = t_embed + h_embed + w_embed + t_min_embed + t_max_embed + c_embed

        c_tokens = torch.concat(
            [
                replicate(t_embed, 4),        # (N, 4, D)
                replicate(h_embed, 4),        # (N, 4, D)
                replicate(w_embed, 4),        # (N, 4, D)
                replicate(t_min_embed, 2),    # (N, 2, D)
                replicate(t_max_embed, 2),    # (N, 2, D)
                replicate(c_embed, 8)         # (N, 8, D)
            ],
            dim=1
        )
        c_tokens = c_tokens + self.learnable_embed
        x = torch.concat([c_tokens, x_tokens], dim=1)

        ids = torch.cat((cond_ids, img_ids), dim=0)
        rotary_emb = self.pos_embed(ids)

        for block in self.blocks:
            x = block(x, rotary_emb)                    # (N, T, D)

        x = x[:, -x_seq:]
        x = self.final_layer(x, cond_for_final_layer)   # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                          # (N, out_channels, H, W)
        return x


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def iMF_XL(**kwargs) -> DiT:
    return DiT(depth=48, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def iMF_L(**kwargs) -> DiT:
    return DiT(depth=32, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def iMF_M(**kwargs) -> DiT:
    return DiT(depth=24, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def iMF_B(**kwargs) -> DiT:
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


if __name__ == '__main__':
    dit = iMF_B()
    params = sum(p.numel() for p in dit.parameters())
    print(f"params: {params}")

    x = torch.randn(size=(1, 4, 32, 32))
    t = torch.rand(size=(1,))
    h = torch.rand(size=(1,))
    w = torch.rand(size=(1,))
    t_min = torch.rand(size=(1,))
    t_max = torch.rand(size=(1,))
    c = torch.randint(low=0, high=999, size=(1,))
    img_ids = prepare_latent_image_ids(16, 16)
    cond_ids = prepare_cond_ids(24)

    res = dit(x, t, h, w, t_min, t_max, c, img_ids, cond_ids)
    print(f"res shape: {res.shape}")
