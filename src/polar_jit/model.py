from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
    half = dim // 2
    frequency = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    embedding = torch.cat(
        (torch.cos(t[:, None].float() * frequency), torch.sin(t[:, None].float() * frequency)),
        dim=-1,
    )
    return F.pad(embedding, (0, dim % 2))


def sincos_2d(dim: int, height: int, width: int):
    if dim % 4:
        raise ValueError("hidden size must be divisible by 4")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4)
    omega = 1.0 / (10_000**omega)
    out_x, out_y = x.flatten()[:, None] * omega, y.flatten()[:, None] * omega
    return torch.cat((out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()), dim=1)[None]


def rotate_half(x):
    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding2D(nn.Module):
    """Parameter-free 2D RoPE matching the official JiT feature layout."""

    def __init__(self, head_dim: int, grid_size: int):
        super().__init__()
        if head_dim % 4:
            raise ValueError("attention head dimension must be divisible by 4")
        axis_dim = head_dim // 2
        frequency = 1.0 / (
            10_000 ** (torch.arange(0, axis_dim, 2).float() / axis_dim)
        )
        position = torch.arange(grid_size).float()
        angles = torch.einsum("i,j->ij", position, frequency).repeat_interleave(2, dim=-1)
        angles_y = angles[:, None, :].expand(grid_size, grid_size, axis_dim)
        angles_x = angles[None, :, :].expand(grid_size, grid_size, axis_dim)
        angles_2d = torch.cat((angles_y, angles_x), dim=-1).reshape(-1, head_dim)
        self.register_buffer("cos", angles_2d.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles_2d.sin()[None, None], persistent=False)

    def forward(self, x):
        return x * self.cos.to(x.dtype) + rotate_half(x) * self.sin.to(x.dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size).to(t.dtype))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size, bottleneck_dim):
        super().__init__()
        self.proj1 = nn.Conv2d(
            in_channels, bottleneck_dim, patch_size, stride=patch_size, bias=False
        )
        self.proj2 = nn.Conv2d(bottleneck_dim, hidden_size, 1)

    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden size must be divisible by number of heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = float(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, rope):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_dropout(self.proj(output))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * ratio * 2 / 3)
        self.w12 = nn.Linear(dim, hidden * 2)
        self.w3 = nn.Linear(hidden, dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        first, second = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(first) * second))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class JiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, heads, attn_dropout, proj_dropout)
        self.mlp = SwiGLUFFN(dim, mlp_ratio, proj_dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(self, x, condition, rope):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        x = x + gate_attn[:, None] * self.attn(
            modulate(self.norm1(x), shift_attn, scale_attn), rope
        )
        return x + gate_mlp[:, None] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2)
        )

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class PolarJiT(nn.Module):
    """JiT-B/16 backbone with spatial and global S0 conditioning."""

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        target_channels=6,
        condition_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        bottleneck_dim=128,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if patch_size < 2 or patch_size & (patch_size - 1):
            raise ValueError("patch_size must be a power of two >= 2")
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = target_channels
        self.x_embedder = BottleneckPatchEmbed(
            target_channels, hidden_size, patch_size, bottleneck_dim
        )
        self.s0_embedder = BottleneckPatchEmbed(
            condition_channels, hidden_size, patch_size, bottleneck_dim
        )
        grid = image_size // patch_size
        self.pos_embed = nn.Parameter(
            sincos_2d(hidden_size, grid, grid), requires_grad=False
        )
        self.rope = RotaryEmbedding2D(hidden_size // num_heads, grid)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.condition_pool = nn.Sequential(
            RMSNorm(hidden_size), nn.Linear(hidden_size, hidden_size)
        )
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    attn_dropout=attn_dropout if depth // 4 <= i < depth * 3 // 4 else 0.0,
                    proj_dropout=proj_dropout if depth // 4 <= i < depth * 3 // 4 else 0.0,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, target_channels)
        self.reset_parameters()

    def reset_parameters(self):
        def initialize(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, tokens):
        batch, count, _ = tokens.shape
        height = width = int(math.sqrt(count))
        if height * width != count:
            raise ValueError("image token count must form a square grid")
        patch, channels = self.patch_size, self.out_channels
        image = tokens.reshape(batch, height, width, patch, patch, channels)
        image = image.permute(0, 5, 1, 3, 2, 4)
        return image.reshape(batch, channels, height * patch, width * patch)

    def forward(self, x_t, t, s0):
        expected = (self.image_size, self.image_size)
        if x_t.shape[-2:] != expected or s0.shape[-2:] != expected:
            raise ValueError(f"expected spatial size {expected}")
        condition_tokens = self.s0_embedder(s0)
        x = self.x_embedder(x_t) + condition_tokens + self.pos_embed.to(x_t.dtype)
        condition = self.t_embedder(t)
        condition = condition + self.condition_pool(condition_tokens.mean(dim=1))
        for block in self.blocks:
            x = block(x, condition, self.rope)
        clean = self.unpatchify(self.final_layer(x, condition))
        return {"clean": clean}
