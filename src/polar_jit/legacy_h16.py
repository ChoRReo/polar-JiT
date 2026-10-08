from __future__ import annotations

import math

import torch
from torch import nn

from .model import (
    BottleneckPatchEmbed,
    FinalLayer,
    JiTBlock,
    RotaryEmbedding2D,
    RMSNorm,
    TimestepEmbedder,
    sincos_2d,
)


class LegacyResidualRefinementHead(nn.Module):
    """Refiner used by commit 0e9be44; it does not read S0 directly."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("refiner hidden channels must be positive")
        self.in_conv = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.out_conv = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

    def forward(self, image):
        return image + self.out_conv(self.activation(self.in_conv(image)))


class LegacyPolarJiTH16(nn.Module):
    """Checkpoint-compatible 0e9be44 JiT architecture enlarged to H/16.

    This intentionally omits the later in-context tokens and the full-resolution
    S0 branch in the refiner. Parameter names match the historical checkpoint.
    """

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        target_channels=6,
        condition_channels=3,
        hidden_size=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        bottleneck_dim=256,
        refiner_hidden_channels=64,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if patch_size < 2 or patch_size & (patch_size - 1):
            raise ValueError("patch_size must be a power of two >= 2")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.out_channels = int(target_channels)
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
                    attn_dropout=(
                        attn_dropout if depth // 4 <= index < depth * 3 // 4 else 0.0
                    ),
                    proj_dropout=(
                        proj_dropout if depth // 4 <= index < depth * 3 // 4 else 0.0
                    ),
                )
                for index in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, target_channels)
        self.refiner = LegacyResidualRefinementHead(
            target_channels, refiner_hidden_channels
        )
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
        nn.init.zeros_(self.refiner.out_conv.weight)
        nn.init.zeros_(self.refiner.out_conv.bias)

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
        return {"clean": self.refiner(clean)}


def strip_checkpoint_prefix(state_dict: dict[str, torch.Tensor]):
    """Accept checkpoints saved from plain, DDP, or torch.compile models."""
    state = dict(state_dict)
    for prefix in ("module.", "_orig_mod."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def describe_legacy_state(state_dict: dict[str, torch.Tensor]):
    """Infer the architecture fields that are observable in a state dict."""
    state = strip_checkpoint_prefix(state_dict)
    block_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    patch_weight = state.get("x_embedder.proj1.weight")
    projection = state.get("x_embedder.proj2.weight")
    refiner = state.get("refiner.in_conv.weight")
    return {
        "hidden_size": None if projection is None else int(projection.shape[0]),
        "depth": 0 if not block_indices else max(block_indices) + 1,
        "bottleneck_dim": None if patch_weight is None else int(patch_weight.shape[0]),
        "patch_size": None if patch_weight is None else int(patch_weight.shape[-1]),
        "target_channels": None if patch_weight is None else int(patch_weight.shape[1]),
        "refiner_hidden_channels": None if refiner is None else int(refiner.shape[0]),
        "has_in_context": "in_context_posemb" in state,
        "has_s0_refiner": "refiner.condition_conv.weight" in state,
    }
