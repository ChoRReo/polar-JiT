from __future__ import annotations

import torch


def s12_dolp_aop(s12: torch.Tensor, s0: torch.Tensor, eps: float = 1e-6):
    """Compute per-channel DoLP/AoP directly from RGB S0/S1/S2 Stokes tensors."""
    if s12.ndim != 4 or s12.shape[1] != 6:
        raise ValueError(f"expected [B,6,H,W], got {tuple(s12.shape)}")
    if s0.ndim != 4 or s0.shape[1] != 3 or s0.shape[0] != s12.shape[0]:
        raise ValueError(f"expected matching S0 [B,3,H,W], got {tuple(s0.shape)}")
    if s0.shape[-2:] != s12.shape[-2:]:
        raise ValueError("S0 and S1/S2 must have the same spatial size")

    s1, s2 = s12[:, :3], s12[:, 3:]
    amplitude = torch.sqrt((s1.square() + s2.square()).clamp_min(0))
    s0_linear = s0.add(1).clamp(0, 2)
    dolp = (amplitude / s0_linear.clamp_min(eps)).clamp(0, 1)
    aop = 0.5 * torch.atan2(s2, s1)
    return dolp, aop
