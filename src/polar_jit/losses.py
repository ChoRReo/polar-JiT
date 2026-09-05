from __future__ import annotations

import torch

from .polarization import s12_dolp_aop


def masked_mean(value, mask, eps=1e-6):
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def generation_weights(mask, object_weight=1.0, background_weight=0.0):
    """Build relative spatial weights for generation losses."""
    if object_weight < 0 or background_weight < 0:
        raise ValueError("object_weight and background_weight must be non-negative")
    if object_weight == 0 and background_weight == 0:
        raise ValueError("at least one spatial weight must be positive")
    foreground = mask.float().clamp(0, 1)
    return foreground * object_weight + (1 - foreground) * background_weight


def reconstruction_losses(clean, target, s0, weights):
    l1 = masked_mean((clean - target).abs(), weights)
    pred_dolp, pred_aop = s12_dolp_aop(clean, s0)
    gt_dolp, gt_aop = s12_dolp_aop(target, s0)
    dolp = masked_mean((pred_dolp - gt_dolp).abs(), weights)
    periodic = 1 - torch.cos(2 * (pred_aop - gt_aop))
    aop = masked_mean(periodic, weights)
    return l1, dolp, aop
