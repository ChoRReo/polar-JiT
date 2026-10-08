from __future__ import annotations

import torch

from .losses import _stable_aop, high_frequency_l1, masked_mean, spatial_gradient_l1
from .ratio_polarization import ratio_dolp_aop


def reconstruction_losses_ratio(
    clean,
    target,
    s0,
    weights,
    patch_size=16,
    patch_boundary_weight=4.0,
    high_frequency_kernel_sizes=(3, 7),
):
    """Reconstruction losses for normalized S1/S0,S2/S0 targets."""
    l1 = masked_mean((clean - target).abs(), weights)
    gradient_l1 = spatial_gradient_l1(
        clean, target, weights, patch_size, patch_boundary_weight
    )
    frequency_l1 = high_frequency_l1(
        clean, target, weights, kernel_sizes=high_frequency_kernel_sizes
    )
    pred_dolp, _ = ratio_dolp_aop(clean, s0)
    gt_dolp, _ = ratio_dolp_aop(target, s0)
    dolp_l1 = masked_mean((pred_dolp - gt_dolp).abs(), weights)
    pred_aop, gt_aop = _stable_aop(clean), _stable_aop(target)
    half_period = torch.pi / 2
    delta = torch.remainder(pred_aop - gt_aop + half_period, torch.pi) - half_period
    gt_s1, gt_s2 = target[:, :3].float(), target[:, 3:].float()
    confidence = torch.sqrt(gt_s1.square() + gt_s2.square()).clamp(0, 1).detach()
    aop_l1 = masked_mean(delta.abs(), weights.float() * confidence)
    return l1, gradient_l1, frequency_l1, dolp_l1, aop_l1
