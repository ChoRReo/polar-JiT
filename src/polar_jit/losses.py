from __future__ import annotations

import torch

from .polarization import s12_dolp_aop


def masked_mean(value, mask, eps=1e-6):
    value = value.float()
    mask = mask.float().expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def generation_weights(mask, object_weight=1.0, background_weight=0.0):
    """Build relative spatial weights for generation losses."""
    if object_weight < 0 or background_weight < 0:
        raise ValueError("object_weight and background_weight must be non-negative")
    if object_weight == 0 and background_weight == 0:
        raise ValueError("at least one spatial weight must be positive")
    foreground = mask.float().clamp(0, 1)
    return foreground * object_weight + (1 - foreground) * background_weight


def spatial_gradient_l1(
    clean,
    target,
    weights,
    patch_size=16,
    patch_boundary_weight=4.0,
):
    """Match S1/S2 image gradients, emphasizing former patch boundaries."""
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if patch_boundary_weight < 1:
        raise ValueError("patch_boundary_weight must be at least 1")

    clean, target, weights = clean.float(), target.float(), weights.float()
    clean_dx = clean[..., 1:] - clean[..., :-1]
    target_dx = target[..., 1:] - target[..., :-1]
    clean_dy = clean[..., 1:, :] - clean[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]

    # A pair receives foreground weight only when both of its pixels are in
    # the object. This prevents object/background contours dominating the loss.
    weight_x = torch.minimum(weights[..., 1:], weights[..., :-1])
    weight_y = torch.minimum(weights[..., 1:, :], weights[..., :-1, :])
    if patch_boundary_weight != 1:
        x_positions = torch.arange(1, clean.shape[-1], device=clean.device)
        y_positions = torch.arange(1, clean.shape[-2], device=clean.device)
        x_boost = torch.where(
            x_positions.remainder(patch_size) == 0,
            patch_boundary_weight,
            1.0,
        ).to(clean.dtype)
        y_boost = torch.where(
            y_positions.remainder(patch_size) == 0,
            patch_boundary_weight,
            1.0,
        ).to(clean.dtype)
        weight_x = weight_x * x_boost[None, None, None, :]
        weight_y = weight_y * y_boost[None, None, :, None]

    loss_x = masked_mean((clean_dx - target_dx).abs(), weight_x)
    loss_y = masked_mean((clean_dy - target_dy).abs(), weight_y)
    return 0.5 * (loss_x + loss_y)


def _stable_aop(s12, eps=1e-4):
    """Compute AoP without differentiating atan2 at an unpolarized vector."""
    s1, s2 = s12[:, :3].float(), s12[:, 3:].float()
    polarized = s1.square() + s2.square() >= eps**2
    safe_s1 = torch.where(polarized, s1, torch.full_like(s1, eps))
    safe_s2 = torch.where(polarized, s2, torch.zeros_like(s2))
    return 0.5 * torch.atan2(safe_s2, safe_s1)


def reconstruction_losses(
    clean,
    target,
    s0,
    weights,
    patch_size=16,
    patch_boundary_weight=4.0,
):
    l1 = masked_mean((clean - target).abs(), weights)
    gradient_l1 = spatial_gradient_l1(
        clean,
        target,
        weights,
        patch_size=patch_size,
        patch_boundary_weight=patch_boundary_weight,
    )
    pred_dolp, _ = s12_dolp_aop(clean, s0)
    gt_dolp, _ = s12_dolp_aop(target, s0)
    dolp_l1 = masked_mean((pred_dolp - gt_dolp).abs(), weights)

    pred_aop, gt_aop = _stable_aop(clean), _stable_aop(target)
    half_period = torch.pi / 2
    aop_delta = torch.remainder(pred_aop - gt_aop + half_period, torch.pi) - half_period

    # AoP is undefined when the target is unpolarized, so its periodic L1
    # distance is weighted by target DoLP confidence.
    gt_s1, gt_s2 = target[:, :3].float(), target[:, 3:].float()
    gt_amplitude = torch.sqrt(gt_s1.square() + gt_s2.square())
    gt_intensity = s0.float().add(1).clamp_min(1e-6)
    gt_confidence = (gt_amplitude / gt_intensity).clamp(0, 1).detach()
    aop_weights = weights.float() * gt_confidence
    aop_l1 = masked_mean(aop_delta.abs(), aop_weights)
    return l1, gradient_l1, dolp_l1, aop_l1
