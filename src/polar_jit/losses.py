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


def reconstruction_losses(clean, target, s0, weights):
    l1 = masked_mean((clean - target).abs(), weights)
    pred_dolp, _ = s12_dolp_aop(clean, s0)
    gt_dolp, _ = s12_dolp_aop(target, s0)
    dolp = masked_mean((pred_dolp - gt_dolp).abs(), weights)

    # Since (S1,S2) points at 2*AoP, its cosine similarity is exactly
    # cos(2*(pred_aop-target_aop)). This avoids differentiating atan2(0,0),
    # which occurs at initialization because JiT's output layer starts at zero.
    pred_s1, pred_s2 = clean[:, :3].float(), clean[:, 3:].float()
    gt_s1, gt_s2 = target[:, :3].float(), target[:, 3:].float()
    direction_eps = 1e-4
    pred_norm = torch.sqrt(pred_s1.square() + pred_s2.square() + direction_eps**2)
    gt_norm = torch.sqrt(gt_s1.square() + gt_s2.square()).clamp_min(direction_eps)
    direction_similarity = (pred_s1 * gt_s1 + pred_s2 * gt_s2) / (
        pred_norm * gt_norm
    )
    periodic = 1 - direction_similarity.clamp(-1, 1)
    # AoP is undefined when the target is unpolarized. Weight its loss by
    # target DoLP so those pixels do not inject arbitrary angle supervision.
    gt_amplitude = torch.sqrt(gt_s1.square() + gt_s2.square())
    gt_intensity = s0.float().add(1).clamp_min(1e-6)
    gt_confidence = (gt_amplitude / gt_intensity).clamp(0, 1).detach()
    aop_weights = weights.float() * gt_confidence
    aop = masked_mean(periodic, aop_weights)
    return l1, dolp, aop
