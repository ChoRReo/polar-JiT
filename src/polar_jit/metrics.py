from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    weights = mask.float().expand_as(value)
    if not torch.any(weights > 0):
        return value.new_tensor(float("nan"))
    return (value * weights).sum() / weights.sum().clamp_min(eps)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((prediction - target).abs(), mask)


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    mse = masked_mean((prediction - target).square(), mask)
    return 10 * torch.log10(prediction.new_tensor(data_range**2) / mse.clamp_min(1e-12))


def _gaussian_kernel(size: int, sigma: float, channels: int, device, dtype):
    if size < 1 or size % 2 == 0:
        raise ValueError("SSIM window size must be a positive odd number")
    if sigma <= 0:
        raise ValueError("SSIM sigma must be positive")
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    gaussian = torch.exp(-(coords.square()) / (2 * sigma**2))
    kernel = gaussian[:, None] * gaussian[None, :]
    kernel = (kernel / kernel.sum()).to(dtype)
    return kernel.expand(channels, 1, size, size).contiguous()


def masked_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mask-aware SSIM; pixels outside the object do not enter local moments."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have the same [B,C,H,W] shape")
    channels = prediction.shape[1]
    weights = mask.float().expand_as(prediction)
    kernel = _gaussian_kernel(
        window_size, sigma, channels, prediction.device, prediction.dtype
    )
    padding = window_size // 2

    def filter_image(value):
        return F.conv2d(value, kernel, padding=padding, groups=channels)

    local_weight = filter_image(weights).clamp_min(1e-12)
    mu_x = filter_image(prediction * weights) / local_weight
    mu_y = filter_image(target * weights) / local_weight
    ex2 = filter_image(prediction.square() * weights) / local_weight
    ey2 = filter_image(target.square() * weights) / local_weight
    exy = filter_image(prediction * target * weights) / local_weight
    var_x = (ex2 - mu_x.square()).clamp_min(0)
    var_y = (ey2 - mu_y.square()).clamp_min(0)
    covariance = exy - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return masked_mean(score, weights)


def periodic_aop_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smallest AoP error in radians for a pi-periodic angle."""
    delta = 2 * (prediction - target)
    return 0.5 * torch.atan2(torch.sin(delta).abs(), torch.cos(delta))


def aop_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> dict[str, torch.Tensor]:
    error = periodic_aop_error(prediction, target)
    normalized_error = error / (math.pi / 2)
    mse = masked_mean(normalized_error.square(), mask)
    embedding_pred = torch.cat((torch.cos(2 * prediction), torch.sin(2 * prediction)), dim=1)
    embedding_target = torch.cat((torch.cos(2 * target), torch.sin(2 * target)), dim=1)
    # Map the periodic embedding to [0, 1] before SSIM.
    embedding_pred = (embedding_pred + 1) * 0.5
    embedding_target = (embedding_target + 1) * 0.5
    return {
        "mae_deg": masked_mean(error, mask) * (180 / math.pi),
        "psnr": -10 * torch.log10(mse.clamp_min(1e-12)),
        "ssim": masked_ssim(
            embedding_pred, embedding_target, mask, window_size=window_size, sigma=sigma
        ),
    }
