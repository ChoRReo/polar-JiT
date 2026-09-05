from __future__ import annotations

import torch

from .metrics import aop_metrics, masked_mae, masked_psnr, masked_ssim
from .polarization import s12_dolp_aop


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def evaluate_stokes_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    s0: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> dict[str, float]:
    """Evaluate batched S1/S2 predictions inside an object mask."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 6:
        raise ValueError("prediction and target must have the same [B,6,H,W] shape")
    expected_s0 = (prediction.shape[0], 3, *prediction.shape[-2:])
    expected_mask = (prediction.shape[0], 1, *prediction.shape[-2:])
    if tuple(s0.shape) != expected_s0:
        raise ValueError(f"s0 must have shape {expected_s0}")
    if tuple(mask.shape) != expected_mask:
        raise ValueError(f"mask must have shape {expected_mask}")
    for name, value in (
        ("prediction", prediction),
        ("target", target),
        ("s0", s0),
        ("mask", mask),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if not torch.any(mask > 0):
        raise ValueError("evaluation mask has no foreground pixels")

    pred_dolp, pred_aop = s12_dolp_aop(prediction, s0)
    gt_dolp, gt_aop = s12_dolp_aop(target, s0)
    aop = aop_metrics(pred_aop, gt_aop, mask, window_size, sigma)
    return {
        "dolp_mae": _scalar(masked_mae(pred_dolp, gt_dolp, mask)),
        "dolp_psnr": _scalar(masked_psnr(pred_dolp, gt_dolp, mask)),
        "dolp_ssim": _scalar(
            masked_ssim(
                pred_dolp,
                gt_dolp,
                mask,
                window_size=window_size,
                sigma=sigma,
            )
        ),
        "aop_mae_deg": _scalar(aop["mae_deg"]),
        "aop_psnr": _scalar(aop["psnr"]),
        "aop_ssim": _scalar(aop["ssim"]),
    }
