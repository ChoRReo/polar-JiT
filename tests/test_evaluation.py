import math

import pytest
import torch

from polar_jit.evaluation import evaluate_stokes_prediction
from polar_jit.metrics import masked_mae, masked_psnr, masked_ssim, periodic_aop_error


def test_periodic_aop_error_wraps_at_180_degrees():
    prediction = torch.tensor([math.radians(89.0)])
    target = torch.tensor([math.radians(-89.0)])

    error_degrees = periodic_aop_error(prediction, target) * 180 / math.pi

    assert torch.allclose(error_degrees, torch.tensor([2.0]), atol=1e-4)


def test_masked_metrics_ignore_background():
    target = torch.zeros(1, 1, 8, 8)
    prediction = torch.ones_like(target)
    mask = torch.zeros_like(target)
    mask[:, :, 2:6, 2:6] = 1
    prediction[:, :, 2:6, 2:6] = 0

    assert masked_mae(prediction, target, mask) == 0
    assert masked_psnr(prediction, target, mask) >= 100
    assert torch.allclose(
        masked_ssim(prediction, target, mask, window_size=3),
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_identity_stokes_evaluation():
    s0 = torch.zeros(1, 3, 8, 8)
    target = torch.rand(1, 6, 8, 8) - 0.5
    mask = torch.ones(1, 1, 8, 8)

    metrics = evaluate_stokes_prediction(
        target.clone(), target, s0, mask, window_size=3
    )

    assert set(metrics) == {
        "dolp_mae",
        "dolp_psnr",
        "dolp_ssim",
        "aop_mae_deg",
        "aop_psnr",
        "aop_ssim",
    }
    assert metrics["dolp_mae"] == 0
    assert metrics["aop_mae_deg"] == 0
    assert metrics["dolp_psnr"] >= 100
    assert metrics["aop_psnr"] >= 100
    assert metrics["dolp_ssim"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["aop_ssim"] == pytest.approx(1.0, abs=1e-5)


def test_evaluation_rejects_empty_mask_and_nonfinite_prediction():
    prediction = torch.zeros(1, 6, 4, 4)
    target = torch.zeros_like(prediction)
    s0 = torch.zeros(1, 3, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)

    with pytest.raises(ValueError, match="no foreground"):
        evaluate_stokes_prediction(prediction, target, s0, mask, window_size=3)

    mask.fill_(1)
    prediction[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        evaluate_stokes_prediction(prediction, target, s0, mask, window_size=3)
