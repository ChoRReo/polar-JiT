import torch

from polar_jit.ratio_data import raw_s12_to_ratio
from polar_jit.ratio_losses import reconstruction_losses_ratio
from polar_jit.ratio_polarization import ratio_dolp_aop


def test_raw_stokes_conversion_to_ratio():
    s0 = torch.full((3, 2, 2), -0.5)  # physical S0 = 0.5
    raw = torch.full((6, 2, 2), 0.25)
    ratio = raw_s12_to_ratio(s0, raw)
    assert torch.allclose(ratio, torch.full_like(ratio, 0.5))


def test_ratio_dolp_is_not_divided_by_s0_twice():
    s0 = torch.full((1, 3, 2, 2), -0.5)
    ratio = torch.zeros(1, 6, 2, 2)
    ratio[:, :3] = 0.25
    dolp, aop = ratio_dolp_aop(ratio, s0)
    assert torch.allclose(dolp, torch.full_like(dolp, 0.25))
    assert torch.allclose(aop, torch.zeros_like(aop))


def test_ratio_losses_have_finite_zero_prediction_gradient():
    prediction = torch.zeros(1, 6, 8, 8, requires_grad=True)
    target = torch.rand_like(prediction).mul(2).sub(1)
    losses = reconstruction_losses_ratio(
        prediction,
        target,
        torch.zeros(1, 3, 8, 8),
        torch.ones(1, 1, 8, 8),
    )
    sum(losses).backward()
    assert all(torch.isfinite(loss) for loss in losses)
    assert torch.isfinite(prediction.grad).all()
