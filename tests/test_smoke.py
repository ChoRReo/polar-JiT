import torch

from polar_jit import ConditionalFlowMatcher, PolarJiT, s12_dolp_aop
from polar_jit.losses import generation_weights, reconstruction_losses, spatial_gradient_l1
from polar_jit.metrics import aop_metrics, masked_mae, masked_psnr, masked_ssim


def test_model_and_flow_cpu():
    model = PolarJiT(image_size=32, patch_size=8, hidden_size=64, depth=2, num_heads=4,
                     bottleneck_dim=16)
    flow = ConditionalFlowMatcher(model)
    s0, target = torch.randn(2, 3, 32, 32), torch.randn(2, 6, 32, 32).clamp(-1, 1)
    pred, velocity, velocity_target, _ = flow(target, s0)
    assert pred["clean"].shape == target.shape == velocity.shape == velocity_target.shape
    assert set(pred) == {"clean"}
    (pred["clean"] - target).square().mean().backward()


def test_refiner_starts_as_identity():
    model = PolarJiT(
        image_size=16,
        patch_size=8,
        hidden_size=32,
        depth=1,
        num_heads=4,
        bottleneck_dim=8,
        refiner_hidden_channels=12,
    )
    image = torch.randn(1, 6, 16, 16)
    assert torch.equal(model.refiner(image), image)


def test_stokes_dolp_aop():
    s0 = torch.full((1, 3, 4, 4), -0.5)  # network S0=-0.5 means physical S0=0.5
    s12 = torch.zeros(1, 6, 4, 4)
    s12[:, :3] = 0.25
    dolp, aop = s12_dolp_aop(s12, s0)
    assert torch.allclose(dolp, torch.full_like(dolp, 0.5))
    assert torch.allclose(aop, torch.zeros_like(aop))


def test_generation_weights_prioritize_object():
    mask = torch.tensor([[[[1.0, 0.0]]]])
    weights = generation_weights(mask, object_weight=10.0, background_weight=0.01)
    assert weights[0, 0, 0, 0] == 10
    assert weights[0, 0, 0, 1] == 0.01


def test_polarization_losses_have_finite_gradient_at_zero_prediction():
    prediction = torch.zeros(2, 6, 8, 8, requires_grad=True)
    target = torch.rand_like(prediction).mul(2).sub(1)
    s0 = torch.zeros(2, 3, 8, 8)
    weights = torch.ones(2, 1, 8, 8)

    losses = reconstruction_losses(prediction, target, s0, weights)
    sum(losses).backward()

    assert all(torch.isfinite(loss) for loss in losses)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_aop_l1_uses_shortest_pi_periodic_distance():
    angle_pred = torch.deg2rad(torch.tensor(89.0))
    angle_target = torch.deg2rad(torch.tensor(-89.0))
    prediction = torch.cat(
        (
            torch.cos(2 * angle_pred).expand(1, 3, 2, 2),
            torch.sin(2 * angle_pred).expand(1, 3, 2, 2),
        ),
        dim=1,
    )
    target = torch.cat(
        (
            torch.cos(2 * angle_target).expand(1, 3, 2, 2),
            torch.sin(2 * angle_target).expand(1, 3, 2, 2),
        ),
        dim=1,
    )
    s0 = torch.zeros(1, 3, 2, 2)
    weights = torch.ones(1, 1, 2, 2)
    _, _, _, aop_l1 = reconstruction_losses(prediction, target, s0, weights)
    assert torch.allclose(aop_l1, torch.deg2rad(torch.tensor(2.0)), atol=1e-5)


def test_gradient_l1_emphasizes_patch_boundaries():
    prediction = torch.zeros(1, 6, 4, 4)
    prediction[..., 2:] = 1
    target = torch.zeros_like(prediction)
    weights = torch.ones(1, 1, 4, 4)
    regular = spatial_gradient_l1(
        prediction, target, weights, patch_size=2, patch_boundary_weight=1
    )
    emphasized = spatial_gradient_l1(
        prediction, target, weights, patch_size=2, patch_boundary_weight=4
    )
    assert emphasized > regular


def test_identity_metrics():
    image = torch.rand(1, 1, 16, 16)
    mask = torch.ones_like(image)
    assert masked_mae(image, image, mask) == 0
    assert masked_psnr(image, image, mask) >= 100
    assert torch.allclose(masked_ssim(image, image, mask), torch.tensor(1.0), atol=1e-5)
    metrics = aop_metrics(image, image + torch.pi, mask)
    assert metrics["mae_deg"] < 1e-4
    assert metrics["psnr"] >= 100
    assert torch.allclose(metrics["ssim"], torch.tensor(1.0), atol=1e-5)
