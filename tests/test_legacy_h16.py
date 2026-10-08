import torch

from polar_jit.legacy_h16 import LegacyPolarJiTH16, describe_legacy_state


def small_legacy_model():
    return LegacyPolarJiTH16(
        image_size=16,
        patch_size=8,
        hidden_size=64,
        depth=2,
        num_heads=4,
        bottleneck_dim=16,
        refiner_hidden_channels=12,
    )


def test_legacy_model_keeps_0e9_checkpoint_layout():
    model = small_legacy_model()
    state = model.state_dict()

    assert "refiner.in_conv.weight" in state
    assert "refiner.out_conv.weight" in state
    assert "refiner.condition_conv.weight" not in state
    assert "in_context_posemb" not in state
    assert describe_legacy_state(state) == {
        "hidden_size": 64,
        "depth": 2,
        "bottleneck_dim": 16,
        "patch_size": 8,
        "target_channels": 6,
        "refiner_hidden_channels": 12,
        "has_in_context": False,
        "has_s0_refiner": False,
    }


def test_legacy_checkpoint_roundtrip_and_forward():
    original = small_legacy_model()
    restored = small_legacy_model()
    restored.load_state_dict(original.state_dict(), strict=True)

    output = restored(
        torch.randn(1, 6, 16, 16),
        torch.full((1,), 0.5),
        torch.randn(1, 3, 16, 16),
    )["clean"]
    assert output.shape == (1, 6, 16, 16)
    assert torch.isfinite(output).all()
