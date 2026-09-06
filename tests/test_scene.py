import torch

from polar_jit.scene import load_scene_bundle, save_scene_bundle


def test_scene_bundle_round_trip(tmp_path):
    prediction = torch.rand(6, 8, 8)
    target = torch.rand(6, 8, 8)
    s0 = torch.rand(3, 8, 8)
    mask = torch.ones(1, 8, 8)

    paths = save_scene_bundle(
        tmp_path,
        name="example",
        prediction=prediction,
        target=target,
        s0=s0,
        mask=mask,
        metadata={"polarization_bits": 8},
    )
    loaded = load_scene_bundle(tmp_path)

    assert all(path.is_file() for path in paths.values())
    assert loaded["name"] == "example"
    assert loaded["metadata"]["polarization_bits"] == 8
    assert torch.equal(loaded["prediction"], prediction)
    assert torch.equal(loaded["target"], target)
    assert torch.equal(loaded["s0"], s0)
    assert torch.equal(loaded["mask"], mask)
