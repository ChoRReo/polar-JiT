import csv

import numpy as np
import torch
from PIL import Image

from polar_jit.data import UnifiedSfPDataset, load_stokes_scene, oracle_mgt_condition


def _save_rgb(path, value):
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def test_dataset_builds_stokes_without_normal_columns(tmp_path):
    values = {"i0.png": 191, "i45.png": 127, "i90.png": 63, "i135.png": 127}
    for name, value in values.items():
        _save_rgb(tmp_path / name, value)
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(tmp_path / "mask.png")

    fieldnames = [
        "sample_id",
        "source",
        "subset",
        "polarization_bits",
        "pol_000",
        "pol_045",
        "pol_090",
        "pol_135",
        "mask",
    ]
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "scene/sample_001",
                "source": "sfpuel",
                "subset": "train",
                "polarization_bits": "8",
                "pol_000": "i0.png",
                "pol_045": "i45.png",
                "pol_090": "i90.png",
                "pol_135": "i135.png",
                "mask": "mask.png",
            }
        )

    sample = UnifiedSfPDataset(tmp_path, split="train", image_size=4, augment=False)[0]

    assert set(sample) == {"name", "source", "s0", "s12", "mask"}
    assert sample["name"] == "scene/sample_001"
    assert sample["s0"].shape == (3, 4, 4)
    assert sample["s12"].shape == (6, 4, 4)
    assert torch.allclose(sample["s0"], torch.full((3, 4, 4), -1 / 255))
    assert torch.allclose(sample["s12"][:3], torch.full((3, 4, 4), 128 / 255))
    assert torch.count_nonzero(sample["s12"][3:]) == 0
    assert torch.all(sample["mask"] == 1)

    scene_s0, scene_s12, scene_mask = load_stokes_scene(
        tmp_path / "i0.png",
        tmp_path / "i45.png",
        tmp_path / "i90.png",
        tmp_path / "i135.png",
        polarization_bits=8,
        image_size=4,
    )
    assert torch.equal(scene_s0, sample["s0"])
    assert torch.equal(scene_s12, sample["s12"])
    assert torch.all(scene_mask == 1)


def test_test_split_contains_deepsfp_test_and_supplement(tmp_path):
    for name in ("i0.png", "i45.png", "i90.png", "i135.png"):
        _save_rgb(tmp_path / name, 127)
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(tmp_path / "mask.png")
    fieldnames = [
        "sample_id",
        "source",
        "subset",
        "polarization_bits",
        "pol_000",
        "pol_045",
        "pol_090",
        "pol_135",
        "mask",
    ]
    common = {
        "source": "deepsfp",
        "polarization_bits": "8",
        "pol_000": "i0.png",
        "pol_045": "i45.png",
        "pol_090": "i90.png",
        "pol_135": "i135.png",
        "mask": "mask.png",
    }
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({**common, "sample_id": "main", "subset": "test+test_supp"})
        writer.writerow({**common, "sample_id": "supp", "subset": "test_supp"})
        writer.writerow({**common, "sample_id": "train", "subset": "train"})

    dataset = UnifiedSfPDataset(tmp_path, split="test", image_size=4, augment=False)

    assert [row["sample_id"] for row in dataset.rows] == ["main", "supp"]


def test_oracle_mgt_condition_uses_plan_sign_and_seven_channels():
    s0 = torch.zeros(3, 2, 2)
    s12 = torch.zeros(6, 2, 2)
    s12[:3] = 1
    normal = torch.zeros(3, 2, 2)
    normal[0] = 1  # phi=0, theta=pi/2
    mask = torch.ones(1, 2, 2)

    oracle = oracle_mgt_condition(s0, s12, normal, mask)

    assert oracle["condition"].shape == (7, 2, 2)
    assert torch.allclose(oracle["theta"], torch.full((1, 2, 2), 0.5))
    assert torch.allclose(oracle["cos2phi"], torch.ones(1, 2, 2))
    assert torch.allclose(oracle["sin2phi"], torch.zeros(1, 2, 2))
    assert torch.allclose(oracle["mgt"], torch.ones(1, 2, 2))

    # For phi=pi/4 and a pure +S2 direction, the plan's minus sign gives -1.
    diagonal = torch.full((2, 2), 2**-0.5)
    normal[0], normal[1] = diagonal, diagonal
    s12[:3].zero_()
    s12[3:] = 1
    oracle = oracle_mgt_condition(s0, s12, normal, mask)
    assert torch.allclose(oracle["mgt"], -torch.ones(1, 2, 2), atol=1e-6)
