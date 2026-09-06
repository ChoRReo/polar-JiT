from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _resize_tensor(tensor: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    args = {"size": (int(image_size), int(image_size)), "mode": mode}
    if mode != "nearest":
        args["align_corners"] = False
    return F.interpolate(tensor[None], **args)[0]


def load_polarization_image(path, polarization_bits: int, image_size: int) -> torch.Tensor:
    """Load one analyzer image as resized RGB in [0,1]."""
    bits = int(polarization_bits)
    if bits < 1:
        raise ValueError("polarization_bits must be positive")
    with Image.open(path) as image:
        arr = np.asarray(image).copy()
    arr = arr.astype(np.float32) / float((1 << bits) - 1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"unsupported polarization image shape in {path}: {arr.shape}")
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return _resize_tensor(tensor, image_size, "bilinear").clamp(0, 1)


def load_mask_image(path, image_size: int) -> torch.Tensor:
    """Load a foreground mask as [1,H,W] with binary values."""
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L")).copy()
    tensor = torch.from_numpy((arr > 0).astype(np.float32))[None]
    return _resize_tensor(tensor, image_size, "nearest")


def analyzer_images_to_stokes(i0, i45, i90, i135):
    """Convert normalized analyzer images to network S0 and target S1/S2."""
    if not (i0.shape == i45.shape == i90.shape == i135.shape):
        raise ValueError("all analyzer images must have the same shape")
    s0_physical = ((i0 + i90) + (i45 + i135)) * 0.5
    s12 = torch.cat((i0 - i90, i45 - i135), dim=0).clamp(-1, 1)
    return s0_physical.clamp(0, 2).sub(1).float(), s12.float()


def load_stokes_scene(
    pol_000,
    pol_045,
    pol_090,
    pol_135,
    polarization_bits: int,
    image_size: int,
    mask=None,
):
    """Load a four-analyzer scene and return S0, S1/S2 and object mask."""
    images = [
        load_polarization_image(path, polarization_bits, image_size)
        for path in (pol_000, pol_045, pol_090, pol_135)
    ]
    s0, s12 = analyzer_images_to_stokes(*images)
    object_mask = (
        load_mask_image(mask, image_size)
        if mask is not None
        else torch.ones(1, int(image_size), int(image_size))
    )
    return s0, s12, object_mask.float()


class UnifiedSfPDataset(Dataset):
    """Independent reader for the existing UnifiedSfP manifest convention."""

    required = (
        "sample_id",
        "polarization_bits",
        "pol_000",
        "pol_045",
        "pol_090",
        "pol_135",
        "mask",
    )

    def __init__(self, root, split="train", image_size=256, sources=None, augment=False):
        self.root = Path(root)
        manifest = self.root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(key not in rows[0] for key in self.required):
            raise ValueError(f"invalid manifest columns: {manifest}")
        selected = set(sources or ())
        if selected:
            rows = [row for row in rows if row.get("source") in selected]
        if split == "train":
            rows = [
                r
                for r in rows
                if r.get("source") != "deepsfp" or r.get("subset") == "train"
            ]
        elif split == "test":
            test_subsets = {"test+test_supp", "test_supp"}
            rows = [
                r
                for r in rows
                if r.get("source") == "deepsfp" and r.get("subset") in test_subsets
            ]
        else:
            raise ValueError("split must be train or test")
        if not rows:
            raise RuntimeError(f"no samples for split={split}")
        self.rows, self.image_size, self.augment = rows, int(image_size), bool(augment)

    def __len__(self):
        return len(self.rows)

    def sample_name(self, index):
        """Return an identifier without loading the sample's image files."""
        return self.rows[index]["sample_id"]

    def _polar(self, row, key):
        return load_polarization_image(
            self.root / row[key], row["polarization_bits"], self.image_size
        )

    def _mask(self, row):
        return load_mask_image(self.root / row["mask"], self.image_size)

    def __getitem__(self, index):
        row = self.rows[index]
        keys = ("pol_000", "pol_045", "pol_090", "pol_135")
        i0, i45, i90, i135 = [self._polar(row, key) for key in keys]
        mask = self._mask(row)
        s0, s12 = analyzer_images_to_stokes(i0, i45, i90, i135)
        if self.augment and torch.rand(()) < 0.5:
            # Horizontal reflection changes the image-frame handedness: S2
            # changes sign while S0 and S1 do not.
            s0, s12, mask = [torch.flip(x, (-1,)) for x in (s0, s12, mask)]
            s12[3:].neg_()
        return {
            "name": self.sample_name(index),
            "source": row.get("source", "unknown"),
            "s0": s0,
            "s12": s12.float(),
            "mask": mask.float(),
        }


def build_dataset(config: dict, split=None):
    data = config["data"]
    if data.get("format", "unified_sfp") != "unified_sfp":
        raise ValueError("only data.format=unified_sfp is currently supported")
    chosen_split = split or data.get("split", "train")
    return UnifiedSfPDataset(
        data["root"],
        chosen_split,
        data.get("image_size", 256),
        data.get("sources"),
        augment=chosen_split == "train" and data.get("augment", True),
    )
