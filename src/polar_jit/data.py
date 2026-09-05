from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


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

    def _resize(self, x, mode):
        args = {"size": (self.image_size, self.image_size), "mode": mode}
        if mode != "nearest":
            args["align_corners"] = False
        return F.interpolate(x[None], **args)[0]

    def _polar(self, row, key):
        with Image.open(self.root / row[key]) as image:
            arr = np.asarray(image).copy()
        scale = float((1 << int(row["polarization_bits"])) - 1)
        arr = arr.astype(np.float32) / scale
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return self._resize(tensor, "bilinear").clamp(0, 1)

    def _mask(self, row):
        with Image.open(self.root / row["mask"]) as image:
            arr = np.asarray(image.convert("L")).copy()
        return self._resize(torch.from_numpy((arr > 0).astype(np.float32))[None], "nearest")

    def __getitem__(self, index):
        row = self.rows[index]
        keys = ("pol_000", "pol_045", "pol_090", "pol_135")
        i0, i45, i90, i135 = [self._polar(row, key) for key in keys]
        mask = self._mask(row)
        s0 = ((i0 + i90) + (i45 + i135)) * 0.5
        s12 = torch.cat((i0 - i90, i45 - i135), dim=0).clamp(-1, 1)
        if self.augment and torch.rand(()) < 0.5:
            # Horizontal reflection changes the image-frame handedness: S2
            # changes sign while S0 and S1 do not.
            s0, s12, mask = [torch.flip(x, (-1,)) for x in (s0, s12, mask)]
            s12[3:].neg_()
        return {
            "name": self.sample_name(index),
            "source": row.get("source", "unknown"),
            # Physical S0 is in [0, 2]; subtracting one puts all network-space
            # Stokes components in [-1, 1] without changing their common scale.
            "s0": s0.clamp(0, 2).sub(1).float(),
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
