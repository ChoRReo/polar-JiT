from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


SCENE_TENSORS = {
    "prediction": "prediction_s12.npy",
    "target": "target_s12.npy",
    "s0": "s0.npy",
    "mask": "mask.npy",
}


def save_scene_bundle(
    output_dir,
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    s0: torch.Tensor,
    mask: torch.Tensor,
    metadata: dict | None = None,
) -> dict[str, Path]:
    """Save a prediction and its evaluation inputs as float32 NPY files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensors = {"prediction": prediction, "target": target, "s0": s0, "mask": mask}
    paths = {}
    for key, tensor in tensors.items():
        path = output / SCENE_TENSORS[key]
        np.save(path, tensor.detach().float().cpu().numpy().astype(np.float32))
        paths[key] = path
    scene_metadata = {"format": "polar_jit_scene_v1", "name": name, **(metadata or {})}
    metadata_path = output / "scene.json"
    metadata_path.write_text(
        json.dumps(scene_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["metadata"] = metadata_path
    return paths


def load_scene_bundle(scene_dir) -> dict:
    """Load and validate a bundle produced by the single-scene inference script."""
    root = Path(scene_dir)
    missing = [
        filename for filename in SCENE_TENSORS.values() if not (root / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"incomplete scene bundle {root}; missing: {missing}")
    metadata_path = root / "scene.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {"name": root.name}
    )
    tensors = {
        key: torch.from_numpy(np.load(root / filename, allow_pickle=False)).float()
        for key, filename in SCENE_TENSORS.items()
    }
    prediction, target = tensors["prediction"], tensors["target"]
    s0, mask = tensors["s0"], tensors["mask"]
    if (
        prediction.ndim != 3
        or prediction.shape[0] != 6
        or target.shape != prediction.shape
    ):
        raise ValueError("scene prediction and target must have the same [6,H,W] shape")
    spatial = prediction.shape[-2:]
    if s0.shape != (3, *spatial):
        raise ValueError("scene S0 must have shape [3,H,W]")
    if mask.shape == spatial:
        mask = mask[None]
    if mask.shape != (1, *spatial):
        raise ValueError("scene mask must have shape [1,H,W] or [H,W]")
    tensors["mask"] = mask
    tensors["name"] = str(metadata.get("name") or root.name)
    tensors["metadata"] = metadata
    return tensors
