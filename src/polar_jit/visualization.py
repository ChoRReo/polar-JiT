from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .polarization import s12_dolp_aop


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _heatmap(value: np.ndarray) -> np.ndarray:
    """Small blue-cyan-yellow-red colormap with no plotting dependency."""
    value = np.clip(value, 0, 1)
    stops = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float32)
    colors = np.array(
        [[20, 20, 80], [0, 180, 220], [250, 230, 50], [180, 0, 0]],
        dtype=np.float32,
    )
    channels = [np.interp(value, stops, colors[:, index]) for index in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    hue = np.mod(hue, 1.0)
    sector = np.floor(hue * 6).astype(np.int32)
    fraction = hue * 6 - sector
    p = value * (1 - saturation)
    q = value * (1 - fraction * saturation)
    t = value * (1 - (1 - fraction) * saturation)
    choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    rgb = np.zeros((*hue.shape, 3), dtype=np.float32)
    for index, components in enumerate(choices):
        selected = sector % 6 == index
        for channel, component in enumerate(components):
            rgb[..., channel][selected] = component[selected]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _masked(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) * mask[..., None]).astype(np.uint8)


def _save_rgb(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)
    return output


def save_prediction_visualizations(
    dolp_path: str | Path,
    aop_path: str | Path,
    s0: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[Path, Path]:
    """Save prediction-only DoLP heatmap and periodic AoP hue map."""
    if (
        s0.ndim != 3
        or prediction.ndim != 3
        or s0.shape[0] != 3
        or prediction.shape[0] != 6
        or s0.shape[-2:] != prediction.shape[-2:]
    ):
        raise ValueError("expected S0 [3,H,W] and prediction [6,H,W]")
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or mask.shape != prediction.shape[-2:]:
        raise ValueError("expected mask [1,H,W] or [H,W]")
    for name, value in (("s0", s0), ("prediction", prediction), ("mask", mask)):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    mask_2d = _to_numpy(mask).clip(0, 1)
    dolp, aop = s12_dolp_aop(prediction[None], s0[None])
    dolp_map = _to_numpy(dolp[0].mean(0)).clip(0, 1)
    aop_map = 0.5 * torch.atan2(
        torch.sin(2 * aop[0]).mean(0),
        torch.cos(2 * aop[0]).mean(0),
    )
    aop_map = _to_numpy(aop_map)

    dolp_rgb = _masked(_heatmap(dolp_map), mask_2d)
    aop_hue = (aop_map + math.pi / 2) / math.pi
    aop_rgb = _hsv_to_rgb(aop_hue, mask_2d, mask_2d)
    return _save_rgb(dolp_path, dolp_rgb), _save_rgb(aop_path, aop_rgb)
