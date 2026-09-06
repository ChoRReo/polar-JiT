#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from array import array
from pathlib import Path

import yaml
from PIL import Image


def resize_float_channel(channel: Image.Image, image_size: int) -> list[float]:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = channel.convert("F").resize((image_size, image_size), resampling)
    return list(resized.getdata())


def load_analyzer(path: Path, bits: int, image_size: int) -> list[list[float]]:
    scale = float((1 << bits) - 1)
    with Image.open(path) as image:
        if image.mode in {"1", "L", "I", "F", "I;16", "I;16B", "I;16L"}:
            channel = [value / scale for value in resize_float_channel(image, image_size)]
            return [channel, channel, channel]
        channels = image.convert("RGB").split()
        return [
            [value / scale for value in resize_float_channel(channel, image_size)]
            for channel in channels
        ]


def load_mask(path: Path, image_size: int) -> list[float]:
    resampling = getattr(Image, "Resampling", Image).NEAREST
    with Image.open(path) as image:
        resized = image.convert("L").resize((image_size, image_size), resampling)
        return [1.0 if value > 0 else 0.0 for value in resized.getdata()]


def stokes_from_analyzers(images):
    i0, i45, i90, i135 = images
    s0_channels, s12_channels = [], []
    for channel in range(3):
        values = zip(i0[channel], i45[channel], i90[channel], i135[channel])
        s0, s1, s2 = [], [], []
        for value_0, value_45, value_90, value_135 in values:
            s0.append(min(2.0, max(0.0, ((value_0 + value_90) + (value_45 + value_135)) * 0.5)))
            s1.append(min(1.0, max(-1.0, value_0 - value_90)))
            s2.append(min(1.0, max(-1.0, value_45 - value_135)))
        s0_channels.append(s0)
        s12_channels.append(s1)
        s12_channels.append(s2)
    # The project layout is S1_RGB followed by S2_RGB, rather than interleaved.
    s12_channels = [
        s12_channels[0],
        s12_channels[2],
        s12_channels[4],
        s12_channels[1],
        s12_channels[3],
        s12_channels[5],
    ]
    return s0_channels, s12_channels


def write_float32_npy(path: Path, channels, image_size: int):
    shape = (len(channels), image_size, image_size)
    header = str({"descr": "<f4", "fortran_order": False, "shape": shape})
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    values = array("f")
    for channel in channels:
        values.extend(channel)
    if sys.byteorder != "little":
        values.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(struct.pack("<BBH", 1, 0, len(header_bytes)))
        handle.write(header_bytes)
        values.tofile(handle)


def heatmap(value: float) -> tuple[int, int, int]:
    stops = (0.0, 0.35, 0.7, 1.0)
    colors = ((20, 20, 80), (0, 180, 220), (250, 230, 50), (180, 0, 0))
    value = min(1.0, max(0.0, value))
    index = next((i for i in range(3) if value <= stops[i + 1]), 2)
    ratio = (value - stops[index]) / (stops[index + 1] - stops[index])
    return tuple(
        round(colors[index][channel] * (1 - ratio) + colors[index + 1][channel] * ratio)
        for channel in range(3)
    )


def hsv_to_rgb(hue: float) -> tuple[int, int, int]:
    hue = hue % 1.0
    sector = int(hue * 6)
    fraction = hue * 6 - sector
    choices = (
        (1.0, fraction, 0.0),
        (1.0 - fraction, 1.0, 0.0),
        (0.0, 1.0, fraction),
        (0.0, 1.0 - fraction, 1.0),
        (fraction, 0.0, 1.0),
        (1.0, 0.0, 1.0 - fraction),
    )
    return tuple(round(component * 255) for component in choices[sector % 6])


def save_dolp_aop(paths, s0_channels, s12_channels, mask, image_size):
    pixels = image_size * image_size
    dolp_pixels, aop_pixels = [], []
    for index in range(pixels):
        if mask[index] <= 0:
            dolp_pixels.append((0, 0, 0))
            aop_pixels.append((0, 0, 0))
            continue
        dolp_values, angles = [], []
        for channel in range(3):
            s1 = s12_channels[channel][index]
            s2 = s12_channels[channel + 3][index]
            amplitude = math.sqrt(s1 * s1 + s2 * s2 + 1e-12)
            dolp_values.append(min(1.0, amplitude / max(s0_channels[channel][index], 1e-6)))
            angles.append(0.5 * math.atan2(s2, s1))
        dolp_pixels.append(heatmap(sum(dolp_values) / 3))
        sin_mean = sum(math.sin(2 * angle) for angle in angles) / 3
        cos_mean = sum(math.cos(2 * angle) for angle in angles) / 3
        aop = 0.5 * math.atan2(sin_mean, cos_mean)
        aop_pixels.append(hsv_to_rgb((aop + math.pi / 2) / math.pi))
    for path, pixels_data in zip(paths, (dolp_pixels, aop_pixels)):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (image_size, image_size))
        image.putdata(pixels_data)
        image.save(path)


def select_rows(config):
    data = config["data"]
    root = Path(data["root"])
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected_sources = set(data.get("sources") or ())
    return root, [
        row
        for row in rows
        if (not selected_sources or row.get("source") in selected_sources)
        and row.get("source") == "deepsfp"
        and row.get("subset") in {"test+test_supp", "test_supp"}
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Export test-set S1/S2 GT and DoLP/AoP visualizations."
    )
    parser.add_argument("--config", default="configs/polar_jit_small.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples cannot be negative")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir or config.get("evaluation", {}).get("gt_dir", "test_gt"))
    image_size = int(config["data"].get("image_size", 256))
    root, rows = select_rows(config)
    if args.max_samples:
        rows = rows[: args.max_samples]
    exported = []
    for index, row in enumerate(rows):
        images = [
            load_analyzer(root / row[key], int(row["polarization_bits"]), image_size)
            for key in ("pol_000", "pol_045", "pol_090", "pol_135")
        ]
        s0_physical, s12 = stokes_from_analyzers(images)
        relative = Path(row["sample_id"])
        s12_path = output / "s12" / relative.with_suffix(".npy")
        dolp_path = output / "dolp" / relative.with_suffix(".png")
        aop_path = output / "aop" / relative.with_suffix(".png")
        write_float32_npy(s12_path, s12, image_size)
        if not args.no_visualize:
            mask = load_mask(root / row["mask"], image_size)
            save_dolp_aop((dolp_path, aop_path), s0_physical, s12, mask, image_size)
        exported.append(
            {
                "name": row["sample_id"],
                "source": row["source"],
                "s12": str(s12_path),
                "dolp": str(dolp_path) if not args.no_visualize else "",
                "aop": str(aop_path) if not args.no_visualize else "",
            }
        )
        print(f"[{index + 1}/{len(rows)}] {row['sample_id']}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "source", "s12", "dolp", "aop"))
        writer.writeheader()
        writer.writerows(exported)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "samples": len(exported),
                "s12_npy": len(exported),
                "visualization_files": 0 if args.no_visualize else len(exported) * 2,
                "manifest": str(manifest),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
