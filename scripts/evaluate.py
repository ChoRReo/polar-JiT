#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from polar_jit import build_dataset, evaluate_stokes_prediction
from polar_jit.visualization import save_prediction_visualizations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/polar_jit_small.yaml")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--split", default=None, choices=("train", "test"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--fail-on-missing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--visualization-dir", default=None)
    parser.add_argument("--max-visualizations", type=int, default=None)
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = config.get("evaluation", {})
    predictions = Path(args.predictions or eval_cfg.get("predictions", "outputs/polar_jit"))
    output = Path(args.output_csv or eval_cfg.get("output_csv", "outputs/metrics.csv"))
    dataset = build_dataset(config, args.split or eval_cfg.get("split", "test"))
    window_size = int(eval_cfg.get("ssim_window_size", 11))
    sigma = float(eval_cfg.get("ssim_sigma", 1.5))
    threshold = float(eval_cfg.get("mask_threshold", 0.5))
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else int(eval_cfg.get("max_samples", 0))
    )
    fail_on_missing = (
        args.fail_on_missing
        if args.fail_on_missing is not None
        else bool(eval_cfg.get("fail_on_missing", False))
    )
    visualize = bool(eval_cfg.get("visualize", True)) and not args.no_visualize
    visualization_dir = Path(
        args.visualization_dir
        or eval_cfg.get("visualization_dir", "outputs/visualizations/polar_jit")
    )
    max_visualizations = (
        args.max_visualizations
        if args.max_visualizations is not None
        else int(eval_cfg.get("max_visualizations", 0))
    )
    if not 0 <= threshold <= 1:
        parser.error("mask threshold must be in [0, 1]")
    if window_size < 1 or window_size % 2 == 0:
        parser.error("SSIM window size must be a positive odd number")
    if sigma <= 0:
        parser.error("SSIM sigma must be positive")
    if max_samples < 0 or max_visualizations < 0:
        parser.error("sample limits cannot be negative")
    rows = []
    visualized = 0
    count = len(dataset) if max_samples == 0 else min(len(dataset), max_samples)
    entries = []
    missing = []
    for index in range(count):
        name = dataset.sample_name(index)
        path = predictions / Path(name).with_suffix(".npy")
        (entries if path.is_file() else missing).append((index, path))
    if missing and fail_on_missing:
        preview = ", ".join(str(path) for _, path in missing[:3])
        raise FileNotFoundError(f"missing {len(missing)} prediction files, including: {preview}")
    if missing:
        warnings.warn(f"skipped {len(missing)} samples without prediction files", stacklevel=1)

    for index, path in entries:
        sample = dataset[index]
        pred = torch.from_numpy(np.load(path, allow_pickle=False)).float()
        if pred.shape != (6, *sample["s0"].shape[-2:]):
            raise ValueError(f"invalid S1/S2 shape in {path}: {tuple(pred.shape)}")
        pred = pred[None]
        target = sample["s12"][None]
        mask = (sample["mask"][None] >= threshold).float()
        metrics = evaluate_stokes_prediction(
            pred,
            target,
            sample["s0"][None],
            mask,
            window_size=window_size,
            sigma=sigma,
        )
        rows.append({"name": sample["name"], **metrics})
        if visualize and (max_visualizations <= 0 or visualized < max_visualizations):
            relative_path = Path(sample["name"]).with_suffix(".png")
            save_prediction_visualizations(
                visualization_dir / "dolp" / relative_path,
                visualization_dir / "aop" / relative_path,
                sample["s0"],
                pred[0],
                mask[0],
            )
            visualized += 1
    if not rows:
        raise RuntimeError("no matching prediction files")
    mean_row = {"name": "mean"}
    for key in rows[0]:
        if key != "name":
            mean_row[key] = float(np.mean([row[key] for row in rows]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
    print(
        json.dumps(
            {
                "metrics_csv": str(output),
                "requested_samples": count,
                "evaluated_samples": len(rows),
                "missing_predictions": len(missing),
                "visualization_dir": str(visualization_dir) if visualize else None,
                "visualized_samples": visualized,
                "visualization_files": visualized * 2,
                "mean": {key: value for key, value in mean_row.items() if key != "name"},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
