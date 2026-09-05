#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from polar_jit import ConditionalFlowMatcher, PolarJiT, build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/polar_jit_small.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None, choices=("train", "test"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", choices=("euler", "heun"), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    infer_cfg = config.get("inference", {})
    output = Path(args.output_dir or infer_cfg.get("output_dir", "outputs/polar_jit"))
    split = args.split or infer_cfg.get("split", "test")
    steps = args.steps if args.steps is not None else int(infer_cfg.get("steps", 20))
    method = args.method or infer_cfg.get("method", "heun")
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else int(infer_cfg.get("max_samples", 0))
    )
    seed = args.seed if args.seed is not None else int(infer_cfg.get("seed", 42))
    device = torch.device(args.device or infer_cfg.get("device", "cuda"))
    if split not in {"train", "test"}:
        parser.error("inference split must be train or test")
    if method not in {"euler", "heun"}:
        parser.error("inference method must be euler or heun")
    if steps < 1:
        parser.error("--steps must be positive")
    if max_samples < 0:
        parser.error("--max-samples cannot be negative")
    model = PolarJiT(**config["model"]).to(device)
    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix == ".safetensors":
        model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    else:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("ema", state.get("model", state)))
    model.eval()
    flow = ConditionalFlowMatcher(model, **config["flow"])
    dataset = build_dataset(config, split=split)
    count = len(dataset) if max_samples == 0 else min(len(dataset), max_samples)
    for index in range(count):
        sample = dataset[index]
        s0 = sample["s0"][None].to(device)
        s12 = flow.sample(s0, steps, method, seed + index)[0].cpu().numpy()
        path = output / Path(sample["name"]).with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, s12.astype(np.float32))
        print(f"[{index + 1}/{count}] {path}")
    print(
        json.dumps(
            {
                "prediction_dir": str(output),
                "samples": count,
                "split": split,
                "steps": steps,
                "method": method,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
