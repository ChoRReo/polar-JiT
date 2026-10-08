#!/usr/bin/env python3
"""Run checkpoints trained from the uncommitted H/16 edit based on 0e9be44."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from polar_jit import ConditionalFlowMatcher, build_dataset
from polar_jit.legacy_h16 import (
    LegacyPolarJiTH16,
    describe_legacy_state,
    strip_checkpoint_prefix,
)


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    if path.suffix == ".safetensors":
        return load_file(str(path), device=str(device)), None
    # Keep the historical Adam optimizer tensors off GPU. H/16 training .pt
    # files can otherwise exhaust VRAM before the model state is even loaded.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    return state, checkpoint.get("config")


def main():
    parser = argparse.ArgumentParser(
        description="Infer with the legacy 0e9be44-based JiT-H/16 checkpoint."
    )
    parser.add_argument(
        "--config", default="configs/polar_jit_h16_legacy_0e9.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None, choices=("train", "test"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", choices=("euler", "heun"), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    infer_cfg = config.get("inference", {})
    output = Path(args.output_dir or infer_cfg["output_dir"])
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
    if steps < 1:
        parser.error("--steps must be positive")
    if max_samples < 0:
        parser.error("--max-samples cannot be negative")

    checkpoint_path = Path(args.checkpoint)
    raw_state, saved_config = load_checkpoint(checkpoint_path, device)
    state = strip_checkpoint_prefix(raw_state)
    observed = describe_legacy_state(state)
    model_config = dict(config["model"])
    if isinstance(saved_config, dict) and isinstance(saved_config.get("model"), dict):
        # A .pt training checkpoint is the strongest record of the manually
        # edited H/16 dimensions. Keep only fields understood by the old model.
        allowed = set(model_config)
        model_config.update(
            {key: value for key, value in saved_config["model"].items() if key in allowed}
        )
    model = LegacyPolarJiTH16(**model_config).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        expected = {
            key: model_config.get(key)
            for key in (
                "patch_size",
                "hidden_size",
                "depth",
                "num_heads",
                "bottleneck_dim",
                "refiner_hidden_channels",
            )
        }
        raise RuntimeError(
            "legacy checkpoint does not match the configured 0e9 H/16 model; "
            f"configured={expected}, observed={observed}. "
            "If this is a safetensors file from a different manual H/16 edit, "
            "copy its original model dimensions into the legacy YAML."
        ) from error

    model.eval()
    flow = ConditionalFlowMatcher(model, **config["flow"])
    dataset = build_dataset(config, split=split)
    count = len(dataset) if max_samples == 0 else min(len(dataset), max_samples)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        sample = dataset[index]
        condition = sample["s0"][None].to(device)
        prediction = flow.sample(condition, steps, method, seed + index)[0]
        path = output / Path(sample["name"]).with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, prediction.float().cpu().numpy().astype(np.float32))
        print(f"[{index + 1}/{count}] {path}", flush=True)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_architecture": observed,
                "prediction_dir": str(output),
                "samples": count,
                "split": split,
                "steps": steps,
                "method": method,
                "legacy_base_commit": "0e9be44b6169a7d48da052c9ca6a591813623b7b",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
