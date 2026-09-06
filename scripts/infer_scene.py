#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from polar_jit import (
    ConditionalFlowMatcher,
    PolarJiT,
    load_stokes_scene,
    save_scene_bundle,
)


def main():
    parser = argparse.ArgumentParser(
        description="Infer S1/S2 for one scene from four analyzer images."
    )
    parser.add_argument("--config", default="configs/polar_jit_small.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pol-000", required=True)
    parser.add_argument("--pol-045", required=True)
    parser.add_argument("--pol-090", required=True)
    parser.add_argument("--pol-135", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--polarization-bits", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", choices=("euler", "heun"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    infer_cfg = config.get("inference", {})
    scene_cfg = config.get("single_scene", {})
    output = Path(args.output_dir or scene_cfg.get("output_dir", "outputs/single_scene"))
    bits = (
        args.polarization_bits
        if args.polarization_bits is not None
        else int(scene_cfg.get("polarization_bits", 8))
    )
    steps = args.steps if args.steps is not None else int(infer_cfg.get("steps", 20))
    method = args.method or infer_cfg.get("method", "heun")
    seed = args.seed if args.seed is not None else int(infer_cfg.get("seed", 42))
    device = torch.device(args.device or infer_cfg.get("device", "cuda"))
    if bits < 1:
        parser.error("--polarization-bits must be positive")
    if steps < 1:
        parser.error("--steps must be positive")
    if method not in {"euler", "heun"}:
        parser.error("inference method must be euler or heun")

    image_size = int(config["model"]["image_size"])
    s0, target, mask = load_stokes_scene(
        args.pol_000,
        args.pol_045,
        args.pol_090,
        args.pol_135,
        polarization_bits=bits,
        image_size=image_size,
        mask=args.mask,
    )

    model = PolarJiT(**config["model"]).to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    if checkpoint.suffix == ".safetensors":
        model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    else:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state.get("ema", state.get("model", state)))
    model.eval()
    flow = ConditionalFlowMatcher(model, **config["flow"])
    prediction = flow.sample(s0[None].to(device), steps, method, seed)[0].cpu()

    name = args.name or Path(args.pol_000).stem
    paths = save_scene_bundle(
        output,
        name=name,
        prediction=prediction,
        target=target,
        s0=s0,
        mask=mask,
        metadata={
            "polarization_bits": bits,
            "image_size": image_size,
            "steps": steps,
            "method": method,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "inputs": {
                "pol_000": args.pol_000,
                "pol_045": args.pol_045,
                "pol_090": args.pol_090,
                "pol_135": args.pol_135,
                "mask": args.mask,
            },
        },
    )
    print(
        json.dumps(
            {
                "scene_dir": str(output),
                "prediction_npy": str(paths["prediction"]),
                "target_npy": str(paths["target"]),
                "evaluate": (
                    "PYTHONPATH=src python3 scripts/evaluate.py "
                    f"--config {args.config} --scene-dir {output}"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
