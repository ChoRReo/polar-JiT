#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from polar_jit import (
    ConditionalFlowMatcher,
    PolarJiT,
    load_official_jit_h16,
)
from polar_jit.losses import generation_weights, masked_mean
from polar_jit.ratio_data import build_ratio_dataset
from polar_jit.ratio_losses import reconstruction_losses_ratio


def write_train_log(path: Path, run_id: str, payload: dict):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def require_finite_losses(log_path, run_id, step, losses):
    nonfinite = [name for name, value in losses.items() if not torch.isfinite(value).all()]
    if not nonfinite:
        return
    write_train_log(
        log_path,
        run_id,
        {
            "event": "nonfinite_loss",
            "step": step,
            "components": nonfinite,
            "values": {name: float(value.detach().float().cpu()) for name, value in losses.items()},
        },
    )
    raise FloatingPointError(f"non-finite training loss at step {step}: {nonfinite}")


def main(default_config="configs/polar_jit_h16_ratio.yaml", condition_key="s0"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    train_cfg, device = config["train"], torch.device(args.device)
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output = Path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    log_path = output / train_cfg.get("log_file", "train_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    dataset = build_ratio_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = PolarJiT(**config["model"]).to(device)
    pretrained_cfg = config.get("pretrained", {})
    pretrained_report = None
    if pretrained_cfg.get("enabled", False) and not args.resume:
        pretrained_report = load_official_jit_h16(
            model,
            pretrained_cfg["path"],
            pretrained_cfg.get("state_key", "model_ema1"),
        )
    flow = ConditionalFlowMatcher(model, **config["flow"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(train_cfg.get("weight_decay", 0)),
    )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"]); step = int(state["step"])

    amp_name = train_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("training DataLoader is empty; reduce train.batch_size")
    max_steps = int(train_cfg["max_steps"])
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 5000))
    if max_steps < 1 or log_every < 1 or save_every < 1:
        raise ValueError("max_steps, log_every and save_every must be positive")
    total_epochs = ceil(max_steps / steps_per_epoch)
    write_train_log(
        log_path,
        run_id,
        {
            "event": "run_start",
            "resume": args.resume or None,
            "step": step,
            "max_steps": max_steps,
            "epoch": step // steps_per_epoch + 1,
            "total_epochs": total_epochs,
            "steps_per_epoch": steps_per_epoch,
            "samples": len(dataset),
            "condition_key": condition_key,
            "condition_channels": int(config["model"]["condition_channels"]),
        },
    )
    if pretrained_report is not None:
        write_train_log(
            log_path,
            run_id,
            {
                "event": "pretrained",
                "pretrained": {
                    key: value
                    for key, value in pretrained_report.items()
                    if key != "loaded_from"
                },
            },
        )
    iterator = iter(loader)
    while step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        target = batch["s12"]
        if condition_key not in batch:
            raise KeyError(f"dataset sample does not provide condition key '{condition_key}'")
        condition = batch[condition_key]
        expected_channels = int(config["model"]["condition_channels"])
        if condition.shape[1] != expected_channels:
            raise ValueError(
                f"condition '{condition_key}' has {condition.shape[1]} channels, "
                f"model expects {expected_channels}"
            )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred, v_pred, v_target, _ = flow(target, condition)
            spatial_weights = generation_weights(batch["mask"])
            loss_flow = masked_mean((v_pred - v_target).square(), spatial_weights)
            clean_l1, gradient_l1, frequency_l1, dolp_l1, aop_l1 = (
                reconstruction_losses_ratio(
                    pred["clean"],
                    target,
                    batch["s0"],
                    spatial_weights,
                    patch_size=int(config["model"]["patch_size"]),
                    patch_boundary_weight=float(
                        train_cfg.get("gradient_patch_boundary_weight", 4.0)
                    ),
                    high_frequency_kernel_sizes=tuple(
                        train_cfg.get("high_frequency_kernel_sizes", (3, 7))
                    ),
                )
            )
            loss = (train_cfg["w_flow"] * loss_flow
                    + train_cfg["w_clean_l1"] * clean_l1
                    + train_cfg.get("w_gradient_l1", 0.1) * gradient_l1
                    + train_cfg.get("w_high_frequency_l1", 0.1) * frequency_l1
                    + train_cfg.get("w_dolp_l1", train_cfg.get("w_dolp", 0.1)) * dolp_l1
                    + train_cfg.get("w_aop_l1", train_cfg.get("w_aop", 0.1)) * aop_l1)
        require_finite_losses(
            log_path,
            run_id,
            step + 1,
            {
                "total": loss,
                "flow": loss_flow,
                "clean_l1": clean_l1,
                "gradient_l1": gradient_l1,
                "high_frequency_l1": frequency_l1,
                "dolp_l1": dolp_l1,
                "aop_l1": aop_l1,
            },
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg.get("grad_clip", 1)),
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            write_train_log(
                log_path,
                run_id,
                {
                    "event": "nonfinite_gradient",
                    "step": step + 1,
                    "error": str(error),
                },
            )
            raise
        optimizer.step(); step += 1
        decay = float(train_cfg.get("ema_decay", 0.9999))
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.mul_(decay).add_(p, alpha=1 - decay)
        epoch = (step - 1) // steps_per_epoch + 1
        batch_in_epoch = (step - 1) % steps_per_epoch + 1
        if step == 1 or step % log_every == 0:
            write_train_log(
                log_path,
                run_id,
                {
                    "event": "train",
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "batch": batch_in_epoch,
                    "batches_per_epoch": steps_per_epoch,
                    "step": step,
                    "max_steps": max_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm.item(),
                    "total": loss.item(),
                    "flow": loss_flow.item(),
                    "clean_l1": clean_l1.item(),
                    "gradient_l1": gradient_l1.item(),
                    "high_frequency_l1": frequency_l1.item(),
                    "dolp_l1": dolp_l1.item(),
                    "aop_l1": aop_l1.item(),
                },
            )
        if step % save_every == 0 or step == max_steps:
            checkpoint_path = output / f"checkpoint-{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": config,
                },
                checkpoint_path,
            )
            save_file(ema.state_dict(), str(output / "model_ema.safetensors"))
            write_train_log(
                log_path,
                run_id,
                {
                    "event": "checkpoint",
                    "epoch": epoch,
                    "step": step,
                    "path": str(checkpoint_path),
                },
            )


if __name__ == "__main__":
    main()
