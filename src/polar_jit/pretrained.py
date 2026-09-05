from __future__ import annotations

from pathlib import Path

import torch


def load_official_jit_b16(model, checkpoint_path, state_key="model_ema1"):
    """Load shape-compatible official JiT-B/16 weights into PolarJiT.

    The official RGB input embedder is also copied into the S0 embedder. The
    task-specific six-channel flow input and output projections remain newly
    initialized when their shapes do not match.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing pretrained checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if state_key not in checkpoint:
        raise KeyError(f"missing state '{state_key}' in {path}; found {list(checkpoint)}")
    source = checkpoint[state_key]
    target = model.state_dict()
    loadable = {}
    loaded_from = {}

    for source_name, tensor in source.items():
        name = source_name.removeprefix("module.").removeprefix("net.")
        candidates = [name]
        if name.startswith("x_embedder."):
            candidates.append("s0_embedder." + name.removeprefix("x_embedder."))
        for candidate in candidates:
            if candidate in target and target[candidate].shape == tensor.shape:
                loadable[candidate] = tensor
                loaded_from[candidate] = source_name

    result = model.load_state_dict(loadable, strict=False)
    loaded_numel = sum(target[name].numel() for name in loadable)
    total_numel = sum(value.numel() for value in target.values())
    return {
        "path": str(path),
        "state_key": state_key,
        "loaded_tensors": len(loadable),
        "loaded_numel": loaded_numel,
        "total_numel": total_numel,
        "coverage": loaded_numel / max(total_numel, 1),
        "missing_tensors": list(result.missing_keys),
        "unexpected_tensors": list(result.unexpected_keys),
        "loaded_from": loaded_from,
    }
