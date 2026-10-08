from __future__ import annotations

import torch

from .data import UnifiedSfPDataset


def raw_s12_to_ratio(s0: torch.Tensor, s12: torch.Tensor, eps: float = 1e-6):
    """Convert network-space S0 plus raw [S1,S2] to [S1/S0,S2/S0]."""
    if s0.ndim != 3 or s0.shape[0] != 3:
        raise ValueError(f"expected S0 [3,H,W], got {tuple(s0.shape)}")
    if s12.ndim != 3 or s12.shape[0] != 6 or s12.shape[-2:] != s0.shape[-2:]:
        raise ValueError(f"expected matching S1,S2 [6,H,W], got {tuple(s12.shape)}")
    denominator = s0.float().add(1).clamp_min(eps)
    return torch.cat(
        (s12[:3].float() / denominator, s12[3:].float() / denominator), dim=0
    ).clamp(-1, 1)


class RatioUnifiedSfPDataset(UnifiedSfPDataset):
    """Opt-in dataset variant whose target is [S1/S0,S2/S0]."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ratio = raw_s12_to_ratio(sample["s0"], sample["s12"])
        sample["s12"] = ratio
        if self.oracle_mgt:
            # Rebuild m_gt from the ratio target so this experiment is fully
            # self-contained and does not inherit the raw-target condition.
            s1, s2 = ratio[:3].mean(0), ratio[3:].mean(0)
            amplitude = torch.sqrt(s1.square() + s2.square() + 1e-12)
            mgt = (
                (s1 / amplitude) * sample["cos2phi"][0]
                - (s2 / amplitude) * sample["sin2phi"][0]
            ).clamp(-1, 1)[None] * sample["mask"]
            sample["mgt"] = mgt
            sample["condition"] = torch.cat(
                (
                    sample["s0"],
                    sample["theta"],
                    sample["cos2phi"],
                    sample["sin2phi"],
                    mgt,
                ),
                dim=0,
            )
        return sample


def build_ratio_dataset(config: dict, split=None):
    data = config["data"]
    if data.get("format", "unified_sfp") != "unified_sfp":
        raise ValueError("only data.format=unified_sfp is currently supported")
    chosen_split = split or data.get("split", "train")
    return RatioUnifiedSfPDataset(
        data["root"],
        chosen_split,
        data.get("image_size", 256),
        data.get("sources"),
        augment=chosen_split == "train" and data.get("augment", True),
        oracle_mgt=data.get("oracle_mgt_condition", False),
    )
