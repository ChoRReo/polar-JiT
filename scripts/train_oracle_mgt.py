#!/usr/bin/env python3
"""Train the oracle normal/polarization-conditioned JiT experiment."""

from train import main


if __name__ == "__main__":
    main(
        default_config="configs/polar_jit_h16_oracle_mgt.yaml",
        condition_key="condition",
    )
