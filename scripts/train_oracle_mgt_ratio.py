#!/usr/bin/env python3
"""Train the isolated ratio-target oracle m_gt experiment."""

from train_ratio import main


if __name__ == "__main__":
    main(
        default_config="configs/polar_jit_h16_oracle_mgt_ratio.yaml",
        condition_key="condition",
    )
