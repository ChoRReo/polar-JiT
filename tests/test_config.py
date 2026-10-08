from pathlib import Path

import yaml


def test_inference_and_evaluation_paths_are_connected():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/polar_jit_h16.yaml").read_text())

    assert config["model"]["condition_channels"] == 3
    assert config["model"]["target_channels"] == 6
    assert config["model"]["patch_size"] == 16
    assert config["model"]["hidden_size"] == 1280
    assert config["model"]["depth"] == 32
    assert config["model"]["num_heads"] == 16
    assert config["model"]["bottleneck_dim"] == 256
    assert config["model"]["in_context_len"] == 32
    assert config["model"]["in_context_start"] == 10
    assert config["model"]["refiner_hidden_channels"] > 0
    assert config["train"]["w_gradient_l1"] > 0
    assert config["train"]["w_high_frequency_l1"] > 0
    assert config["train"]["w_dolp_l1"] > 0
    assert config["train"]["w_aop_l1"] > 0
    assert config["inference"]["split"] == config["evaluation"]["split"] == "test"
    assert config["inference"]["output_dir"] == config["evaluation"]["predictions"]
    assert config["evaluation"]["visualize"] is True


def test_oracle_mgt_experiment_uses_seven_channel_condition():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/polar_jit_h16_oracle_mgt.yaml").read_text())

    assert config["model"]["condition_channels"] == 7
    assert config["data"]["oracle_mgt_condition"] is True
    assert config["inference"]["condition_key"] == "condition"
    assert config["inference"]["output_dir"] == config["evaluation"]["predictions"]


def test_ratio_experiments_use_separate_paths():
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs/polar_jit_h16.yaml").read_text())
    ratio = yaml.safe_load((root / "configs/polar_jit_h16_ratio.yaml").read_text())
    oracle_ratio = yaml.safe_load(
        (root / "configs/polar_jit_h16_oracle_mgt_ratio.yaml").read_text()
    )

    assert raw["train"]["output_dir"] != ratio["train"]["output_dir"]
    assert raw["evaluation"]["gt_dir"] == "test_gt"
    assert ratio["evaluation"]["gt_dir"] == "test_gt_ratio"
    assert oracle_ratio["model"]["condition_channels"] == 7
    assert oracle_ratio["inference"]["condition_key"] == "condition"
