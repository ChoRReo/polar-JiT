from pathlib import Path

import yaml


def test_inference_and_evaluation_paths_are_connected():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/polar_jit_small.yaml").read_text())

    assert config["model"]["condition_channels"] == 3
    assert config["model"]["target_channels"] == 6
    assert config["model"]["refiner_hidden_channels"] > 0
    assert config["train"]["w_gradient_l1"] > 0
    assert config["train"]["w_dolp_l1"] > 0
    assert config["train"]["w_aop_l1"] > 0
    assert config["inference"]["split"] == config["evaluation"]["split"] == "test"
    assert config["inference"]["output_dir"] == config["evaluation"]["predictions"]
    assert config["evaluation"]["visualize"] is True
