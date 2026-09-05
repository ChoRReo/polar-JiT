import torch
from PIL import Image

from polar_jit.visualization import save_prediction_visualizations


def test_save_prediction_dolp_and_aop_visualizations(tmp_path):
    height, width = 8, 10
    s0 = torch.zeros(3, height, width)
    prediction = torch.zeros(6, height, width)
    prediction[3:] = 0.5
    mask = torch.ones(1, height, width)
    mask[:, :, :2] = 0
    dolp_output = tmp_path / "dolp" / "sample.png"
    aop_output = tmp_path / "aop" / "sample.png"

    outputs = save_prediction_visualizations(
        dolp_output, aop_output, s0, prediction, mask
    )

    assert outputs == (dolp_output, aop_output)
    for output in outputs:
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (width, height)
            assert image.getpixel((0, 0)) == (0, 0, 0)
