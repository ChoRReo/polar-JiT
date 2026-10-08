# Polar JiT Flow

一个独立的、像素空间的偏振恢复项目。默认模型以 RGB `S0` 为条件，直接生成
RGB Stokes 分量 `[S1, S2]`，不依赖 Stable Diffusion、VAE 或 ControlNet。
`S1/S0,S2/S0` 作为独立 `_ratio` 实验保留，不会覆盖默认数据或脚本。

## 架构

```text
S0 ──► H/16 condition patch embed ──────────┐
       ├─ spatial token addition             ▼
       └─ global pool ──► AdaLN condition ─► 32-block JiT-H/16 ─► unpatchify
time ──► timestep embed ────────────────┘             ▲                    │
noise ──► S1,S2 flow ─► H/16 patch embed ─────────────────────────────────┘
                  full-resolution S0 ─────► residual 3x3 refiner ─► clean S1,S2
```

- 主干按官方 JiT-H/16 设置为 patch 16、hidden 1280、32 blocks、16 heads 和
  bottleneck 256，并采用 RMSNorm、QK-Norm、2D RoPE、SwiGLU 与 AdaLN-Zero。
- 默认训练路径为 `x_t = t*[S1,S2] + (1-t)*noise`；网络预测 clean Stokes，并换算为速度场损失。
- S0 patch token 逐位置注入生成 token，并在全局池化后用于调制所有 JiT block。
- unpatchify 后的零初始化残差卷积头跨 patch 融合相邻像素，并直接读取全分辨率
  S0 以恢复 patch embedding 中损失的局部纹理；初始化时是严格恒等映射，不改变
  JiT 的零输出初始化。
- 推理只读取 S0，通过 Euler 或 Heun ODE 积分生成 `S1,S2`。

该实现根据 [LTH14/JiT](https://github.com/LTH14/JiT) 的公开 MIT 实现重新组织。
保留官方 H/16 主干形式，但将 ImageNet 类别条件替换为当前的 S0 空间条件与
全局 AdaLN 条件；官方在第 10 个 block 注入的 32 个 class in-context tokens
相应替换为 S0 全局条件 tokens，不引入类别 embedding 或 CFG。

## 数据

当前支持原项目 `UnifiedSfP_png/manifest.csv` 约定。必需字段为：

```text
sample_id, source, subset, polarization_bits,
pol_000, pol_045, pol_090, pol_135, mask
```

数据读取后统一转换为 RGB Stokes：

```text
S0 = ((I0 + I90) + (I45 + I135)) / 2
S1 = I0 - I90
S2 = I45 - I135
```

分析器图像归一化到 `[0,1]` 后，物理 `S0` 位于 `[0,2]`。送入网络前对 S0
执行 `S0_net=S0-1`；默认生成目标是裁剪到 `[-1,1]` 的
`[S1_RGB,S2_RGB] [6,H,W]`。DoLP 为 `sqrt(S1^2+S2^2)/S0`，AoP 为
`0.5*atan2(S2,S1)`。最终指标在前景像素和
RGB 通道上共同取平均。

默认配置已经指向：

```text
/home/xserver/pjt/datasets/UnifiedSfP_png
```

训练参数均由 `configs/polar_jit_h16.yaml` 管理。所有训练损失直接使用 object
mask：mask 内权重为 1，mask 外权重为 0，背景不参与优化。

## 安装

```bash
python3 -m pip install -e .
```

## 训练

```bash
PYTHONPATH=src python3 scripts/train.py --config configs/polar_jit_h16.yaml --device cuda
```

默认配置设置为 `pretrained.enabled: false`，模型全部参数随机初始化并从头训练，
不会读取本地 JiT-H/16 checkpoint。这里保留 JiT-H/16 的模型结构，但不使用其
预训练参数。

训练会保存可恢复的 `.pt` checkpoint 和只包含 EMA 模型的
`model_ema.safetensors`。当前入口是单 GPU 版本，结构本身兼容后续 DDP 封装。
监控信息会同时打印到终端并追加保存至输出目录下的 `train_log.jsonl`。每条训练
记录包含当前/总 epoch、epoch 内 batch、当前/总 step、学习率及各项 loss；断点
恢复后 epoch 会根据已完成 step 连续计算。日志文件名可通过 `train.log_file` 修改。
训练损失由 flow MSE、Stokes 分量 clean L1、空间梯度 L1、多尺度高频 L1、
DoLP L1 和 AoP L1 组成。梯度损失对跨越 16×16 patch 边界的误差额外加权，
以直接抑制块状接缝；3×3 和 7×7 高通残差监督用于恢复细节与中尺度纹理。
DoLP/AoP 均由预测与 GT 的 `S1,S2` 动态计算。AoP L1 使用周期为 π 的最短角距离，
并按 GT DoLP 加权；在 `S1=S2=0` 的无偏振位置停止未定义的角度梯度，从而避免
`atan2(0,0)` 导致 NaN。各项权重及 patch 边界倍率均由 YAML 的 `train` 段控制。
若仍出现非有限 loss 或梯度，训练会立即停止并将具体错误项写入日志，防止继续
保存已污染的权重。

## 推理

```bash
PYTHONPATH=src python3 scripts/infer.py \
  --config configs/polar_jit_h16.yaml \
  --checkpoint checkpoints/polar_jit_h16_stokes/model_ema.safetensors
```

预测文件为 `[6,H,W]` 的 float32 NPY，通道顺序是
`[S1_R,S1_G,S1_B,S2_R,S2_G,S2_B]`。
DoLP 和 AoP 不作为生成通道，而是在损失与评估阶段由 S0 与 Stokes 分量动态计算。

推理的 `split`、输出目录、采样步数、Euler/Heun 方法、最大样本数、随机种子和
设备默认从 YAML 的 `inference` 段读取，也可用同名命令行参数临时覆盖。例如：

```bash
python3 scripts/infer.py \
  --config configs/polar_jit_h16.yaml \
  --checkpoint checkpoints/polar_jit_h16_stokes/model_ema.safetensors \
  --steps 40 --method heun --max-samples 100
```

### 单场景四方向推理

对于不在 manifest 中的单个场景，可以直接提供四张偏振方向图像：

```bash
PYTHONPATH=src python3 scripts/infer_scene.py \
  --config configs/polar_jit_h16.yaml \
  --checkpoint checkpoints/polar_jit_h16_stokes/model_ema.safetensors \
  --pol-000 /path/to/I0.png \
  --pol-045 /path/to/I45.png \
  --pol-090 /path/to/I90.png \
  --pol-135 /path/to/I135.png \
  --mask /path/to/mask.png \
  --polarization-bits 8 \
  --name my_scene \
  --output-dir outputs/single_scene/my_scene
```

四张图会按训练数据的相同规则缩放到 `model.image_size × model.image_size`，再
转换为 S0、S1、S2。`--mask` 可省略，此时整幅图均视为前景。输出目录包含：

```text
prediction_s12.npy  # 模型预测的 S1,S2，[6,H,W]
target_s12.npy      # 四方向图计算的 S1,S2 GT，[6,H,W]
s0.npy              # 网络空间 S0，[3,H,W]
mask.npy            # 评估 mask，[1,H,W]
scene.json           # 场景和采样参数
```

随后直接用同一个评估脚本输出单场景指标和预测 DoLP/AoP 图：

```bash
PYTHONPATH=src python3 scripts/evaluate.py \
  --config configs/polar_jit_h16.yaml \
  --scene-dir outputs/single_scene/my_scene
```

默认生成 `metrics.csv`、`visualizations/dolp/my_scene.png` 和
`visualizations/aop/my_scene.png`。也可以通过 `--output-csv`、
`--visualization-dir` 或 `--no-visualize` 覆盖。

## 评估

导出当前测试集的数值 GT 和 DoLP/AoP 可视化到仓库根目录：

```bash
PYTHONPATH=src python3 scripts/export_test_gt.py \
  --config configs/polar_jit_h16.yaml
```

默认输出到 `test_gt/`：`s12/` 中是 `[S1_RGB,S2_RGB]` float32 NPY，`dolp/`
和 `aop/` 中是仅显示 mask 内目标的 PNG，`manifest.csv` 记录所有对应路径。
输出目录可通过 `evaluation.gt_dir` 或 `--output-dir` 修改。

```bash
PYTHONPATH=src python3 scripts/evaluate.py \
  --config configs/polar_jit_h16.yaml
```

当前评估只统计 object mask 内的指标，输出：

- DoLP：MAE、PSNR、SSIM；
- AoP：周期安全的 MAE（度）、PSNR、SSIM。

当前数据清单的 `test` 会选取 DeepSfP 的 `test+test_supp` 与 `test_supp`
两个子集，共 65 个样本，不会把训练集或 `unlisted` 样本混入评估。

AoP 的 PSNR 使用相对于最大周期误差 90° 的归一化误差；AoP SSIM 在
`[cos(2AoP), sin(2AoP)]` 周期表示上计算，避免 0°/180° 边界产生伪误差。
SSIM 的窗口大小、标准差和 mask 阈值可通过配置中的 `evaluation` 修改。

每个预测只保存两张与原图同尺寸的可视化结果：`dolp/<sample>.png` 是预测
DoLP 热力图，`aop/<sample>.png` 是预测 AoP 周期色相图。不再输出 S0、GT、
误差图或拼接图，mask 外统一置黑。CSV、预测目录和可视化目录默认均由 YAML
的 `evaluation` 段指定；`max_visualizations: 0` 表示可视化全部已评估样本。

需要临时覆盖配置时可使用：

```bash
python3 scripts/evaluate.py \
  --config configs/polar_jit_h16.yaml \
  --predictions outputs/polar_jit_h16_stokes \
  --output-csv outputs/metrics/experiment.csv \
  --visualization-dir outputs/visualizations/experiment \
  --max-visualizations 50 --fail-on-missing
```

使用 `--no-visualize` 可只计算 CSV 指标。评估结束后，终端还会输出 JSON 汇总，
包括样本数、缺失预测数、可视化数和六项平均指标。

## 独立 S1/S0、S2/S0 实验

归一化目标使用独立的数据类、损失、评估、可视化、配置和入口；默认脚本仍保持
原始 `S1,S2` 语义。两套实验的 checkpoint、预测、GT、指标和可视化目录也彼此隔离。

```bash
# 普通 S0 条件
PYTHONPATH=src python3 scripts/train_ratio.py \
  --config configs/polar_jit_h16_ratio.yaml --device cuda

PYTHONPATH=src python3 scripts/infer_ratio.py \
  --config configs/polar_jit_h16_ratio.yaml \
  --checkpoint checkpoints/polar_jit_h16_stokes_ratio/model_ema.safetensors

PYTHONPATH=src python3 scripts/export_test_gt_ratio.py \
  --config configs/polar_jit_h16_ratio.yaml

PYTHONPATH=src python3 scripts/evaluate_ratio.py \
  --config configs/polar_jit_h16_ratio.yaml
```

单场景入口为 `scripts/infer_scene_ratio.py`。Oracle ratio 实验使用：

```bash
PYTHONPATH=src python3 scripts/train_oracle_mgt_ratio.py \
  --config configs/polar_jit_h16_oracle_mgt_ratio.yaml --device cuda
```

ratio 数据目标为 `[S1_RGB/S0_RGB,S2_RGB/S0_RGB]`；其 DoLP 直接由 ratio
幅值计算，不会再次除以 S0。

## 旧 0e9 H/16 checkpoint 测试

曾在提交 `0e9be44b6169a7d48da052c9ca6a591813623b7b` 基础上手工放大到 H/16
训练的 checkpoint，不能直接载入当前 `PolarJiT`：旧结构没有 32 个 in-context
tokens，refiner 也只有 `Conv-SiLU-Conv`，不含当前的全分辨率 S0 条件卷积。

专用兼容入口保持旧参数名称和前向结构，同时将模型尺寸设为 H/16：

```bash
PYTHONPATH=src python3 scripts/infer_legacy_h16.py \
  --config configs/polar_jit_h16_legacy_0e9.yaml \
  --checkpoint /path/to/checkpoint-STEP.pt

PYTHONPATH=src python3 scripts/evaluate.py \
  --config configs/polar_jit_h16_legacy_0e9.yaml
```

也支持旧的 `model_ema.safetensors`。对于 `.pt` 文件，推理脚本优先读取 checkpoint
内保存的 `config.model` 尺寸；对于不含配置的 safetensors，如果当时手工修改的
H/16 参数与 YAML 不同，脚本会报告从权重形状识别出的结构，按报告调整 legacy
YAML 即可。旧模型预测的是原始 `S1,S2`，因此使用普通 `evaluate.py`，不是 ratio
评估入口。

## Oracle m_gt 条件实验

`configs/polar_jit_h16_oracle_mgt.yaml` 启用法向量与偏振 GT 构造的 7 通道条件：

```text
[S0_R, S0_G, S0_B, theta/pi, cos(2phi), sin(2phi), m_gt]
```

其中 `phi=atan2(ny,nx)`，并按 Oracle 验证计划计算
`m_gt=C_gt*cos(2phi)-S_gt*sin(2phi)`。`C_gt/S_gt` 来自 RGB 平均后的标量
`S1,S2` 方向。除 S0 外的几何通道在 object mask 外置零。该条件在推理时仍需要
normal GT 和 polarization GT，因此仅用于验证 reflection prior，不是可部署输入。

```bash
PYTHONPATH=src python3 scripts/train_oracle_mgt.py --device cuda

PYTHONPATH=src python3 scripts/infer.py \
  --config configs/polar_jit_h16_oracle_mgt.yaml \
  --checkpoint checkpoints/polar_jit_h16_oracle_mgt/model_ema.safetensors
```

## 测试

```bash
python3 -m pip install -e '.[dev]'
pytest -q
```

测试覆盖默认 S1/S2 与独立 ratio 数据转换、模型与 flow 的前向/反向、mask 前景加权、DoLP/AoP
指标、AoP 跨 ±90° 边界的周期误差，以及 DoLP/AoP PNG 的尺寸和格式。

## 推荐实验顺序

1. 比较 Euler 10/20 步和 Heun 10/20 步。
2. 比较载入官方 H/16 权重与从头训练。
3. 比较 S0 空间 token 注入与仅使用全局 AdaLN condition。

## 与旧项目的主要区别

| 项目 | 旧方案 | 本项目 |
|---|---|---|
| 生成空间 | SD VAE latent | 像素空间 S1、S2（可选独立 ratio 实验） |
| 主干 | 12 通道 SD1.5 UNet | JiT |
| 条件网络 | ControlNet | S0 patch embedding |
| 训练目标 | DDPM noise prediction | Flow Matching velocity |
| 推理依赖 | SD1.5、VAE、CLIP | 单一模型 |

## 许可与来源

本项目采用 MIT License。JiT 架构思想与训练参数化参考：

- Tianhong Li, Kaiming He, *Back to Basics: Let Denoising Generative Models Denoise*.
- [LTH14/JiT PyTorch implementation](https://github.com/LTH14/JiT), MIT License.
