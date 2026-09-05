# Polar JiT Flow

一个独立的、像素空间的偏振恢复项目。模型以 RGB `S0` 为条件，直接生成
RGB Stokes 分量 `[S1, S2]`，不依赖 Stable Diffusion、VAE 或 ControlNet。

## 架构

```text
S0 ──► B/16 condition patch embed ──────────┐
       ├─ spatial token addition             ▼
       └─ global pool ──► AdaLN condition ─► 12-block JiT-B/16 ─► clean S1/S2
time ──► timestep embed ────────────────┘             ▲
noise ──► S1/S2 linear flow ─► B/16 patch embed ─────┘
```

- 主干按官方 JiT-B/16 设置为 patch 16、hidden 768、12 blocks、12 heads 和
  bottleneck 128，并采用 RMSNorm、QK-Norm、2D RoPE、SwiGLU 与 AdaLN-Zero。
- 训练路径为 `x_t = t*[S1,S2] + (1-t)*noise`；网络预测 clean S1/S2，并换算为速度场损失。
- S0 patch token 逐位置注入生成 token，并在全局池化后用于调制所有 JiT block。
- 推理只读取 S0，通过 Euler 或 Heun ODE 积分生成 S1/S2。

该实现根据 [LTH14/JiT](https://github.com/LTH14/JiT) 的公开 MIT 实现重新组织。
保留官方 B/16 主干形式，但将 ImageNet 类别条件替换为当前的 S0 空间条件与
全局 AdaLN 条件，不引入类别 embedding、CFG 或官方的 class in-context tokens。

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

分析器图像归一化到 `[0,1]` 后，物理 `S0` 位于 `[0,2]`，`S1/S2` 位于
`[-1,1]`。送入网络前只对 S0 执行 `S0_net=S0-1`，使输入与输出均位于
`[-1,1]`，同时保持三个 Stokes 分量的共同强度尺度。模型条件为
`S0_net [3,H,W]`，生成目标为 `[S1_RGB,S2_RGB] [6,H,W]`。
DoLP 与 AoP 按 RGB 通道直接由 Stokes 定义计算，最终指标在前景像素和
RGB 通道上共同取平均。

默认配置已经指向：

```text
/home/xserver/pjt/datasets/UnifiedSfP_png
```

训练参数均由 `configs/polar_jit_small.yaml` 管理。`train.object_weight` 与
`train.background_weight` 控制生成损失的空间权重；默认前景与背景之比为
`10:0.01`，重点生成 mask 中的 object，背景仅保留极弱约束。

## 安装

```bash
python3 -m pip install -e .
```

## 训练

```bash
python3 scripts/train.py --config configs/polar_jit_small.yaml --device cuda
```

默认配置会从 `jit-b-16/checkpoint-last.pth` 的 `model_ema1` 初始化兼容权重。
官方 RGB patch embed 会额外用于初始化 S0 embedder；time embed、位置编码、
12 个 Transformer blocks、final norm 与 final AdaLN 会按形状迁移。六通道
S1/S2 输入首层、六通道输出层和 S0 全局池化层保持任务专用初始化。设置
`pretrained.enabled: false` 可从头训练。

训练会保存可恢复的 `.pt` checkpoint 和只包含 EMA 模型的
`model_ema.safetensors`。当前入口是单 GPU 版本，结构本身兼容后续 DDP 封装。
监控信息会同时打印到终端并追加保存至输出目录下的 `train_log.jsonl`。每条训练
记录包含当前/总 epoch、epoch 内 batch、当前/总 step、学习率及各项 loss；断点
恢复后 epoch 会根据已完成 step 连续计算。日志文件名可通过 `train.log_file` 修改。

## 推理

```bash
python3 scripts/infer.py \
  --config configs/polar_jit_small.yaml \
  --checkpoint checkpoints/polar_jit_b16_stokes/model_ema.safetensors
```

预测文件为 `[6,H,W]` 的 float32 NPY，通道顺序是
`[S1_R,S1_G,S1_B,S2_R,S2_G,S2_B]`。DoLP 和 AoP 不作为生成通道，而是在
损失与评估阶段由 S0/S1/S2 动态计算。

推理的 `split`、输出目录、采样步数、Euler/Heun 方法、最大样本数、随机种子和
设备默认从 YAML 的 `inference` 段读取，也可用同名命令行参数临时覆盖。例如：

```bash
python3 scripts/infer.py \
  --config configs/polar_jit_small.yaml \
  --checkpoint checkpoints/polar_jit_b16_stokes/model_ema.safetensors \
  --steps 40 --method heun --max-samples 100
```

## 评估

```bash
python3 scripts/evaluate.py \
  --config configs/polar_jit_small.yaml
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
  --config configs/polar_jit_small.yaml \
  --predictions outputs/polar_jit_b16_stokes \
  --output-csv outputs/metrics/experiment.csv \
  --visualization-dir outputs/visualizations/experiment \
  --max-visualizations 50 --fail-on-missing
```

使用 `--no-visualize` 可只计算 CSV 指标。评估结束后，终端还会输出 JSON 汇总，
包括样本数、缺失预测数、可视化数和六项平均指标。

## 测试

```bash
python3 -m pip install -e '.[dev]'
pytest -q
```

测试覆盖 S0/S1/S2 数据转换、模型与 flow 的前向/反向、mask 前景加权、DoLP/AoP
指标、AoP 跨 ±90° 边界的周期误差，以及 DoLP/AoP PNG 的尺寸和格式。

## 推荐实验顺序

1. 比较 Euler 10/20 步和 Heun 10/20 步。
2. 比较载入官方 B/16 权重与从头训练。
3. 比较 S0 空间 token 注入与仅使用全局 AdaLN condition。

## 与旧项目的主要区别

| 项目 | 旧方案 | 本项目 |
|---|---|---|
| 生成空间 | SD VAE latent | 原始 S1/S2 Stokes 像素空间 |
| 主干 | 12 通道 SD1.5 UNet | JiT |
| 条件网络 | ControlNet | S0 patch embedding |
| 训练目标 | DDPM noise prediction | Flow Matching velocity |
| 推理依赖 | SD1.5、VAE、CLIP | 单一模型 |

## 许可与来源

本项目采用 MIT License。JiT 架构思想与训练参数化参考：

- Tianhong Li, Kaiming He, *Back to Basics: Let Denoising Generative Models Denoise*.
- [LTH14/JiT PyTorch implementation](https://github.com/LTH14/JiT), MIT License.
