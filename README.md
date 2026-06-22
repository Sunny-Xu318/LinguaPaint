# LinguaPaint — 文本引导跨模态图像修复

LinguaPaint 是一个基于文本引导的图像修复（image inpainting）项目。给定一张存在缺损的图像、一张二值 mask 以及一段对画面内容的文本描述，模型会结合视觉上下文与文本语义生成缺损区域的内容，使修复结果在语义、结构和纹理上保持一致。

项目使用 PyTorch 实现，整体采用 ControlNet 风格的单路径 U-Net 架构：CLIP 文本编码器在瓶颈处通过自注意力 + 跨模态注意力直接注入文本语义，配合 LPIPS 感知损失与 Classifier-Free Guidance（CFG），适用于自然图像、艺术图像、文物影像、用户内容平台等多种需要语义可控修复的场景。

## 核心特性

- **CLIP 文本引导**：使用预训练 CLIP 文本编码器（冻结）提取词级与句级特征，在 U-Net 瓶颈处通过跨模态注意力注入文本语义。
- **ControlNet 风格条件注入**：单路径 U-Net 生成器，瓶颈使用堆叠的 Transformer block（自注意力 + 文本交叉注意力 + FFN），结构对齐于现代扩散/可控生成模型。
- **U-Net + Self-Attention Bottleneck**：4 级下采样到 ``image_size / 8`` 的瓶颈分辨率，再叠加 N 个 Transformer block 进行长距离建模与文本条件注入；解码器 3 级上采样并使用跳跃连接保留高分辨率细节。
- **LPIPS 感知损失**：在像素级 L1 之外引入基于 VGG 的 LPIPS 感知损失，显著提升修复结果的纹理与结构真实感。
- **Classifier-Free Guidance**：训练阶段以一定概率用空文本替换条件，提升模型对“无文本”输入的建模能力；推理阶段通过条件 / 非条件输出加权组合得到可调节的文本引导强度。
- **CLIP 驱动的图文匹配**：DAMSM 风格图文匹配损失同时使用冻结的 CLIP 文本与视觉编码器，仅训练投影层，提供高质量、细粒度的语义监督。
- **训练稳定性优化**：内置生成器 EMA、梯度裁剪、判别器梯度门控等工程化优化。

## 架构概览

整体网络由生成器（单路径 U-Net）和单个 PatchGAN 判别器组成，生成器内部包含以下组件：

| 模块 | 作用 |
| --- | --- |
| `TextEncoder` | 冻结 CLIP 文本编码器 + 可训练投影层，输出词级特征 `W` 和句级特征 `s`；同时缓存空文本 token，供 CFG 使用 |
| `ConditionalAugmentation` | 对句向量重参数化采样，缓解文本稀疏性 |
| `ImageEncoder` | U-Net 风格图像编码器，4 级下采样输出 `image_size/8` 瓶颈特征及 3 个跳跃特征 |
| `TransformerBottleneck` | 堆叠的 Transformer block（self-attn + 文本 cross-attn + FFN），在瓶颈处注入文本语义 |
| `Decoder` | 3 级上采样的 U-Net 解码器，使用跳跃连接保留细节，输出 `tanh` 范围的图像 |
| `SNPatchDiscriminator` | 谱归一化 PatchGAN 判别器，使用 hinge loss 约束修复区域真实感 |
| `ClipImageEncoder` | DAMSM 损失专用，基于冻结 CLIP 视觉编码器 + 可训练投影层 |
| `LPIPSLoss` | 基于 VGG 的 LPIPS 感知损失，权重已冻结 |

数据流（训练态）：

```text
Text  ──► CLIP (frozen) ──► Projection ──┬──► Word features ──┐
                                         └──► Sentence ──► CA ─┤
                                                               │
                                              text features ◄──┘
                                                  │
Masked image + Mask ──► ImageEncoder ──► Bottleneck (Self-Attn + Cross-Attn(text)) ──► Decoder ──► Inpainted image
                              └──► skip connections ─────────────────────────────────────┘
```

推理时若启用 CFG，会同时跑一次条件 forward 与一次空文本 forward，输出按 `output = uncond + s · (cond − uncond)` 组合，再与原图 mask 合成。

## 项目结构

```text
.
├── configs/
│   └── default.yaml          # 训练默认配置
├── src/
│   ├── __init__.py
│   ├── data.py               # JSONL 数据集、CLIP tokenizer 封装
│   ├── losses.py             # KL、L1、LPIPS、CLIP 图文匹配、对抗损失
│   ├── model.py              # 生成器、判别器、CLIP 文本/视觉编码器封装
│   └── utils.py              # 配置加载、随机种子、EMA、梯度裁剪等
├── train.py                  # 训练入口
├── test.py                   # 推理入口
├── requirements.txt
└── README.md
```

## 环境配置

### 依赖要求

- Python >= 3.8
- PyTorch >= 1.13（建议 2.x，TransformerBottleneck 使用 SDPA）
- CUDA GPU（推荐 12GB 以上显存）

### 安装步骤

```bash
conda create -n linguapaint python=3.9 -y
conda activate linguapaint

pip install -r requirements.txt
```

如果默认安装的 PyTorch 与本机 CUDA 不匹配，请到 [PyTorch 官网](https://pytorch.org/) 选择对应的安装命令。

`requirements.txt` 中的依赖：

- `torch`、`torchvision`：模型训练与推理
- `transformers`：加载预训练 CLIP 文本编码器、视觉编码器与 tokenizer
- `lpips`：LPIPS 感知损失
- `Pillow`：图像 I/O
- `PyYAML`：配置文件解析
- `tqdm`：训练进度条
- `numpy`：数值计算

首次启动时会自动从 HuggingFace Hub 拉取 CLIP 权重（默认 `openai/clip-vit-base-patch32`，文本与视觉编码器合计约 600 MB），并从 `lpips` 包内置链接拉取 VGG 权重。如运行环境无法访问外网，可提前在有网络的机器上预下载并通过 `HF_HOME`、`TORCH_HOME` 环境变量指向本地缓存目录。

## 数据准备

### 数据格式

训练与验证使用 JSONL 文件，每行一个样本：

```json
{"image": "images/0001.jpg", "mask": "masks/0001.png", "text": "a colorful bird with long tail feathers"}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `image` | 完整真值图像。 |
| `mask` | 缺损区域 mask，单通道；非零像素表示需要修复的区域。 |
| `text` | 描述图像内容的文本，建议为英文，包含主体、动作、颜色、形状、纹理等信息。 |

路径既可以是绝对路径，也可以是相对于 manifest 文件所在目录的相对路径。

### 推荐目录结构

```text
data/
├── train.jsonl
├── val.jsonl
├── images/
│   ├── 0001.jpg
│   └── 0002.jpg
└── masks/
    ├── 0001.png
    └── 0002.png
```

### 数据建议

- 图像建议预处理到 256×256 或更高分辨率，避免过度压缩。
- mask 建议同时包含规则中心 mask 与不规则缺损 mask，提升模型对真实损伤模式的鲁棒性。
- 文本描述应聚焦图像主体，过短或过于宽泛的描述会削弱文本引导的作用。

## 训练

### 启动训练

```bash
python train.py --config configs/default.yaml
```

训练流程：

1. 加载 CLIP tokenizer、冻结的 CLIP 文本编码器与视觉编码器，并按需初始化冻结的 LPIPS-VGG。
2. 加载图像、mask、文本三元组，组建 mini-batch。
3. 单次生成器 forward（按 `cfg_dropout` 概率随机置空文本），先更新 PatchGAN 判别器，再更新生成器与 DAMSM 投影层。
4. 每个 step 进行梯度裁剪与生成器 EMA 更新。
5. 每个 epoch 结束保存样例可视化、最新 checkpoint，并按 `save_every` 周期保存阶段性 checkpoint。

### 输出目录

```text
runs/linguapaint/
├── vocab.json                      # CLIP tokenizer 元信息（记录使用的 CLIP 模型名）
├── checkpoints/
│   ├── latest.pt                   # 最近一次 checkpoint
│   └── epoch_xxxx.pt               # 周期性 checkpoint
└── samples/
    └── epoch_xxxx.png              # 可视化样例（masked / inpainted / GT）
```

### 断点续训

```bash
python train.py \
    --config configs/default.yaml \
    --resume runs/linguapaint/checkpoints/latest.pt
```

恢复时会一并加载模型、判别器、CLIP 视觉编码器投影层、优化器与 EMA 状态。

## 推理

### 单图修复

```bash
python test.py \
    --checkpoint runs/linguapaint/checkpoints/latest.pt \
    --image data/images/0001.jpg \
    --mask data/masks/0001.png \
    --text "a colorful bird with long tail feathers" \
    --guidance-scale 3.0 \
    --output result.png
```

| 参数 | 说明 |
| --- | --- |
| `--checkpoint` | 训练得到的模型权重文件。 |
| `--image` | 待修复图像，缺损区域可为黑色或任意值。 |
| `--mask` | 缺损 mask，非零像素表示需要修复。 |
| `--text` | 文本描述，引导缺损区域生成。 |
| `--output` | 修复结果保存路径。 |
| `--vocab` | 可选，CLIP tokenizer 元信息文件路径，默认从 checkpoint 同级目录的 `vocab.json` 读取。 |
| `--device` | 可选，覆盖配置中的设备（如 `cuda`、`cpu`）。 |
| `--use-ema` | 可选，加载 EMA 权重进行推理（通常更稳定，质量更高）。 |
| `--guidance-scale` | 可选，CFG 强度。`1.0` 为纯条件输出，`>1` 增强文本控制。默认读取 config 中 `inference.guidance_scale`。 |

推理时输出图像中非 mask 区域将完整保留原图像素。当 `--guidance-scale != 1.0` 时模型会做两次 forward（条件 + 空文本）然后线性组合。

### 推理质量建议

- 文本描述应与图像主体强相关，建议包含主体、颜色和动作信息。
- 复杂破损建议先用形态学操作适当扩展 mask，避免边界 artifacts。
- 启用 `--use-ema` 通常能获得更平滑、更稳定的修复结果。
- CFG 调参经验：`1.0` 弱引导、`3.0` 默认、`5–7.5` 强引导（注意过高可能产生过饱和或失真）。

## 配置说明

`configs/default.yaml` 中的核心字段：

```yaml
seed: 42
device: cuda

data:
  train_manifest: data/train.jsonl
  val_manifest: data/val.jsonl
  image_size: 256
  max_words: 77              # CLIP tokenizer 最大长度
  num_workers: 4

model:
  clip_model_name: openai/clip-vit-base-patch32  # HuggingFace CLIP 模型名
  text_hidden_dim: 256       # 文本投影层输出维度，与下游模块共享
  base_channels: 64          # 图像编码器/解码器基础通道数
  ca_dim: 256                # 条件增强维度
  attn_heads: 4              # 多头注意力头数
  num_bottleneck_blocks: 2   # 瓶颈处 Transformer block 数量

train:
  epochs: 100
  batch_size: 8
  lr_g: 0.0001               # 生成器学习率
  lr_d: 0.0001               # 判别器学习率
  betas: [0.5, 0.999]
  out_dir: runs/linguapaint
  save_every: 5              # 周期性 checkpoint 间隔
  log_every: 20              # 日志刷新步数
  grad_clip: 1.0             # 生成器梯度裁剪阈值，0 表示关闭
  ema_decay: 0.999           # 生成器 EMA 衰减率，0 表示关闭
  cfg_dropout: 0.1           # 训练时随机用空文本替换条件的概率，0 表示禁用 CFG

loss:
  lambda_kl: 0.01            # CA KL 散度权重
  lambda_app: 10.0           # L1 像素损失权重
  lambda_lpips: 1.0          # LPIPS 感知损失权重，0 表示关闭并跳过加载 VGG
  lambda_damsm: 0.5          # CLIP DAMSM 图文匹配损失权重
  lambda_adv: 0.1            # PatchGAN 对抗损失权重
  lpips_net: vgg             # LPIPS backbone：vgg / alex / squeeze

inference:
  guidance_scale: 3.0        # 推理默认 CFG 强度，可被命令行 --guidance-scale 覆盖
```

### 调参建议

- 显存不足：减小 `batch_size`、`base_channels`、`num_bottleneck_blocks` 或 `image_size`；显存最紧张时可将 `lambda_lpips` 设为 0 跳过 VGG。
- 训练不稳定：降低 `lr_g`、`lambda_adv`，开启 `grad_clip`，或提高 `ema_decay`。
- 文本引导偏弱：提高 `cfg_dropout`（推理 CFG 才能起作用）、`lambda_damsm`、推理时增大 `--guidance-scale`，并确保训练文本描述质量。
- 修复结果模糊：提高 `lambda_lpips` 与 `lambda_adv`，或增加训练轮数。
- 修复区域语义飘移：降低 `--guidance-scale`、提高 `lambda_app`，或检查 mask 是否过大。

## 应用场景

LinguaPaint 适用于多种基于文本语义控制的图像修复需求，例如：

- 老照片、文物影像、艺术作品的破损修复。
- 用户上传图像中的目标移除与背景填充。
- 受损视频帧、医学影像等专业图像的语义级补全。
- 任意需要“文本可控生成 + 区域级修复”的下游业务。

## 常见问题

**首次启动时 HuggingFace CLIP 权重下载失败？**
请检查网络连通性，或在有网络的机器上预下载 `openai/clip-vit-base-patch32` 后通过 `HF_HOME` 环境变量指向缓存目录；也可以替换为已有的本地 CLIP 模型路径。

**LPIPS / VGG 权重下载失败？**
LPIPS 包首次运行会下载 VGG 权重到 `~/.cache/torch/hub/checkpoints/`。可在有网络环境预下载，或将 `lambda_lpips` 设为 0 暂时关闭 LPIPS。

**想换更大的 CLIP 模型？**
将 `model.clip_model_name` 设为 `openai/clip-vit-large-patch14` 等更大模型即可，文本与视觉编码器会同时切换；投影层会自动适配输入维度，无需调整其他超参，但显存占用会显著增加。

**CFG 应该设多大？**
推荐起步 `3.0`。若文本控制力不够、生成偏离描述，可提高到 `5–7.5`；若发现颜色饱和度异常、纹理过强，则适当降低。注意必须先在训练阶段启用 `cfg_dropout`（默认 0.1），否则推理时 CFG 没有意义。

**推理结果中 mask 区域出现明显边界？**
检查 mask 是否为干净二值，避免羽化或反锯齿；必要时在调用前对 mask 做 `>0.5` 二值化。

**训练时显存爆炸？**
将 `train.batch_size` 降到 4、`model.base_channels` 降到 32、`num_bottleneck_blocks` 降到 1 通常足够运行；若仍显存紧张可进一步降低 `data.image_size` 至 192 或 128，或将 `lambda_lpips` 设为 0。

**判别器损失迅速归零或飞涨？**
属于 GAN 训练常见现象。可先降低 `lr_d` 或 `lambda_adv`，并保持 `grad_clip` 开启。

**断点恢复后训练损失出现跳变？**
EMA 权重在新一轮训练初期需要预热，属于正常现象，建议继续训练数千步后再观察。

