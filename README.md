# UAV-Crack-Seg：基于 ECA-UNet 的无人机/道路裂缝像素级分割

> **🏆 2026 VLP 挑战赛参赛作品（2026 VLP Challenge Submission）**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-ee4c2c.svg)](https://pytorch.org/)
[![MMSegmentation](https://img.shields.io/badge/MMSegmentation-1.2.2-00b4d8.svg)](https://github.com/open-mmlab/mmsegmentation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/Competition-2026%20VLP%20Challenge-orange.svg)](#-参赛声明)

本项目将 **SAR-ECA-UNet**（一种带高效通道注意力的轻量化 UNet）迁移应用于
**无人机航拍与道路表面裂缝的像素级语义分割**：输入一张图像，输出与原图等大的
二值掩码（裂缝 / 背景），可用于自动化巡检、裂缝定位与量化。

在 ECA-UNet 基础上，本项目进一步引入 **ImageNet 预训练编码器（timm ResNet34 /
ResNet50、EfficientNet-B3/B4、EfficientNetV2-S）+ UNet 解码器（TimmUNet）与
SegFormer（mit_b1）**，并把多个结构差异较大的模型做**概率级跨结构加权集成**，
最终在 170 张验证集上取得 **mIoU 80.53**；首次提交 300 张测试集即取得
**mIoU 80.82、排名 5/39**（见下文“最终方案”）。

---

## ✨ 项目特性

- **编码器阶段通道注意力（ECA）**：在每个编码器阶段之后插入高效通道注意力模块
  （Efficient Channel Attention），增强与裂缝相关的特征通道、抑制噪声通道；
- **深度可分离卷积轻量化（L-Block）**：将卷积块的首个 3×3 卷积替换为
  Depthwise 3×3 + Pointwise 1×1，显著降低参数量与计算量；
- **面向漏检的 Focal-Tversky 损失（改进）**：在 CE 之外引入 Focal-Tversky
  （alpha=0.3、beta=0.7、gamma=4/3），重点惩罚漏报、聚焦难样本，提升召回率；
- **多尺度深度监督（改进）**：辅助头由 1 个扩展为 3 个（解码器 in_index=3/2/1），
  权重逐级减小，强化多尺度特征学习；
- **更高分辨率 + 旋转增强（改进）**：训练裁剪 256→384，并加入 ±30° 随机旋转，
  减少细长裂缝丢失、覆盖裂缝的多样朝向；
- **滑动窗口推理**：对高分辨率无人机图像自动滑窗、重叠融合，避免接缝断裂；
- **可复现 / 可部署**：提供 ONNX 导出脚本与阈值扫描脚本，权重可通过网盘共享。

---

## 📂 目录结构

```text
uav-crack-seg/
├── crackseg/                    # 自定义模型包（导入即注册到 MMSeg）
│   ├── models/
│   │   ├── eca_block.py         # ECA 通道注意力及可选注意力门控
│   │   ├── unet_eca.py          # UNetECA 骨干（ECA + 深度可分离卷积）
│   │   ├── timm_unet.py         # TimmUNet：timm 预训练编码器 + UNet 解码器
│   │   └── losses/
│   │       ├── focal_tversky_loss.py   # Focal-Tversky 损失（改进）
│   │       └── cldice_loss.py          # clDice 连通性损失（v3）
│   └── utils/ops_shim.py        # mmcv._ext 轻量算子桩（免编译）
├── configs/
│   ├── crack_seg_256x256.py     # 基线入口配置
│   ├── crack_seg_improved.py    # 改进入口（384/FocalTversky/3辅助头/旋转）
│   ├── crack_seg_preunet.py     # TimmUNet（ResNet34 预训练）入口
│   ├── crack_seg_preunet_r50.py # TimmUNet（ResNet50 预训练）入口
│   ├── crack_seg_v3.py / v4.py  # clDice / 跳跃门分支
│   ├── crack_seg_dlv3p.py       # DeepLabV3+（对照，已停用）
│   └── _base_/
│       ├── models/sar_eca_unet.py
│       ├── datasets/crack.py
│       ├── schedules/schedule_40k.py
│       └── default_runtime.py
├── scripts/
│   ├── prepare_crack_data.py    # 原始数据 -> MMSeg 格式（含 train/val 切分）
│   ├── batch_inference.py       # 批量推理：掩码/叠加图/results.json
│   ├── threshold_sweep.py       # 阈值扫描（无需重训，选最优判决阈值）
│   ├── ensemble_eval.py         # 等权多模型集成评估
│   └── weighted_ensemble.py     # ★ 加权跨结构集成（最终方案，mIoU>80）
│       make_submission.py       # 生成赛道D提交 result.zip（300 张二值PNG）
├── tools/
│   ├── train.py                 # 训练启动器
│   ├── test.py                  # 测试/评估启动器
│   └── export_onnx.py           # ONNX 导出（PyTorch 模型强制要求）
├── requirements.txt             # pip 依赖
├── environment.yml              # conda 环境文件
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ 环境安装

> 已在 **原生 Windows 11 + Python 3.11 + CUDA 12.8 + RTX 50 系（Blackwell）**
> 上验证通过。本项目使用**纯 Python 的 mmcv-lite**，无需编译 C++/CUDA 算子，
> 因此原生 Windows 即可运行，不强制使用 WSL2。

### 方式一：使用 Conda（推荐）

```powershell
# 创建并激活环境
conda create -n uav_crack python=3.11 -y
conda activate uav_crack

# 安装 CUDA 12.8 版 PyTorch（RTX 50 系必须 cu128；其他显卡按实际 CUDA 选择）
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖（含 mmcv-lite、mmsegmentation）
pip install -r requirements.txt
```

也可直接用环境文件一键创建：`conda env create -f environment.yml`。

### 方式二：CPU / 其他 CUDA 版本

- 无 NVIDIA 显卡时可装 CPU 版 PyTorch（训练很慢，仅建议调试）：
  `pip install torch torchvision`；
- 其他 CUDA 版本请到 [PyTorch 官网](https://pytorch.org/get-started/locally/)
  选择对应索引后，再 `pip install -r requirements.txt`。

### 关于 mmcv（重要）

请勿安装完整版 `mmcv`（在新显卡 / 原生 Windows 上常因缺少编译工具而失败）。
本项目依赖 `mmcv-lite==2.1.0`（纯 Python），完整版缺失的 `mmcv._ext` 由
`crackseg/utils/ops_shim.py` 在 `tools/`、`scripts/` 各入口**自动注入轻量桩**，
无需任何手动配置。

---

## 📊 数据准备

### 原始数据结构（Cityscapes 风格）

本项目使用的 UAV 裂缝数据集为 Cityscapes 风格目录：

```text
UAV-Crack-dataset/
├── leftImg8bit/
│   ├── train/UAV-CrackX4|X8|X16/*.jpg   # 1200 张有标注训练图
│   └── val/*.jpg                        # 300 张无标注图（预留测试/推理）
└── gtFine/
    └── train/UAV-CrackX4|X8|X16/*.png   # 8 位灰度掩码，取值 {0,1}
```

> 标注约定：单通道 8 位灰度 PNG，**背景 = 0，裂缝 = 1**。
> 若你的掩码是 0/255，请先二值化为 0/1（MMSeg 中 255 为“忽略值”）。
>
> 数据集遵循 **CC BY-NC-SA 3.0**，仅限本次比赛非商业使用、禁止二次分发；
> 因此 `data/` 不纳入仓库（已在 `.gitignore` 中忽略）。

### 一键整理为 MMSeg 格式

```bash
# Linux / WSL
python scripts/prepare_crack_data.py \
    --src "/path/to/UAV-Crack-dataset" \
    --dst "data/Crack"

# 原生 Windows（PowerShell）
python scripts/prepare_crack_data.py \
    --src "C:/Users/<你>/Desktop/数据集/UAV-Crack-dataset" \
    --dst "data/Crack"
```

脚本会按 **“同一 DJI 航片帧”分组**切分（默认 15% 验证），避免同一航片的补丁
同时出现在训练与验证集造成数据泄漏，并生成如下目录：

```text
data/Crack/
├── images/
│   ├── train/*.jpg   val/*.jpg   test/*.jpg（无标注）
└── annotations/
    ├── train/*.png   val/*.png
```

---

## 🚀 模型训练

```bash
# 建议先做冒烟测试（仅 20 iter + 验证）确认环境无误
python tools/train.py configs/crack_seg_256x256.py \
    --work-dir work_dirs/smoke --amp \
    --cfg-options train_cfg.max_iters=20 train_cfg.val_interval=20

# 基线正式训练（默认 40000 iter，256 裁剪，batch 8）
python tools/train.py configs/crack_seg_256x256.py --work-dir work_dirs/crack --amp

# ★ 改进模型正式训练（384 裁剪，batch 6，Focal-Tversky + 3 辅助头 + 旋转）
python tools/train.py configs/crack_seg_improved.py --work-dir work_dirs/improved --amp

# TimmUNet（timm ResNet34 预训练编码器 + UNet 解码器，384，batch 6，AdamW，24k）
python tools/train.py configs/crack_seg_preunet.py --work-dir work_dirs/preunet --amp

# TimmUNet（timm ResNet50 预训练编码器，384，batch 4，AdamW，24k）
python tools/train.py configs/crack_seg_preunet_r50.py --work-dir work_dirs/preunet_r50 --amp
```

主要超参数（可在 `configs/` 中修改）：

| 设置 | 基线 | 改进 |
| --- | --- | --- |
| 训练裁剪尺寸 | 256 × 256 | 384 × 384 |
| Batch size | 8 | 6 |
| 训练迭代 | 40000 | 40000 |
| 优化器 | SGD（momentum 0.9，weight decay 5e-4） | 同左 |
| 学习率 | 0.01，Poly 衰减，预热 500 iter | 同左 |
| 主头损失 | CE + Dice | CE(0.6) + Focal-Tversky(1.0) |
| 辅助头 | 1 个（in_index=3，权重 0.4） | 3 个（in_index=3/2/1） |
| 训练增强 | 翻转/色彩/缩放 | 额外加入 ±30° 随机旋转 |
| 滑窗推理 | crop256 / stride170 | crop(368,384) / stride(245,256) |

---

## 📈 测试与评估

```bash
python tools/test.py configs/crack_seg_256x256.py \
    work_dirs/crack/best_mIoU_iter_40000.pth \
    --work-dir work_dirs/crack/eval \
    --show-dir work_dirs/crack/vis
```

评估指标对齐赛道 D：**F1、mIoU、Crack IoU、Acc、Pr、Re**。
配置中的 IoUMetric 会同时计算 mIoU 与逐类 IoU（Crack IoU）、mDice（即 F1）、
mFscore（精确率 Pr、召回率 Re）以及 aAcc（总体准确率 Acc）。
权重文件名以 `work_dirs/crack/` 中实际生成的为准（`best_mIoU_*.pth` 或 `iter_40000.pth`）。

### 单模型结果（170 张验证集，独立评估）

赛道 D 六项指标（官方顺序 F1 / mIoU / Crack IoU / Acc / Pr / Re）：

| 模型 | F1 | mIoU | Crack IoU | Acc | Pr | Re |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 ECA-UNet（40k） | 69.58 | 75.29 | 53.35 | 97.32 | 81.37 | 60.77 |
| ECA-UNet improved（best, 24k） | 75.74 | 79.21 | 60.96 | 97.56 | 75.91 | 75.58 |
| ECA-UNet v3（clDice，best 32k） | — | 79.07 | — | — | — | — |
| ECA-UNet v4（跳跃门，best 26k） | — | 79.02 | — | — | — | — |
| DeepLabV3+ + ResNet50_v1c（best 6k） | — | 75.11 | — | — | — | — |
| **TimmUNet（timm ResNet34 预训练，best 20k）** | 76.46 | **79.67** | 61.89 | 97.56 | 74.26 | 78.79 |
| TimmUNet（timm ResNet50 预训练，best 12k） | — | 79.16 | — | 97.58 | 87.70 | — |
| TimmUNet（ResNet50 短程补训，best 3k） | 76.49 | **79.75** | 61.94 | 97.66 | 77.37 | 75.64 |
| TimmUNet（EfficientNet-B3 预训练，best 18k） | 76.31 | 79.57 | 61.70 | 97.54 | 74.09 | 78.67 |
| TimmUNet（EfficientNet-B4 预训练，best 8k） | 76.44 | 79.65 | 61.86 | 97.54 | 73.92 | 79.14 |
| SegFormer（mit_b1，ADE20K 骨干，best 18k） | 75.69 | 79.09 | 60.89 | 97.40 | 71.57 | 80.31 |
| **TimmUNet（EfficientNetV2-S，21ft1k，EMA，best 14k）** | 76.82 | **79.93** | 62.37 | 97.59 | 74.56 | 79.23 |

> - ECA-UNet improved 通过 Focal-Tversky 把召回从 60.8 拉到 75.6，F1 提升约 6.2；
> - DeepLabV3+ 的大视野 ASPP（dilation 12/24/36）不适合纤细裂缝，落后约 4 点，已停用；
> - TimmUNet 借助 ImageNet 预训练编码器，单模 mIoU 即达 79.67、召回 78.8；
> - r50ft 偏高精确率（Pr 77.4）、b4 偏高召回（Re 79.1），二者 Pr/Re 互补，利于集成；
> - SegFormer（Transformer）召回最高（80.3）但精确率偏低；EfficientNetV2-S 配合
>   EMA 是最强单模（79.93），且与 B 系列误差模式不同。

### 最终方案：跨结构加权集成（mIoU > 80）

同数据/增强/容量的 ECA-UNet 各分支误差高度相关，等权集成饱和于约 **79.63**；
而 TimmUNet（预训练编码器）与 ECA-UNet 的误差相关性更低。把四个模型的滑窗
softmax 概率按 **3 : 2 : 1 : 1** 加权平均、阈值 **0.50** 判决，得到：

| 方案 | F1 | mIoU | Crack IoU | Acc | Pr | Re |
| --- | --- | --- | --- | --- | --- | --- |
| ECA-UNet improved（单模基线） | 75.74 | 79.21 | 60.96 | 97.56 | 75.91 | 75.58 |
| **最终加权集成（3:2:1:1，t=0.50）** | **76.98** | **80.06** | **62.57** | **97.65** | **76.06** | **77.91** |
| 相对单模基线 | **+1.24** | **+0.85** | **+1.61** | **+0.09** | **+0.15** | **+2.33** |

**六项指标全部提升、无一下降，mIoU 突破 80。** 复现命令（仓库根目录）：

```bash
python scripts/weighted_ensemble.py \
    --member configs/crack_seg_preunet.py work_dirs/preunet/best_mIoU_iter_20000.pth 3 \
    --member configs/crack_seg_improved.py work_dirs/improved/best_mIoU_iter_24000.pth 2 \
    --member configs/crack_seg_v3.py work_dirs/v3/best_mIoU_iter_32000.pth 1 \
    --member configs/crack_seg_v4.py work_dirs/v4/best_mIoU_iter_26000.pth 1
```

> 权重与阈值均为**推理超参数**，在验证集上做模型选择，未改动任何标签或权重；
> 结果同时保存在 `work_dirs/ensemble/weighted_ensemble.json`。
>
> **阈值前沿（如实记录）**：上述权重下，阈值 0.46–0.50 的 mIoU 均为 80.06–80.07；
> 若更看重精确率，阈值 0.55 时 Pr=77.24、mIoU=80.00（仍 ≥80）。
>
> **预训练权重来源（合规说明）**：TimmUNet 使用 timm 提供的 ImageNet（RSB/A1）
> ResNet34 / ResNet50 权重
> （`resnet34_a1_0-46f8f793.pth`、`resnet50_a1_0-14fe96d1.pth`，均经 timm 官方
> GitHub Release 获取），仅作编码器初始化并在本数据集上微调；比赛技术要求未限制
> 网络结构或预训练权重，已在致谢注明来源。

#### 加入 ResNet50 TimmUNet（5 成员，当前最佳）

把 ResNet50 TimmUNet（best 12k，mIoU 79.16）作为第 5 个成员，对权重 × 阈值再做
网格搜索，最优为权重 **3 : 2 : 1 : 1 : 1（r34 : improved : v3 : v4 : r50）**、
阈值 **0.46**：

| 方案 | F1 | mIoU | Crack IoU | Acc | Pr | Re |
| --- | --- | --- | --- | --- | --- | --- |
| 4 成员加权集成（3:2:1:1，t=0.50） | 76.98 | 80.06 | 62.57 | 97.65 | 76.06 | 77.91 |
| **5 成员加权集成（3:2:1:1:1，t=0.46）** | **77.03** | **80.09** | **62.64** | 97.64 | 75.49 | **78.63** |

> 加入 r50 仅再 +0.03 mIoU，说明 UNet 家族在本数据上的集成已接近饱和；
> 若更看重精确率，可选权重 3:2:1:1:2、阈值 0.54：Pr=77.62、Acc=97.69、mIoU=80.01。

```bash
# 权重传 0 触发 5 成员权重模板自动搜索（顺序：r34, improved, v3, v4, r50）
python scripts/weighted_ensemble.py \
    --member configs/crack_seg_preunet.py work_dirs/preunet/best_mIoU_iter_20000.pth 0 \
    --member configs/crack_seg_improved.py work_dirs/improved/best_mIoU_iter_24000.pth 0 \
    --member configs/crack_seg_v3.py work_dirs/v3/best_mIoU_iter_32000.pth 0 \
    --member configs/crack_seg_v4.py work_dirs/v4/best_mIoU_iter_26000.pth 0 \
    --member configs/crack_seg_preunet_r50.py work_dirs/preunet_r50/best_mIoU_iter_12000.pth 0
```


#### 加入 EfficientNet-B3/B4 与 r50ft（最终 5 强模型，mIoU 80.46）

进一步训练结构差异更大的 ImageNet 预训练编码器：EfficientNet-B3（best 18k，
79.57）、EfficientNet-B4（best 8k，79.65，高召回 Re 79.1），并对 ResNet50 做
短程补训（r50ft，best 3k，79.75，高精确率 Pr 77.4）。把成员扩展到 7 个并对
权重 × 阈值做网格搜索，最优权重为 **(r34, improved, v3, v4, r50ft, b3, b4) =
(3, 2, 0, 0, 3, 3, 3)**、阈值 **0.47**——v3/v4 权重自动降为 0（强多样模型存在后
已无贡献），最终有效集成是 5 个强模型：

| 方案 | F1 | mIoU | Crack IoU | Acc | Pr | Re |
| --- | --- | --- | --- | --- | --- | --- |
| 5 成员旧集成（3:2:1:1:1，t=0.46） | 77.03 | 80.09 | 62.64 | 97.64 | 75.49 | 78.63 |
| **最终 5 强模型集成（3:2:3:3:3，t=0.47）** | **77.54** | **80.46** | **63.31** | **97.70** | **76.41** | **78.69** |

> 推理端已验证无额外增益：更密的滑窗步长（stride (184,192) 或 (122,128)）结果
> 与默认完全一致（图像仅 672×378，crop 368×384 已覆盖绝大部分）；同结构继续加
> 模型增益递减（80.09→80.36→80.46）。

```bash
# 7 成员权重传 0 触发 _7 模板自动搜索（顺序：r34, improved, v3, v4, r50ft, b3, b4）
python scripts/weighted_ensemble.py \
    --member configs/crack_seg_preunet.py work_dirs/preunet/best_mIoU_iter_20000.pth 0 \
    --member configs/crack_seg_improved.py work_dirs/improved/best_mIoU_iter_24000.pth 0 \
    --member configs/crack_seg_v3.py work_dirs/v3/best_mIoU_iter_32000.pth 0 \
    --member configs/crack_seg_v4.py work_dirs/v4/best_mIoU_iter_26000.pth 0 \
    --member configs/crack_seg_preunet_r50_ft.py work_dirs/preunet_r50_ft/best_mIoU_iter_3000.pth 0 \
    --member configs/crack_seg_preunet_b3.py work_dirs/preunet_b3/best_mIoU_iter_18000.pth 0 \
    --member configs/crack_seg_preunet_b4.py work_dirs/preunet_b4/best_mIoU_iter_8000.pth 0
```

##### 首次测试集成绩（已上榜）

将该集成（当时为 6 成员、验证 80.36 版）提交至平台，300 张测试集成绩为
**mIoU 80.82、Crack F1 78.33、Crack IoU 64.38、Crack Precision 76.65、
Crack Recall 80.08、aAcc 97.40，排名 5/39**。测试 mIoU 比本地验证高约 0.73；
含 b4 的最终 5 强模型版（验证 80.46）预计测试约 81.2。

#### 加入 SegFormer 与 EfficientNetV2-S（最终 7 模型，mIoU 80.53）

为进一步抬高上限，再训练两类成员：
    * **SegFormer（mit_b1）**：与卷积网络结构差异最大的 Transformer，骨干取自
      mmseg 官方 ADE20K 检查点（仅取 ``backbone.*``），best 18k、mIoU 79.09，
      召回最高（80.3）、精确率偏低（71.6）；
    * **EfficientNetV2-S + EMA**：ImageNet-21K→1K 预训练（前期 Fused-MBConv），
      训练中启用 EMA（momentum=1e-3），best 14k、mIoU **79.93，为最强单模**。

把成员扩展到 9 个（v3/v4 仍为 0）做权重 × 阈值网格，最优为
**(r34, improved, v3, v4, r50ft, b3, b4, segformer, v2s) =
(3, 2, 0, 0, 3, 3, 3, 2, 3)**、阈值 **0.49**，最终有效集成是 7 个模型：

| 方案 | F1 | mIoU | Crack IoU | Acc | Pr | Re |
| --- | --- | --- | --- | --- | --- | --- |
| 最终 5 强模型集成（3:2:3:3:3，t=0.47） | 77.54 | 80.46 | 63.31 | 97.70 | 76.41 | 78.69 |
| **最终 7 模型集成（3:2:3:3:3:2:3，t=0.49）** | **77.63** | **80.53** | **63.44** | **97.71** | **76.59** | **78.70** |

> 加入 SegFormer / V2-S 仅再 +0.07 mIoU（80.46→80.53），跨结构集成也已接近
> 饱和；按测试集约 +0.7 的偏移，该版测试 mIoU 预计约 81.2–81.3。

```bash
# 9 成员权重传 0 触发 _9 模板自动搜索
# 顺序：r34, improved, v3, v4, r50ft, b3, b4, segformer, v2s
python scripts/weighted_ensemble.py \
    --member configs/crack_seg_preunet.py work_dirs/preunet/best_mIoU_iter_20000.pth 0 \
    --member configs/crack_seg_improved.py work_dirs/improved/best_mIoU_iter_24000.pth 0 \
    --member configs/crack_seg_v3.py work_dirs/v3/best_mIoU_iter_32000.pth 0 \
    --member configs/crack_seg_v4.py work_dirs/v4/best_mIoU_iter_26000.pth 0 \
    --member configs/crack_seg_preunet_r50_ft.py work_dirs/preunet_r50_ft/best_mIoU_iter_3000.pth 0 \
    --member configs/crack_seg_preunet_b3.py work_dirs/preunet_b3/best_mIoU_iter_18000.pth 0 \
    --member configs/crack_seg_preunet_b4.py work_dirs/preunet_b4/best_mIoU_iter_8000.pth 0 \
    --member configs/crack_seg_segformer.py work_dirs/segformer/best_mIoU_iter_18000.pth 0 \
    --member configs/crack_seg_preunet_v2s.py work_dirs/preunet_v2s/best_mIoU_iter_14000.pth 0
```

### 生成平台提交文件 result.zip

赛道 D 平台（http://vlp.chd.edu.cn/tasks/4）要求提交 300 张测试图的二值掩码，
**不是 JSON**。`scripts/make_submission.py` 直接用上述最终 7 模型加权集成
（r34 : improved : r50ft : b3 : b4 : segformer : v2s = 3:2:3:3:3:2:3、
阈值 0.49）对 300 张测试图推理，并按平台规范打包：

```bash
python scripts/make_submission.py          # 默认阈值 0.49
```

输出 `work_dirs/submission/result.zip`，zip 内结构为 `result/*.png`：

- 恰好 300 张单通道 PNG，分辨率 **672×378**；
- 像素 **0=路面、1=裂缝**；
- 文件名与测试 `.jpg` 完全对应、仅扩展名改为 `.png`；
- 不含任何多余文件（脚本末尾会自动做结构/数量/尺寸/取值自检）。

---

## 🔍 批量推理

对 `data/Crack/images/test` 中的 300 张无标注图批量推理，输出二值掩码与红色叠加图：

```bash
python scripts/batch_inference.py \
    --config configs/crack_seg_256x256.py \
    --checkpoint work_dirs/crack/best_mIoU_iter_40000.pth \
    --input data/Crack/images/test \
    --out-dir work_dirs/crack/predictions
```

结果位于：

```text
work_dirs/crack/predictions/
├── masks/      # 二值掩码（裂缝=255）
└── overlay/    # 红色半透明叠加图
```

模型已配置滑动窗口推理（256 窗口 / 170 步长 / 重叠 86 px），更大的原始无人机
图像同样可直接推理；如需保留 GeoTIFF 地理坐标或导出 Shapefile，可结合 `rasterio` 处理。

---

## 🧬 ONNX 导出与权重提交（可复现要求）

比赛要求提供可执行的模型权重；若使用 PyTorch，还需**额外提供 ONNX 格式转换文件**：

```bash
# 训练得到 .pth 后导出 ONNX（脚本会自动做数值一致性校验）
python tools/export_onnx.py \
    --config configs/crack_seg_256x256.py \
    --checkpoint work_dirs/crack/best_mIoU_iter_40000.pth \
    --output work_dirs/crack/crack_seg.onnx
```

随后将 **`.pth` 与 `.onnx`** 一并上传至**百度网盘 / 阿里云盘**共享，链接有效期需
**≥ 90 天**，并在提交表单中填写下载链接与提取码（文件大小 ≤ 20GB）。

> ONNX 输入为已归一化 RGB（1×3×256×256），输出为 2 类 logits，argmax 即得 {0,1}（1=裂缝）。

---

## ✅ 提交清单（平台需 4 项材料，缺一不可）

1. **测试集推理结果文件**：赛道 D 为 **result.zip**（内含 `result/` 文件夹 + 恰好
   300 张 672×378、取值 0/1 的二值 PNG，文件名与测试图对应），由
   `scripts/make_submission.py` 一键生成；不是 JSON；
2. **开源代码仓库 URL**：即本 GitHub 公开仓库地址；
3. **模型权重下载链接与提取码**：含 `.pth` + `.onnx`，网盘链接有效期 ≥ 90 天；
4. **技术报告（PPT，≤ 5 页）**：方法概述、创新点、实验结果、ablation study。

---

## ❓ 常见问题

1. **`No CUDA / torch 不可用`**：PyTorch 装成了 CPU 版本，请按 CUDA 版本重装对应 wheel。
2. **`mmcv 安装/编译失败`**：请改装纯 Python 的 `mmcv-lite==2.1.0`（见上文“关于 mmcv”），
   不要安装完整版 mmcv；缺失算子由仓库内桩自动补齐。
3. **预测全黑 / 裂缝学不到**：检查掩码是否为 {0,1}（若为 255 会被当作忽略值），
   并确认损失配置正确；召回偏低可使用改进配置或下调判决阈值。
4. **CUDA 显存不足**：调小 `batch_size`（如 4）、启用 `--amp`。
5. **单卡报 SyncBN 错误**：本项目默认使用 BN；若自行改为 SyncBN，请在多卡分布式下使用。
6. **指标虚高**：确认 train/val 按航片帧分组切分，而非按补丁随机切分。
7. **不想重训、只想提 F1/召回**：运行 `scripts/threshold_sweep.py` 选取最优阈值，
   并在测试集上使用同一阈值规则。

---

## 🙏 致谢与引用

本项目基于以下开源工作，在此致谢：

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)（OpenMMLab）；
- [timm（PyTorch Image Models）](https://github.com/huggingface/pytorch-image-models)，
  Wightman et al.，提供 ImageNet 预训练 ResNet34/ResNet50（RSB/A1）、EfficientNet-B3/B4
  与 EfficientNetV2-S（21ft1k）编码器权重；
- [SegFormer](https://arxiv.org/abs/2105.15203)，Xie et al., NeurIPS 2021；mit_b1 骨干
  权重取自 mmseg 官方 ADE20K 检查点；
- [ECA-Net: Efficient Channel Attention](https://doi.org/10.1109/CVPR42600.2020.01155)，Wang et al., CVPR 2020；
- SAR-ECA-UNet：[LZS1991/SAR-ECA-UNet](https://github.com/LZS1991/SAR-ECA-UNet)；
- U-Net：[Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597)。

如本项目对你有帮助，欢迎 Star 并引用相关基础工作。

---

## 📌 参赛声明

本仓库为 **2026 VLP 挑战赛（2026 VLP Challenge）参赛作品**，包含完整的代码、
注释、环境配置文件与运行说明。仓库以 **公开（Public）** 权限发布，遵循 MIT License。

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。数据集的使用请遵循其各自的许可协议。
