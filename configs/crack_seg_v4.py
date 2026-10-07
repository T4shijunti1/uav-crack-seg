# -*- coding: utf-8 -*-
"""v4 配置：UAV 裂缝分割（ECA-UNet + 跳跃连接注意力门 + Focal-Tversky + 深度监督）。

在改进配置 ``configs/crack_seg_improved.py`` 的基础上做**单一受控改动**：
在解码器每一级的跳跃连接上加入 **Attention U-Net 风格的注意力门
（SkipAttentionGate）**。门控信号来自解码器更深层、语义更强的特征，在 concat
之前对编码器跳跃特征逐像素加权，自适应抑制背景/噪声区域的无关激活、突出裂缝
边界，从而减少孤立误检（FP）、抬高 Crack IoU / mIoU。

与其他模型的关系（用于集成与 ablation）：
    - improved：无 clDice、无注意力门；
    - v3：improved + clDice（拓扑损失），无注意力门；
    - v4：improved + 跳跃注意力门，**无 clDice**。
v3 与 v4 是改进模型上两条正交、误差特性不同的分支，与 improved 一起做三路
集成可获得最稳的精度增益。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_v4.py --work-dir work_dirs/v4 --amp
"""
# 导入并注册自定义模型（含 UNetECA 的注意力门、FocalTverskyLoss）
custom_imports = dict(imports=['crackseg.models'], allow_failed_imports=False)

# 直接继承改进配置（数据集 / 损失 / 训练管线 / 40k 调度全部沿用）
_base_ = ['./crack_seg_improved.py']

# ---- 单一改动：骨干启用跳跃连接注意力门 ----
model = dict(backbone=dict(use_skip_gate=True))

# 每 2000 迭代验证一次，便于选取 mIoU 最优点（save_best=mIoU 自动保存）。
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
