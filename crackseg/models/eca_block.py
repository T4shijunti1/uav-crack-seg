# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""高效通道注意力（ECA）及可选的注意力门控模块。

参考文献：
    - ECA-Net: Efficient Channel Attention for Deep Convolutional Neural
      Networks, Wang et al., CVPR 2020.
      https://doi.org/10.1109/CVPR42600.2020.01155
    - Attention U-Net: Learning Where to Look for the Pancreas,
      Oktay et al., MIDL 2018.

本文件包含三个模块：
    1. ECABlock：核心模块，使用一维自适应卷积做跨通道交互，参数量极小；
    2. SpatialGate：零参数（+9 参数平滑）的空间门控，可用于跳跃连接；
    3. SkipAttentionGate：经典 Attention U-Net 的注意力门控。
当前裂缝分割配置默认仅使用 ECABlock，后两者作为可选扩展保留。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule


class ECABlock(BaseModule):
    """高效通道注意力模块（Efficient Channel Attention）。

    与 SE（Squeeze-and-Excitation）模块不同，ECA **不做通道降维**，而是在
    全局平均池化后，用一个核大小为 k 的一维卷积捕获局部跨通道交互，整个模块
    只有 k 个参数（k 通常为 3~9），开销极低。

    计算流程：
        1. 全局平均池化：(B, C, H, W) -> (B, C, 1, 1)；
        2. reshape 为 (B, 1, C)，经 1D 卷积学习通道权重；
        3. sigmoid 归一化后逐通道乘回原特征。

    Args:
        channels (int): 输入特征的通道数 C。
        gamma (int): 自适应核大小计算公式中的 gamma 参数，默认 2。
        b (int): 自适应核大小计算公式中的 b 参数，默认 1。
        init_cfg (dict or None): 权重初始化配置，默认 None。

    Note:
        自适应核大小公式为 ``k = |log2(C)/gamma + b/gamma|``，结果取奇数且不小于 3。
        通道数 64->1024 时，k 约为 {3, 5, 5, 7, 9}。
    """

    def __init__(self, channels, gamma=2, b=1, init_cfg=None):
        super().__init__(init_cfg)
        # ---- 根据通道数自适应计算一维卷积核大小 ----
        kernel_size = int(
            abs(
                (torch.log2(torch.tensor(float(channels))) / gamma +
                 b / gamma).item()))
        if kernel_size % 2 == 0:      # 卷积核需为奇数，保证对称 padding
            kernel_size += 1
        kernel_size = max(kernel_size, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """前向传播。

        Args:
            x (Tensor): 输入特征图，形状 (B, C, H, W)。

        Returns:
            Tensor: 通道注意力加权后的特征图，形状与输入一致 (B, C, H, W)。
        """
        y = self.avg_pool(x)                       # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)        # (B, 1, C)
        y = self.conv(y)                           # (B, 1, C)
        y = self.sigmoid(y)                        # 通道权重归一化到 (0,1)
        y = y.transpose(-1, -2).unsqueeze(-1)      # (B, C, 1, 1)
        return x * y.expand_as(x)                  # 逐通道缩放


class SpatialGate(BaseModule):
    """超轻量空间门控（可选，用于跳跃连接）。

    1. 将 skip、gate 特征分别沿通道维求均值 -> (B,1,H,W)，零参数；
    2. 相加后经一个 3×3 depthwise 卷积平滑（仅 9 个参数）；
    3. sigmoid 得到空间注意力图，对 skip 特征逐像素调制。

    Args:
        skip_channels (int): 跳跃特征通道数（仅用于接口说明，不参与计算）。
        gate_channels (int): 门控特征通道数（仅用于接口说明，不参与计算）。
        use_depthwise_smooth (bool): 是否使用 3×3 depthwise 卷积平滑，默认 True。
        init_cfg (dict or None): 权重初始化配置，默认 None。
    """

    def __init__(self,
                 skip_channels,
                 gate_channels,
                 use_depthwise_smooth=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.use_depthwise_smooth = use_depthwise_smooth
        if use_depthwise_smooth:
            # 单通道 depthwise 3×3，仅 9 个参数
            self.smooth = nn.Conv2d(
                1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_feat, gate_feat):
        """前向传播，返回空间加权后的跳跃特征。"""
        skip_spatial = skip_feat.mean(dim=1, keepdim=True)
        gate_spatial = gate_feat.mean(dim=1, keepdim=True)
        attention = skip_spatial + gate_spatial
        if self.use_depthwise_smooth:
            attention = self.smooth(attention)
        attention = self.sigmoid(attention)
        return skip_feat * attention


class SkipAttentionGate(BaseModule):
    """经典 Attention U-Net 的跳跃连接注意力门控（可选）。

    在 concat 之前，对编码器的跳跃特征做门控，门控信号来自解码器上采样后的
    特征，从而自适应地过滤无关/噪声区域、突出目标边界。

    Args:
        skip_channels (int): 跳跃特征通道数。
        gate_channels (int): 门控信号通道数。
        inter_channels (int or None): 中间压缩通道数，默认取 skip_channels//2。
        init_cfg (dict or None): 权重初始化配置，默认 None。
    """

    def __init__(self,
                 skip_channels,
                 gate_channels,
                 inter_channels=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        inter_channels = inter_channels or max(skip_channels // 2, 1)

        # 对跳跃特征做 1×1 通道压缩
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels))
        # 对门控信号做 1×1 通道压缩
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels))
        # 压缩为单通道注意力系数
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, skip_feat, gate_feat):
        """前向传播，返回门控后的跳跃特征（形状与 skip_feat 一致）。

        门控信号（解码器更深层特征）空间尺寸通常小于跳跃特征，这里在投影后
        将其上采样到与跳跃特征一致，再相加生成注意力系数。
        """
        w_skip = self.W_skip(skip_feat)
        w_gate = self.W_gate(gate_feat)
        if w_gate.shape[-2:] != w_skip.shape[-2:]:
            w_gate = F.interpolate(
                w_gate,
                size=w_skip.shape[-2:],
                mode='bilinear',
                align_corners=False)
        psi = self.psi(torch.relu(w_skip + w_gate))
        return skip_feat * psi
