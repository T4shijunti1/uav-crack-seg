# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""带 ECA 通道注意力的轻量级 UNet 骨干（UNetECA）。

本实现改编自 MMSegmentation 官方 UNet 骨干，并在其基础上做了两点增强：
    1. **编码器阶段 ECA**：在每个编码器阶段（下采样 + 卷积）之后立即插入一个
       ECABlock，对特征通道进行重标定——增强与裂缝相关的通道、抑制噪声通道；
    2. **深度可分离卷积轻量化（L-Block）**：每个卷积块的第一个 3×3 卷积可替换为
       「Depthwise 3×3 + Pointwise 1×1」，显著降低参数量。

输出约定（与官方 UNet 一致，供解码头按索引取特征）：
    forward 返回一个列表 dec_outs：
        dec_outs[0]            : 瓶颈层（编码器最深层）输出；
        dec_outs[1..S-1]       : 解码器逐级上采样输出；
        dec_outs[-1]（索引 4）: 解码器最终输出，通道数 = base_channels。
    因此解码头 ``in_index=4, in_channels=64``，辅助头 ``in_index=3, in_channels=128``。
"""
import warnings
from functools import partial

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmseg.models.utils import UpConvBlock
from mmseg.registry import MODELS

from .eca_block import ECABlock, SkipAttentionGate


class BasicConvBlock(nn.Module):
    """UNet 编/解码器通用卷积块，默认包含两层卷积。

    第一层为 stride 卷积（stride 由外部传入），后续层 stride=1。
    当 ``use_dw=True`` 时，**第一个 3×3 卷积**替换为深度可分离卷积
    （Depthwise 3×3 + Pointwise 1×1），第二个 3×3 卷积保持标准卷积以保证
    边界表达能力——与论文 L-Block 的设计一致。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        num_convs (int): 卷积层数，默认 2。
        stride (int): 第一层卷积的步长（1 或 2），默认 1。
        dilation (int): 卷积膨胀率，默认 1。
        with_cp (bool): 是否使用梯度检查点节省显存（会变慢），默认 False。
        conv_cfg (dict or None): 卷积层配置，默认 None。
        norm_cfg (dict or None): 归一化层配置，默认 BN。
        act_cfg (dict or None): 激活层配置，默认 ReLU。
        dcn: 可变形卷积占位，本实现不支持，必须为 None。
        plugins: 插件占位，本实现不支持，必须为 None。
        use_dw (bool): 第一层是否使用深度可分离卷积，默认 False。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None,
                 use_dw=False):
        super().__init__()
        assert dcn is None, '可变形卷积尚未实现。'
        assert plugins is None, '插件尚未实现。'
        self.with_cp = with_cp

        layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else out_channels
            cur_stride = stride if i == 0 else 1
            # 第一层膨胀率固定为 1，后续层使用给定膨胀率
            cur_dilation = 1 if i == 0 else dilation
            padding = cur_dilation

            # 仅对每个卷积块的“第一个卷积”启用深度可分离卷积
            if use_dw and i == 0:
                block = nn.Sequential(
                    # Depthwise：每个输入通道各自做 3×3 卷积（groups=c_in）
                    ConvModule(
                        c_in,
                        c_in,
                        kernel_size=3,
                        stride=cur_stride,
                        dilation=cur_dilation,
                        padding=padding,
                        groups=c_in,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    # Pointwise：1×1 卷积做跨通道信息融合
                    ConvModule(
                        c_in,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                block = ConvModule(
                    c_in,
                    out_channels,
                    kernel_size=3,
                    stride=cur_stride,
                    dilation=cur_dilation,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            layers.append(block)

        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播；开启梯度检查点时用 checkpoint 包裹以节省显存。"""
        if self.with_cp and x.requires_grad:
            return cp.checkpoint(self.convs, x)
        return self.convs(x)


@MODELS.register_module()
class UNetECA(BaseModule):
    """带 ECA 通道注意力的轻量级 UNet 骨干。

    典型结构（num_stages=5, base_channels=64）：
        编码器各阶段输出通道数：64, 128, 256, 512, 1024；
        每个编码器阶段 = [可选 MaxPool 下采样] -> BasicConvBlock -> [可选 ECA]；
        解码器共 4 个 UpConvBlock，逐级融合跳跃连接并上采样。

    Args:
        in_channels (int): 输入图像通道数，RGB 图像为 3，默认 3。
        base_channels (int): 第一阶段输出通道数，默认 64。
        num_stages (int): 编码器阶段数，默认 5。
        strides (Sequence[int]): 各编码器阶段步长，默认全 1（用 MaxPool 下采样）。
        enc_num_convs (Sequence[int]): 各编码器阶段卷积层数。
        dec_num_convs (Sequence[int]): 各解码器阶段卷积层数。
        downsamples (Sequence[bool]): 各编码器阶段后是否用 MaxPool 下采样。
        enc_dilations (Sequence[int]): 各编码器阶段膨胀率。
        dec_dilations (Sequence[int]): 各解码器阶段膨胀率。
        enc_use_eca (bool): 是否在每个编码器阶段后插入 ECA，默认 True。
        use_dw (bool): 是否启用深度可分离卷积轻量化，默认 True。
        with_cp (bool): 是否使用梯度检查点，默认 False。
        conv_cfg (dict or None): 卷积配置，默认 None。
        norm_cfg (dict): 归一化配置，默认 BN。
        act_cfg (dict): 激活配置，默认 ReLU。
        upsample_cfg (dict): 解码器上采样配置，默认 InterpConv。
        norm_eval (bool): 是否冻结归一化层统计量，默认 False。
        pretrained (str or None): 预训练权重路径（已废弃，建议用 init_cfg）。
        init_cfg (dict or None): 初始化配置，默认 None。
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 enc_use_eca=True,
                 use_skip_gate=False,
                 use_dw=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg 与 pretrained 不能同时设置。'

        # ---- 权重初始化配置 ----
        if isinstance(pretrained, str):
            warnings.warn('pretrained 已废弃，请改用 init_cfg。', stacklevel=2)
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None and init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm']),
            ]
        elif pretrained is not None:
            raise TypeError('pretrained 必须为字符串或 None。')

        # ---- 参数合法性校验 ----
        assert dcn is None and plugins is None, 'dcn/plugins 尚未实现。'
        assert len(strides) == num_stages
        assert len(enc_num_convs) == num_stages
        assert len(dec_num_convs) == (num_stages - 1)
        assert len(downsamples) == (num_stages - 1)
        assert len(enc_dilations) == num_stages
        assert len(dec_dilations) == (num_stages - 1)

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # 用 partial 绑定 use_dw，使解码器的 UpConvBlock 在内部构建卷积块时
        # 自动带上轻量化开关（UpConvBlock 本身不认识 use_dw 参数）。
        conv_block = partial(BasicConvBlock, use_dw=use_dw)

        cur_in = in_channels
        for i in range(num_stages):
            stage_layers = []

            if i != 0:
                # ---- 解码器（先构建，与官方顺序保持一致）----
                if strides[i] == 1 and downsamples[i - 1]:
                    stage_layers.append(nn.MaxPool2d(kernel_size=2))
                use_upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=conv_block,
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2**(i - 1),
                        out_channels=base_channels * 2**(i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if use_upsample else None,
                        dcn=None,
                        plugins=None))

            # ---- 编码器卷积块 ----
            stage_layers.append(
                BasicConvBlock(
                    in_channels=cur_in,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None,
                    use_dw=use_dw))

            # ---- 编码器阶段后插入 ECA（仅插入一次）----
            if enc_use_eca:
                stage_layers.append(ECABlock(base_channels * 2**i))

            self.encoder.append(nn.Sequential(*stage_layers))
            cur_in = base_channels * 2**i  # 下一阶段输入通道数

        # ---- 解码器跳跃连接注意力门（Attention U-Net 风格，可选）----
        # 第 i 个解码器融合 skip=base*2^i（编码器）与 gate=base*2^(i+1)（更深层）。
        self.use_skip_gate = use_skip_gate
        if use_skip_gate:
            self.skip_gates = nn.ModuleList([
                SkipAttentionGate(
                    skip_channels=base_channels * 2**i,
                    gate_channels=base_channels * 2**(i + 1))
                for i in range(num_stages - 1)
            ])

    def forward(self, x):
        """前向传播，返回多尺度特征列表 dec_outs（见模块说明）。"""
        self._check_input_divisible(x)

        enc_outs = []
        for enc_block in self.encoder:
            x = enc_block(x)
            enc_outs.append(x)

        dec_outs = [x]                 # 索引 0：瓶颈层输出
        for i in reversed(range(len(self.decoder))):
            skip = enc_outs[i]
            # 注意力门控：用更深层解码器特征门控编码器跳跃特征，抑制无关区域
            if self.use_skip_gate:
                skip = self.skip_gates[i](skip, x)
            x = self.decoder[i](skip, x)
            dec_outs.append(x)
        return dec_outs

    def train(self, mode=True):
        """设置训练/评估模式；norm_eval=True 时冻结 BN 统计量。"""
        super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()

    def _check_input_divisible(self, x):
        """校验输入尺寸能被整体下采样倍率整除，避免上采样尺寸错位。"""
        h, w = x.shape[-2:]
        down_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                down_rate *= 2
        assert h % down_rate == 0 and w % down_rate == 0, (
            f'输入尺寸 {(h, w)} 需能被整体下采样倍率 {down_rate} 整除。')
