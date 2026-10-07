# -*- coding: utf-8 -*-
"""ImageNet 预训练编码器 + UNet 解码器骨干（TimmUNet）。

设计动机
--------
在 1030 张小样本上，从零训练的轻量 ECA-UNet mIoU 在 79 左右到顶，而
DeepLabV3+ 的大视野 ASPP 又不适合纤细裂缝。本骨干取两者之长：
    * **编码器**：使用 timm 提供的 **ImageNet 预训练** CNN（默认 ResNet34），
      复用成熟的边缘/纹理特征与语义先验，小样本上显著优于随机初始化；
    * **解码器**：保留对细裂缝最友好的 **逐级上采样 + 同尺度跳跃拼接** UNet
      结构，并额外上采样到 stride=1，最大限度保留裂缝的纤细几何。

输出约定（与 UNetECA 完全一致，可直接复用 improved 的解码头/辅助头）
    forward 返回列表 dec_outs（按“粗→细”排列），对默认 ResNet34：
        dec_outs[0] : stride 16，256 通道（瓶颈，头未使用）
        dec_outs[1] : stride 8， 512 通道（辅助头 in_index=1）
        dec_outs[2] : stride 4， 256 通道（辅助头 in_index=2）
        dec_outs[3] : stride 2， 128 通道（辅助头 in_index=3）
        dec_outs[4] : stride 1，  64 通道（主解码头 in_index=4）

合规
    仅使用 timm/ImageNet 官方公开的图像分类预训练权重，不使用任何裂缝相关
    任务权重；来源在 README 致谢中注明。比赛技术要求未限制预训练编码器。
"""
import os
# 本机网络无法访问 HuggingFace，禁用 HF Hub，让 timm 回退到 pretrained_cfg 中的
# 官方权重 URL（如 download.pytorch.org），避免长时间重试超时。
os.environ.setdefault('HF_HUB_DISABLE', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmseg.registry import MODELS


class DecoderBlock(nn.Module):
    """上采样 → 与编码器同尺度特征拼接 → 两层 3×3 卷积。"""

    def __init__(self, in_channels, skip_channels, out_channels, norm_cfg):
        super().__init__()
        self.block = nn.Sequential(
            ConvModule(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


@MODELS.register_module()
class TimmUNet(BaseModule):
    """timm 预训练编码器 + UNet 解码器骨干。

    Args:
        encoder_name (str): timm 模型名，默认 ``resnet34``。
        pretrained (bool): 是否加载 ImageNet 预训练权重，默认 True。
        in_channels (int): 输入通道，默认 3。
        dec_channels (tuple): 解码器各阶段（stride 16/8/4/2）输出通道，
            需与解码头 in_channels 约定一致。
        norm_cfg (dict): 归一化配置。
        norm_eval (bool): 是否冻结编码器 BN 的统计，默认 False。
        init_cfg: 兼容基类，预训练权重由 timm 在构建时加载。
    """

    def __init__(self,
                 encoder_name='resnet34',
                 pretrained=True,
                 in_channels=3,
                 dec_channels=(256, 512, 256, 128),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        import timm

        # 1) 先构建多尺度特征提取器（pretrained=False，不触发任何下载）。
        #    FeatureListNet 直接复用骨干各层（conv1/layer1...），命名与完整模型一致。
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=in_channels)

        # 2) 手动从官方权重 URL 下载并载入，彻底绕开 timm 的 HuggingFace
        #    探测逻辑（本机无法访问 HF，会反复 HEAD 超时）。骨干键全部命中，
        #    仅分类头权重（fc.*）被忽略。
        if pretrained:
            pcfg = timm.get_pretrained_cfg(encoder_name)
            assert pcfg is not None and pcfg.url, \
                f'未找到 {encoder_name} 的官方权重 URL'
            state_dict = torch.hub.load_state_dict_from_url(
                pcfg.url, map_location='cpu')
            missing, unexpected = self.encoder.load_state_dict(
                state_dict, strict=False)
            assert not missing, f'预训练权重缺失键: {missing[:5]}'

        enc_ch = list(self.encoder.feature_info.channels())
        # enc_ch(resnet34) = [64, 64, 128, 256, 512]
        c16, c8, c4, c2 = dec_channels

        # 解码器：最深 f4(enc_ch[4]) 逐级向上，与 f3/f2/f1/f0 拼接
        self.dec16 = DecoderBlock(enc_ch[4], enc_ch[3], c16, norm_cfg)
        self.dec8 = DecoderBlock(c16, enc_ch[2], c8, norm_cfg)
        self.dec4 = DecoderBlock(c8, enc_ch[1], c4, norm_cfg)
        self.dec2 = DecoderBlock(c4, enc_ch[0], c2, norm_cfg)
        # 再上采样到 stride=1，输出 64 通道供主解码头
        self.dec1 = nn.Sequential(
            ConvModule(
                c2, 64, kernel_size=3, padding=1,
                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')),
            ConvModule(
                64, 64, kernel_size=3, padding=1,
                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')),
        )
        self.norm_eval = norm_eval

    def forward(self, x):
        input_size = x.shape[-2:]
        f0, f1, f2, f3, f4 = self.encoder(x)
        d16 = self.dec16(f4, f3)
        d8 = self.dec8(d16, f2)
        d4 = self.dec4(d8, f1)
        d2 = self.dec2(d4, f0)
        d1_up = F.interpolate(
            d2, size=input_size, mode='bilinear', align_corners=False)
        d1 = self.dec1(d1_up)
        # 粗 → 细
        return [d16, d8, d4, d2, d1]

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
