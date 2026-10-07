# -*- coding: utf-8 -*-
"""UAV-Crack-Seg 自定义模型包。

本包基于 OpenMMLab 的 MMSegmentation 框架，提供：
- ECABlock：高效通道注意力模块（Efficient Channel Attention）；
- UNetECA：在每个编码器阶段后嵌入 ECA、并使用深度可分离卷积轻量化的 UNet 骨干。

模型注册方式：入口配置通过 ``custom_imports=['crackseg.models']`` 在运行时导入并
完成注册（``@MODELS.register_module()``）；也可手动 ``import crackseg.models``。

注意：这里**不在包导入时自动导入 models**，以便在导入 mmseg 之前先完成 mmcv
算子兼容桩的安装（见 ``crackseg.utils.ops_shim``）。
"""
__all__ = []
__version__ = '1.0.0'
