# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""将训练好的裂缝分割模型导出为 ONNX 格式。

比赛《技术要求》规定：PyTorch（.pth）模型需**额外提供 ONNX 格式转换文件**，
本脚本即用于该目的，导出后可随权重一并通过网盘共享。

导出的 ONNX 模型约定：
    - 输入名 "input"：形状 (B, 3, H, W)，RGB、已按 ImageNet 均值/方差归一化；
    - 输出名 "seg_logits"：形状 (B, 2, H, W)，为两个类别的 logits，
      对其 argmax(dim=1) 即得 {0,1} 标签（1 = 裂缝）；
    - batch 维为动态轴，可一次推理任意张数（图像尺寸默认固定 256）。

用法：
    python tools/export_onnx.py \
        --config configs/crack_seg_256x256.py \
        --checkpoint work_dirs/crack/best_mIoU_iter_40000.pth \
        --output work_dirs/crack/crack_seg.onnx

若已安装 onnxruntime，脚本会自动做一次 ONNX 与 PyTorch 的数值一致性校验。
"""
import argparse
import os.path as osp
import sys

import numpy as np
import torch

# 把仓库根目录加入 sys.path，使 crackseg 包可被导入
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

# 在导入 mmseg 之前安装 mmcv 算子兼容桩（完整 mmcv 环境下为空操作）
from crackseg.utils import (  # noqa: E402
    install_mmcv_ops_shim,
    install_trusted_checkpoint_loader,
)
install_mmcv_ops_shim()
install_trusted_checkpoint_loader()

from mmengine.model import revert_sync_batchnorm  # noqa: E402
from mmseg.apis import init_model  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402


class SegLogitsWrapper(torch.nn.Module):
    """ONNX 导出包装器：只保留 backbone + 主解码头，直接输出分割 logits。

    MMSeg 的 EncoderDecoder.forward 为配合数据样本（data samples）设计，
    对 ONNX 导出不友好；这里直接串联 backbone 与 decode_head 的纯张量计算，
    且不包含推理后处理（argmax），把后处理留给调用方，便于跨平台部署。
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.decode_head = model.decode_head

    def forward(self, x):
        """前向：x 为已归一化图像张量，返回分割 logits。"""
        feats = self.backbone(x)
        seg_logits = self.decode_head(feats)
        return seg_logits


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='导出裂缝分割模型为 ONNX')
    parser.add_argument('--config', required=True, help='模型配置文件')
    parser.add_argument('--checkpoint', required=True, help='模型检查点 (.pth)')
    parser.add_argument('--output', default='crack_seg.onnx', help='输出 ONNX 路径')
    parser.add_argument('--size', type=int, default=256, help='输入图像边长')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset 版本')
    return parser.parse_args()


def main():
    """主函数：加载模型 -> 包装 -> 导出 ONNX -> （可选）数值校验。"""
    args = parse_args()
    register_all_modules()

    # 导出在 CPU 上进行，兼容性最好；并把 SyncBN 还原为普通 BN
    model = init_model(args.config, args.checkpoint, device='cpu')
    model = revert_sync_batchnorm(model)
    model.eval()

    wrapper = SegLogitsWrapper(model).eval()

    # 构造一个随机输入用于导出（实际部署需按同样方式归一化）
    dummy_input = torch.randn(1, 3, args.size, args.size)

    torch.onnx.export(
        wrapper,
        dummy_input,
        args.output,
        input_names=['input'],
        output_names=['seg_logits'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # 使用成熟的 legacy(TorchScript) 导出器，免装 onnxscript
        dynamic_axes={
            'input': {0: 'batch'},
            'seg_logits': {0: 'batch'},
        })
    print('ONNX 已导出 -> %s' % args.output)

    # ---- 可选：onnxruntime 数值一致性校验 ----
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            args.output, providers=['CPUExecutionProvider'])
        x = np.random.randn(1, 3, args.size, args.size).astype('float32')
        onnx_out = sess.run(None, {'input': x})[0]
        with torch.no_grad():
            torch_out = wrapper(torch.from_numpy(x)).numpy()
        max_diff = float(np.abs(onnx_out - torch_out).max())
        print('ONNX 输出形状 %s，与 PyTorch 最大绝对误差 %.2e' %
              (onnx_out.shape, max_diff))
    except Exception as error:  # 未安装 onnxruntime 时跳过
        print('跳过 ONNX 数值校验（可执行 pip install onnxruntime）：%s' % error)


if __name__ == '__main__':
    main()
