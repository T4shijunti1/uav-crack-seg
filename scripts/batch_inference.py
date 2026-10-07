# -*- coding: utf-8 -*-
"""批量推理脚本：对一个文件夹的图片做裂缝分割。

输出三类结果到 out_dir：
    - masks/        ：二值掩码 PNG（裂缝=255，背景=0）；
    - overlay/      ：在原图上用红色半透明标出裂缝的叠加图，便于人工核对；
    - results.json ：每张图的结构化摘要（图像 ID、裂缝像素数/占比、置信度）。

实现说明：
    不使用 mmseg 的 ``inference_model`` 高层 API（在本项目滑窗配置下会返回
    退化的单行 logits），而是采用与 ``tools/test.py``、``threshold_sweep.py``
    一致、且已验证正确的路径：dataset 取样本 -> SegDataPreProcessor 预处理 ->
    model.inference（滑动窗口）-> softmax 概率 -> 阈值判决。

注意：results.json 为便于核对的本地摘要；正式参赛提交的“测试集推理结果文件”
请以 VLP 平台提供的评估工具包所要求的字段格式为准（可用本脚本结果转换）。

用法：
    python scripts/batch_inference.py \
        --config configs/crack_seg_improved.py \
        --checkpoint work_dirs/improved/best_mIoU_iter_24000.pth \
        --input data/Crack/images/test \
        --out-dir work_dirs/improved/predictions
"""
import argparse
import json
import os
import os.path as osp
import sys

# 把仓库根目录加入 sys.path，使 crackseg 包可被导入
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from crackseg.utils import (  # noqa: E402
    install_mmcv_ops_shim,
    install_trusted_checkpoint_loader,
)
install_mmcv_ops_shim()
install_trusted_checkpoint_loader()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from mmseg.registry import DATASETS, MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402

# 支持的图片后缀
IMG_SUFFIX = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='裂缝分割批量推理')
    parser.add_argument('--config', required=True, help='模型配置文件')
    parser.add_argument('--checkpoint', required=True, help='模型检查点')
    parser.add_argument('--input', required=True, help='输入图片目录')
    parser.add_argument('--out-dir', required=True, help='结果输出目录')
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='裂缝概率阈值，默认 0.5（=softmax argmax）')
    parser.add_argument(
        '--device', default='cuda:0', help='推理设备，如 cuda:0 或 cpu')
    return parser.parse_args()


def load_trusted_checkpoint(model, ckpt_path):
    """加载本项目自己训练的检查点（含 mmengine 元信息，需 weights_only=False）。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)


def make_overlay(image_bgr, mask):
    """生成红色半透明叠加图。

    Args:
        image_bgr (ndarray): 原图（BGR）。
        mask (ndarray): 二值掩码（裂缝=1）。

    Returns:
        ndarray: BGR 叠加图。
    """
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)   # BGR 中红色为 (0,0,255)
    return cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)


def main():
    """主函数：构建模型/数据集 -> 逐张滑窗推理 -> 保存掩码/叠加图 -> 汇总 JSON。"""
    args = parse_args()

    # 无 GPU 时回退到 CPU
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        print('未检测到 CUDA，回退使用 CPU。')
        args.device = 'cpu'

    register_all_modules()
    cfg = Config.fromfile(args.config)
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg['custom_imports'])

    # ---- 构建模型并载入权重 ----
    model = MODELS.build(cfg.model)
    load_trusted_checkpoint(model, args.checkpoint)
    model.cfg = cfg
    model.to(args.device).eval()

    # ---- 构建仅含图像（无标注）的数据集，复用配置中的类别信息 ----
    dataset_cfg = dict(
        type='BaseSegDataset',
        data_root=args.input,
        data_prefix=dict(img_path=''),
        metainfo=dict(classes=('background', 'crack')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='PackSegInputs'),
        ])
    dataset = DATASETS.build(dataset_cfg)
    n = len(dataset)
    print('共发现 %d 张图片，开始推理……' % n)

    # 准备输出目录
    mask_dir = osp.join(args.out_dir, 'masks')
    overlay_dir = osp.join(args.out_dir, 'overlays')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    records = []
    for idx in range(n):
        item = dataset[idx]
        fn = item['data_samples'].img_path
        data_batch = {
            'inputs': [item['inputs']],
            'data_samples': [item['data_samples']],
        }
        pre = model.data_preprocessor(data_batch, training=False)
        inputs, data_samples = pre['inputs'], pre['data_samples']
        if not isinstance(data_samples, list):
            data_samples = [data_samples]
        with torch.no_grad():
            batch_img_metas = [ds.metainfo for ds in data_samples]
            seg_logits = model.inference(inputs, batch_img_metas)
        crack_prob = seg_logits.softmax(dim=1)[0, 1]
        crack_prob = crack_prob.detach().cpu().numpy().astype(np.float32)
        pred = (crack_prob >= args.threshold).astype(np.uint8)

        image_bgr = cv2.imread(fn)
        h, w = image_bgr.shape[:2]
        # 对齐到原图尺寸（正常情况下滑窗已与原图等大）
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            crack_prob = cv2.resize(
                crack_prob, (w, h), interpolation=cv2.INTER_LINEAR)

        base = osp.splitext(osp.basename(fn))[0]
        cv2.imwrite(osp.join(mask_dir, base + '.png'), pred * 255)
        cv2.imwrite(
            osp.join(overlay_dir, base + '_overlay.jpg'),
            make_overlay(image_bgr, pred))

        # ---- 结构化记录（置信度仅在预测为裂缝的像素上统计）----
        crack_pixels = int(pred.sum())
        mean_conf = (
            float(crack_prob[pred > 0].mean()) if crack_pixels > 0 else 0.0)
        records.append(
            dict(
                image_id=osp.basename(fn),
                height=int(h),
                width=int(w),
                crack_pixels=crack_pixels,
                crack_ratio=round(float(pred.mean()), 6),
                mean_crack_confidence=round(mean_conf, 6)))

        if (idx + 1) % 20 == 0 or idx + 1 == n:
            print('已处理 %d/%d' % (idx + 1, n))

    # ---- 写出 JSON 摘要 ----
    json_path = osp.join(args.out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            dict(
                task='UAV-Crack',
                threshold=args.threshold,
                num_images=len(records),
                results=records),
            f,
            ensure_ascii=False,
            indent=2)

    print('完成。掩码：%s；叠加图：%s；摘要：%s' %
          (mask_dir, overlay_dir, json_path))


if __name__ == '__main__':
    main()
