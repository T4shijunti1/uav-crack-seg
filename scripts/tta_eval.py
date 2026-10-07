# -*- coding: utf-8 -*-
"""测试时增强（TTA）评估脚本：在不重训练、不改动权重的前提下提升分割精度。

做法：对每张图生成多种“放大尺度 × 翻转”的视图，分别做滑窗推理，
再把各视图的 softmax 概率还原到原始尺寸后平均（概率级集成）。
概率平均通常单调地不降低期望精度，因此可在“保持其他指标不下降”的前提下
争取 mIoU 提升。裂缝无固定朝向，故水平、垂直翻转均为合法增强；
为避免滑窗窗口（高 368）大于放大后图像，这里只使用 >=1.0 的放大尺度。

用法（仓库根目录）：
    python scripts/tta_eval.py \
        --config configs/crack_seg_improved.py \
        --checkpoint work_dirs/improved/best_mIoU_iter_24000.pth \
        --scales 1.0,1.25 --flips none,h,v
"""
import argparse
import copy
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

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.structures import BaseDataElement  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from mmseg.models import SegTTAModel  # noqa: E402
from mmseg.registry import DATASETS, METRICS, MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402

# 版本兼容：mmengine 0.10.7 调整了数据元素的取数方式，而 mmseg 1.2.2 的
# IoUMetric 仍以 data_sample['pred_sem_seg']['data'] 方式取字段，补回字符串字段访问。
if not hasattr(BaseDataElement, '__getitem__'):
    BaseDataElement.__getitem__ = lambda self, key: getattr(self, key)

from mmengine.structures import PixelData  # noqa: E402
_orig_pix_getitem = PixelData.__getitem__


def _pix_getitem(self, key):
    # 字符串按字段名访问；int/slice 保留原有的张量切片语义
    if isinstance(key, str):
        return getattr(self, key)
    return _orig_pix_getitem(self, key)


PixelData.__getitem__ = _pix_getitem


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='TTA 评估')
    parser.add_argument('--config', required=True, help='模型配置文件')
    parser.add_argument('--checkpoint', required=True, help='模型检查点')
    parser.add_argument(
        '--scales', default='1.0,1.25',
        help='逗号分隔的放大尺度，仅允许 >=1.0，如 1.0,1.25,1.5')
    parser.add_argument(
        '--flips', default='none,h,v',
        help='逗号分隔的翻转方式：none/h/v，如 none,h,v')
    parser.add_argument(
        '--device', default='cuda:0', help='推理设备，如 cuda:0 或 cpu')
    return parser.parse_args()


def build_tta_pipeline(scales, flips):
    """构造 TestTimeAug 管线（放大尺度 × 翻转 的笛卡尔积）。

    Args:
        scales (list[float]): 放大尺度列表。
        flips (list[str]): 翻转方式列表，none/h/v。

    Returns:
        list: 可直接赋给数据集的 pipeline。
    """
    # 尺度子流程：按比例缩放（图像与标签同步）
    resize_choices = [
        dict(type='Resize', scale_factor=float(s), keep_ratio=True)
        for s in scales
    ]
    # 翻转子流程：不翻转 / 水平 / 垂直
    flip_choices = []
    for f in flips:
        f = f.strip().lower()
        if f == 'none':
            flip_choices.append(
                dict(type='RandomFlip', prob=0., direction='horizontal'))
        elif f == 'h':
            flip_choices.append(
                dict(type='RandomFlip', prob=1., direction='horizontal'))
        elif f == 'v':
            flip_choices.append(
                dict(type='RandomFlip', prob=1., direction='vertical'))
        else:
            raise ValueError('不支持的翻转方式：%s' % f)

    return [
        dict(type='LoadImageFromFile'),
        dict(
            type='TestTimeAug',
            transforms=[
                resize_choices,
                flip_choices,
                [dict(type='LoadAnnotations')],
                [dict(type='PackSegInputs')],
            ]),
    ]


def main():
    """主函数：构建内层模型并载入权重 -> TTA 包装 -> 逐图评估 -> 输出指标。"""
    args = parse_args()
    scales = [s for s in args.scales.split(',') if s.strip()]
    assert all(float(s) >= 1.0 for s in scales), '仅允许放大尺度（>=1.0）'
    flips = [f for f in args.flips.split(',') if f.strip()]

    # 无 GPU 时回退 CPU
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        print('未检测到 CUDA，回退使用 CPU。')
        args.device = 'cpu'

    register_all_modules()
    cfg = Config.fromfile(args.config)
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg['custom_imports'])

    # ---- 构建内层分割模型并直接载入权重（避免 TTA 包装导致的键名前缀问题）----
    base_model = MODELS.build(cfg.model)
    ckpt = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    base_model.load_state_dict(state_dict, strict=True)
    base_model.cfg = cfg
    base_model.to(args.device).eval()

    # ---- TTA 包装 ----
    tta_model = SegTTAModel(module=base_model).to(args.device).eval()

    # ---- 构建验证数据集（替换为 TTA 管线）----
    dataset_cfg = copy.deepcopy(cfg.val_dataloader.dataset)
    dataset_cfg.pipeline = build_tta_pipeline(scales, flips)
    dataset = DATASETS.build(dataset_cfg)
    print('共 %d 张验证图；每图 %d 个视图（尺度 %s × 翻转 %s）' %
          (len(dataset), len(scales) * len(flips), scales, flips))

    # ---- 指标 ----
    metric = METRICS.build(cfg.val_evaluator)
    metric.dataset_meta = dataset.metainfo

    for i in range(len(dataset)):
        item = dataset[i]
        # 单样本伪批处理：得到 outer=增强数 的多批数据格式
        data_batch = pseudo_collate([item])
        with torch.no_grad():
            preds = tta_model.test_step(data_batch)
        metric.process(data_batch, preds)
        if (i + 1) % 20 == 0 or i + 1 == len(dataset):
            print('已评估 %d/%d' % (i + 1, len(dataset)))

    metrics = metric.evaluate(size=len(dataset))
    print('\n===== TTA 评估结果 =====')
    for k, v in metrics.items():
        print('%-12s: %.4f' % (k, v))


if __name__ == '__main__':
    main()
