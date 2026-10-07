# -*- coding: utf-8 -*-
"""裂缝判决阈值扫描（无需重新训练）。

背景：
    二分类分割默认对 softmax 结果取 argmax，等价于以 0.5 为阈值判定裂缝。
    当模型偏保守（精确率高、召回率低）时，下调阈值可提升召回，进而提高
    F1 与 Crack IoU。本脚本在验证集上只做一次滑窗推理，随后批量计算多个
    阈值下赛道 D 的 6 项指标，给出最优阈值。

用法（在仓库根目录）：
    python scripts/threshold_sweep.py configs/crack_seg_256x256.py \
        work_dirs/crack/best_mIoU_iter_40000.pth

输出：
    打印每个阈值的 F1 / mIoU / Crack IoU / Acc / Pr / Re，并把结果写入
    work_dirs/crack/threshold_sweep.json。
"""
import json
import os.path as osp
import sys

import numpy as np

# 把仓库根目录加入 sys.path
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from crackseg.utils import install_mmcv_ops_shim  # noqa: E402

install_mmcv_ops_shim()

from mmengine.config import Config  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from mmseg.registry import DATASETS, MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402
import torch  # noqa: E402


def load_trusted_checkpoint(model, ckpt_path):
    """加载本项目自己训练的检查点（含 mmengine 元信息，需 weights_only=False）。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

# 待扫描的裂缝概率阈值（argmax 对应 0.5）
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
IGNORE_INDEX = 255


def compute_metrics(counts):
    """根据聚合的混淆计数计算 6 项指标（百分数）。"""
    tp, fp, fn, tn = counts
    crack_iou = tp / (tp + fp + fn + 1e-12)
    bg_iou = tn / (tn + fp + fn + 1e-12)
    miou = 0.5 * (crack_iou + bg_iou)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
    pr = tp / (tp + fp + 1e-12)
    re = tp / (tp + fn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {
        'F1': 100 * f1,
        'mIoU': 100 * miou,
        'Crack IoU': 100 * crack_iou,
        'Acc': 100 * acc,
        'Pr': 100 * pr,
        'Re': 100 * re,
    }


def main():
    if len(sys.argv) < 3:
        print('用法: python scripts/threshold_sweep.py <config> <checkpoint>')
        sys.exit(1)
    config_path, ckpt_path = sys.argv[1], sys.argv[2]

    register_all_modules()
    cfg = Config.fromfile(config_path)
    # 注册自定义模型（crackseg.models）
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg['custom_imports'])

    # 构建模型并载入权重
    model = MODELS.build(cfg.model)
    load_trusted_checkpoint(model, ckpt_path)
    model.cfg = cfg
    model.to('cuda').eval()

    # 构建验证数据集（直接索引，避免多进程）
    dataset = DATASETS.build(cfg.val_dataloader['dataset'])
    n = len(dataset)
    print(f'验证集样本数: {n}')

    # 每个阈值一组 [tp, fp, fn, tn] 计数（int64）
    counts = {t: np.zeros(4, dtype=np.int64) for t in THRESHOLDS}

    for i in range(n):
        item = dataset[i]
        data_batch = {
            'inputs': [item['inputs']],
            'data_samples': [item['data_samples']],
        }
        # 预处理器返回 dict(inputs=..., data_samples=...)，按键访问
        pre = model.data_preprocessor(data_batch, training=False)
        inputs, data_samples = pre['inputs'], pre['data_samples']
        # batch=1 时预处理器可能返回单个 SegDataSample，统一成列表
        if not isinstance(data_samples, list):
            data_samples = [data_samples]
        with torch.no_grad():
            # inference 需要的是每张图的 metainfo 字典列表，而非 data_samples
            batch_img_metas = [ds.metainfo for ds in data_samples]
            seg_logits = model.inference(inputs, batch_img_metas)
        prob = seg_logits.softmax(dim=1)[0, 1].detach().cpu().numpy()
        gt = data_samples[0].gt_sem_seg.data[0].cpu().numpy()

        valid = gt != IGNORE_INDEX
        g = gt[valid].astype(bool)          # True=裂缝
        p = prob[valid]

        for t in THRESHOLDS:
            pred = p >= t
            tp = int(np.count_nonzero(pred & g))
            fp = int(np.count_nonzero(pred & ~g))
            fn = int(np.count_nonzero(~pred & g))
            tn = int(np.count_nonzero(~pred & ~g))
            counts[t] += np.array([tp, fp, fn, tn], dtype=np.int64)

        if (i + 1) % 40 == 0 or i + 1 == n:
            print(f'  已完成 {i + 1}/{n}')

    # 汇总
    results = {}
    header = ['阈值', 'F1', 'mIoU', 'Crack IoU', 'Acc', 'Pr', 'Re']
    print('\n' + '  '.join(f'{h:>9}' for h in header))
    best_t, best_f1 = None, -1.0
    for t in THRESHOLDS:
        m = compute_metrics(counts[t])
        results[f'{t:.2f}'] = {k: round(v, 2) for k, v in m.items()}
        row = [f'{t:.2f}'] + [f'{m[k]:.2f}' for k in header[1:]]
        print('  '.join(f'{c:>9}' for c in row))
        if m['F1'] > best_f1:
            best_f1, best_t = m['F1'], t

    print(f"\n按 F1 最优阈值 = {best_t:.2f}（F1={best_f1:.2f}）")
    results['_best_by_F1'] = best_t

    out_json = osp.join(osp.dirname(ckpt_path), 'threshold_sweep.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'结果已保存: {out_json}')


if __name__ == '__main__':
    main()
