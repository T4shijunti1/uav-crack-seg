# -*- coding: utf-8 -*-
"""多模型集成（Ensemble）评估脚本：在不改动任何权重的前提下提升分割精度。

原理：
    不同改进路径训练出的模型会犯“不完全相同”的错误。对同一张图，把各模型
    滑窗推理得到的 softmax 概率做**算术平均**，再 argmax（或按阈值判决），
    可让彼此一致的正确预测被保留、彼此不一致的偶然错误被相互抵消。概率级
    平均在期望意义上单调不降低精度，因此符合“保持其他指标不下降、争取
    mIoU 提升”的目标。

工程实现（避免与正在进行的训练争抢显存）：
    逐个模型依次完成**整库推理**，把每张图的“裂缝概率图”转成 numpy 存到
    CPU 内存（170 张约 170MB/模型），随即把该模型从显存释放；全部模型跑完
    后再在 CPU 上做概率平均与指标统计。这样任一时刻显存里最多只有一个额外
    的推理模型，可与后台训练安全并行。

用法（仓库根目录，--pair 可重复，每次给“配置 + 检查点”两个路径）：
    python scripts/ensemble_eval.py \
        --pair configs/crack_seg_improved.py work_dirs/improved/best_mIoU_iter_24000.pth \
        --pair configs/crack_seg_v3.py work_dirs/v3/best_mIoU_iter_24000.pth

输出：
    打印多个判决阈值下赛道 D 的 6 项指标（F1 / mIoU / Crack IoU / Acc / Pr / Re），
    并把结果写入 work_dirs/ensemble/ensemble_eval.json。
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

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
from mmengine.utils import import_modules_from_strings  # noqa: E402
from mmseg.registry import DATASETS, MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402

IGNORE_INDEX = 255
# 集成后待评估的裂缝判决阈值（argmax 对应 0.5）
THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55]


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='多模型集成评估')
    parser.add_argument(
        '--pair', nargs=2, action='append', metavar=('CONFIG', 'CHECKPOINT'),
        required=True,
        help='一组“配置路径 检查点路径”，可重复以加入多个模型')
    parser.add_argument(
        '--device', default='cuda:0', help='推理设备，如 cuda:0 或 cpu')
    return parser.parse_args()


def load_trusted_checkpoint(model, ckpt_path):
    """加载本项目自己训练的检查点（含 mmengine 元信息，需 weights_only=False）。"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)


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


def collect_probs(config_path, ckpt_path, dataset, device):
    """用单个模型完成整库推理，返回每张图的（裂缝概率, 有效掩膜, 真值）。

    Args:
        config_path (str): 该模型配置文件。
        ckpt_path (str): 该模型检查点。
        dataset: 已构建的验证数据集（各模型共用同一验证划分）。
        device (str): 推理设备。

    Returns:
        list[tuple[np.ndarray, np.ndarray, np.ndarray]]: 每图三项，均为 CPU numpy。
    """
    cfg = Config.fromfile(config_path)
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg['custom_imports'])

    model = MODELS.build(cfg.model)
    load_trusted_checkpoint(model, ckpt_path)
    model.cfg = cfg
    model.to(device).eval()

    out = []
    n = len(dataset)
    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            data_batch = {
                'inputs': [item['inputs']],
                'data_samples': [item['data_samples']],
            }
            pre = model.data_preprocessor(data_batch, training=False)
            inputs, data_samples = pre['inputs'], pre['data_samples']
            if not isinstance(data_samples, list):
                data_samples = [data_samples]
            batch_img_metas = [ds.metainfo for ds in data_samples]
            seg_logits = model.inference(inputs, batch_img_metas)
            prob = seg_logits.softmax(dim=1)[0, 1].cpu().numpy()
            gt = data_samples[0].gt_sem_seg.data[0].cpu().numpy()
            valid = gt != IGNORE_INDEX
            out.append((prob.astype(np.float32), valid, gt))
            if (i + 1) % 50 == 0 or i + 1 == n:
                print('    %s: %d/%d' % (osp.basename(ckpt_path), i + 1, n))

    # 释放该模型占用的显存，再处理下一个模型
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return out


def main():
    """主函数：逐模型取概率 -> 平均 -> 多阈值统计 6 项指标。"""
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        print('未检测到 CUDA，回退使用 CPU。')
        args.device = 'cpu'

    register_all_modules()

    # 用第一个模型的配置构建验证数据集（各配置的验证划分与目录完全一致）
    first_cfg = Config.fromfile(args.pair[0][0])
    if 'custom_imports' in first_cfg:
        import_modules_from_strings(**first_cfg['custom_imports'])
    dataset = DATASETS.build(first_cfg.val_dataloader['dataset'])
    n = len(dataset)
    print('验证集样本数: %d；集成模型数: %d' % (n, len(args.pair)))

    # 逐模型收集概率（CPU），all_probs[k] 为第 k 个模型的整库结果列表
    all_probs = []
    for cfg_path, ckpt_path in args.pair:
        print('推理模型: %s' % ckpt_path)
        all_probs.append(
            collect_probs(cfg_path, ckpt_path, dataset, args.device))

    # 概率平均 + 多阈值混淆计数
    counts = {t: np.zeros(4, dtype=np.int64) for t in THRESHOLDS}
    for i in range(n):
        # 各模型该图裂缝概率平均（valid、gt 各模型一致，取第一个即可）
        prob = np.mean(
            np.stack([all_probs[k][i][0] for k in range(len(all_probs))], 0),
            axis=0)
        valid = all_probs[0][i][1]
        gt = all_probs[0][i][2]
        g = gt[valid].astype(bool)
        p = prob[valid]
        for t in THRESHOLDS:
            pred = p >= t
            tp = int(np.count_nonzero(pred & g))
            fp = int(np.count_nonzero(pred & ~g))
            fn = int(np.count_nonzero(~pred & g))
            tn = int(np.count_nonzero(~pred & ~g))
            counts[t] += np.array([tp, fp, fn, tn], dtype=np.int64)

    # 汇总输出
    results = {}
    header = ['阈值', 'F1', 'mIoU', 'Crack IoU', 'Acc', 'Pr', 'Re']
    print('\n' + '  '.join('%9s' % h for h in header))
    best_t, best_f1 = None, -1.0
    for t in THRESHOLDS:
        m = compute_metrics(counts[t])
        results['%.2f' % t] = {k: round(v, 2) for k, v in m.items()}
        row = ['%.2f' % t] + ['%.2f' % m[k] for k in header[1:]]
        print('  '.join('%9s' % c for c in row))
        if m['F1'] > best_f1:
            best_f1, best_t = m['F1'], t

    print('\n集成模型按 F1 最优阈值 = %.2f（F1=%.2f）' % (best_t, best_f1))
    results['_best_by_F1'] = best_t
    results['_members'] = [ck for _, ck in args.pair]

    out_dir = osp.join(ROOT, 'work_dirs', 'ensemble')
    os.makedirs(out_dir, exist_ok=True)
    out_json = osp.join(out_dir, 'ensemble_eval.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('结果已保存: %s' % out_json)


if __name__ == '__main__':
    main()
