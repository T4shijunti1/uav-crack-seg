# -*- coding: utf-8 -*-
"""加权跨结构集成（Weighted Ensemble）评估脚本。

在 ``ensemble_eval.py``（等权平均）基础上进一步：
    1. 逐个模型收集整库“裂缝概率图”（CPU numpy）；
    2. 在 CPU 上对**模型权重 × 判决阈值**做小规模网格搜索；
    3. 选出 mIoU 最高的组合，并打印赛道 D 的 6 项指标
       （F1 / mIoU / Crack IoU / Acc / Pr / Re）。

权重与阈值都属于“推理超参数”，在验证集上做模型选择是标准做法；
本脚本只在概率层面组合，不改动任何模型权重。

用法（仓库根目录，--member 可重复，每次给“配置 检查点 权重”）：
    python scripts/weighted_ensemble.py \
        --member configs/crack_seg_preunet.py work_dirs/preunet/best_mIoU_iter_20000.pth 3 \
        --member configs/crack_seg_improved.py work_dirs/improved/best_mIoU_iter_24000.pth 2 \
        --member configs/crack_seg_v3.py work_dirs/v3/best_mIoU_iter_32000.pth 1

若不传权重，脚本会自动搜索若干预设权重模板。
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

# 复用 ensemble_eval.py 中已验证的“逐模型取概率 / 指标计算”逻辑
from scripts.ensemble_eval import (  # noqa: E402
    collect_probs,
    compute_metrics,
)

# 更细的判决阈值网格（步长 0.01），用于刻画 mIoU / Pr / Re 的帕累托前沿
THRESHOLDS = [round(0.40 + 0.01 * i, 2) for i in range(21)]

# 概率缓存：同一组成员重复搜索时无需再次推理
CACHE_NPZ = osp.join(ROOT, 'work_dirs', 'ensemble', 'probs_cache.npz')
CACHE_META = osp.join(ROOT, 'work_dirs', 'ensemble', 'probs_cache.json')

# 4 个成员时自动搜索的权重模板（顺序与 --member 一致：P, improved, v3, v4）
WEIGHT_TEMPLATES_4 = [
    (1, 1, 1, 1),
    (2, 1, 1, 1),
    (2, 2, 1, 1),
    (3, 2, 1, 1),
    (3, 3, 1, 1),
    (2, 3, 1, 1),
    (3, 3, 2, 2),
    (2, 2, 1, 0),
    (2, 1, 0, 0),
    (1, 1, 0, 0),
    (3, 2, 0, 0),
]

# 5 个成员时自动搜索的权重模板（顺序：P=r34, improved, v3, v4, R=r50）
WEIGHT_TEMPLATES_5 = [
    (3, 2, 1, 1, 1),
    (3, 2, 1, 1, 2),
    (3, 2, 1, 1, 3),
    (2, 2, 1, 1, 1),
    (2, 2, 1, 1, 2),
    (2, 2, 1, 1, 3),
    (3, 3, 1, 1, 1),
    (3, 3, 1, 1, 2),
    (3, 2, 1, 0, 2),
    (2, 2, 1, 0, 2),
    (3, 2, 0, 0, 2),
    (2, 2, 0, 0, 2),
]


# 6 个成员时自动搜索的权重模板
# 顺序：r34, improved, v3, v4, r50ft, b3
# 分组：ECA 族(improved/v3/v4)、ResNet 族(r34/r50ft)、EfficientNet(b3)
WEIGHT_TEMPLATES_6 = [
    (1, 1, 1, 1, 1, 1),
    (3, 2, 1, 1, 2, 2),
    (3, 2, 1, 1, 2, 3),
    (3, 2, 1, 1, 3, 3),
    (2, 2, 1, 1, 2, 2),
    (2, 2, 1, 1, 2, 3),
    (3, 2, 0, 0, 2, 2),
    (3, 2, 0, 0, 2, 3),
    (2, 2, 0, 0, 2, 3),
    (3, 2, 1, 1, 2, 1),
    (3, 2, 1, 1, 1, 2),
    (4, 2, 1, 1, 2, 2),
    (3, 3, 1, 1, 2, 2),
    (3, 2, 1, 1, 3, 2),
    (2, 2, 1, 1, 3, 3),
]


# 7 个成员时自动搜索的权重模板
# 顺序：r34, improved, v3, v4, r50ft, b3, b4
WEIGHT_TEMPLATES_7 = [
    (1, 1, 1, 1, 1, 1, 1),
    (3, 2, 1, 1, 3, 3, 3),
    (3, 2, 1, 1, 3, 3, 2),
    (3, 2, 1, 1, 2, 2, 3),
    (3, 2, 1, 1, 3, 2, 3),
    (3, 2, 0, 0, 3, 3, 3),
    (3, 2, 0, 0, 2, 2, 3),
    (2, 2, 1, 1, 2, 2, 2),
    (3, 2, 1, 1, 2, 3, 3),
    (3, 2, 1, 1, 3, 3, 4),
    (4, 2, 1, 1, 3, 3, 3),
    (3, 3, 1, 1, 3, 3, 3),
    (2, 2, 1, 1, 3, 3, 3),
]


# 8 个成员时自动搜索的权重模板
# 顺序：r34, improved, v3, v4, r50ft, b3, b4, segformer
WEIGHT_TEMPLATES_8 = [
    (1, 1, 1, 1, 1, 1, 1, 1),
    (3, 2, 0, 0, 3, 3, 3, 1),
    (3, 2, 0, 0, 3, 3, 3, 2),
    (3, 2, 0, 0, 3, 3, 3, 3),
    (3, 2, 0, 0, 2, 2, 2, 2),
    (3, 2, 0, 0, 3, 3, 2, 2),
    (3, 2, 0, 0, 3, 2, 3, 2),
    (2, 2, 0, 0, 2, 2, 2, 2),
    (3, 2, 0, 0, 3, 3, 3, 4),
    (4, 2, 0, 0, 3, 3, 3, 2),
    (3, 3, 0, 0, 3, 3, 3, 2),
    (3, 2, 0, 0, 3, 3, 2, 3),
    (3, 2, 0, 0, 2, 3, 3, 3),
]


# 9 个成员时自动搜索的权重模板
# 顺序：r34, improved, v3, v4, r50ft, b3, b4, segformer, v2s
WEIGHT_TEMPLATES_9 = [
    (1, 1, 1, 1, 1, 1, 1, 1, 1),
    (3, 2, 0, 0, 3, 3, 3, 0, 3),
    (3, 2, 0, 0, 3, 3, 3, 1, 3),
    (3, 2, 0, 0, 3, 3, 3, 2, 3),
    (3, 2, 0, 0, 3, 3, 3, 0, 2),
    (3, 2, 0, 0, 3, 3, 3, 0, 4),
    (3, 2, 0, 0, 2, 2, 2, 0, 3),
    (3, 2, 0, 0, 3, 3, 2, 0, 3),
    (3, 2, 0, 0, 3, 2, 3, 0, 3),
    (2, 2, 0, 0, 2, 2, 2, 0, 2),
    (4, 2, 0, 0, 3, 3, 3, 0, 3),
    (3, 3, 0, 0, 3, 3, 3, 0, 3),
    (3, 2, 0, 0, 3, 3, 2, 1, 3),
    (3, 2, 0, 0, 2, 3, 3, 1, 3),
    (3, 2, 0, 0, 3, 3, 3, 1, 4),
]


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='加权跨结构集成评估')
    parser.add_argument(
        '--member', nargs=3, action='append',
        metavar=('CONFIG', 'CHECKPOINT', 'WEIGHT'),
        required=True, help='一组“配置 检查点 权重”，权重传 0 表示参与自动搜索')
    parser.add_argument(
        '--device', default='cuda:0', help='推理设备，如 cuda:0 或 cpu')
    return parser.parse_args()


def evaluate_combo(all_probs, weights, n):
    """给定一组权重，返回各阈值下的聚合混淆计数。

    Args:
        all_probs (list): 每个成员的整库概率结果列表。
        weights (tuple[float]): 各成员权重（无需归一化，函数内归一化）。
        n (int): 验证集样本数。

    Returns:
        dict: {阈值: np.array([tp, fp, fn, tn])}。
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    counts = {t: np.zeros(4, dtype=np.int64) for t in THRESHOLDS}
    for i in range(n):
        prob = np.zeros(np.asarray(all_probs[0][i][0]).shape, dtype=np.float64)
        for k in range(len(all_probs)):
            prob += w[k] * np.asarray(all_probs[k][i][0], dtype=np.float64)
        valid = np.asarray(all_probs[0][i][1]).astype(bool)
        gt = np.asarray(all_probs[0][i][2])
        g = gt[valid].astype(bool)
        p = prob[valid]
        for t in THRESHOLDS:
            pred = p >= t
            tp = int(np.count_nonzero(pred & g))
            fp = int(np.count_nonzero(pred & ~g))
            fn = int(np.count_nonzero(~pred & g))
            tn = int(np.count_nonzero(~pred & ~g))
            counts[t] += np.array([tp, fp, fn, tn], dtype=np.int64)
    return counts


def save_cache(all_probs, members):
    """把各成员整库概率缓存到磁盘（图像尺寸不一，用 object 数组）。"""
    os.makedirs(osp.dirname(CACHE_NPZ), exist_ok=True)
    data = {}
    for k, (_, ck, _) in enumerate(members):
        probs = np.array([p for p, _, _ in all_probs[k]], dtype=object)
        valids = np.array([v for _, v, _ in all_probs[k]], dtype=object)
        gts = np.array([g for _, _, g in all_probs[k]], dtype=object)
        data['p%d' % k] = probs
        data['v%d' % k] = valids
        data['g%d' % k] = gts
    np.savez(CACHE_NPZ, **data)
    with open(CACHE_META, 'w', encoding='utf-8') as f:
        json.dump([osp.basename(ck) for _, ck, _ in members], f)


def load_cache(members):
    """命中缓存则返回 all_probs，否则返回 None。"""
    if not (osp.exists(CACHE_NPZ) and osp.exists(CACHE_META)):
        return None
    with open(CACHE_META, encoding='utf-8') as f:
        meta = json.load(f)
    if meta != [osp.basename(ck) for _, ck, _ in members]:
        return None
    z = np.load(CACHE_NPZ, allow_pickle=True)
    all_probs = []
    for k in range(len(members)):
        probs, valids, gts = z['p%d' % k], z['v%d' % k], z['g%d' % k]
        all_probs.append(
            [(probs[i], valids[i], gts[i]) for i in range(len(probs))])
    print('已加载概率缓存，跳过模型推理: %s' % CACHE_NPZ)
    return all_probs


def main():
    """主函数：取概率 -> 权重×阈值网格搜索 -> 输出最优组合 6 项指标。"""
    args = parse_args()
    import torch
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import DATASETS
    from mmseg.utils import register_all_modules

    if torch.cuda.is_available() and args.device.startswith('cuda'):
        pass
    elif args.device.startswith('cuda'):
        print('未检测到 CUDA，回退 CPU。')
        args.device = 'cpu'

    register_all_modules()

    members = [(c, k, float(wi)) for c, k, wi in args.member]
    m_num = len(members)

    # 用第一个成员的配置构建共用的验证数据集
    first_cfg = Config.fromfile(members[0][0])
    if 'custom_imports' in first_cfg:
        import_modules_from_strings(**first_cfg['custom_imports'])
    dataset = DATASETS.build(first_cfg.val_dataloader['dataset'])
    n = len(dataset)
    print('验证集样本数: %d；成员数: %d' % (n, m_num))

    # 逐成员取概率（优先读缓存）
    all_probs = load_cache(members)
    if all_probs is None:
        all_probs = []
        for cfg_path, ckpt_path, _ in members:
            print('推理成员: %s' % ckpt_path)
            all_probs.append(
                collect_probs(cfg_path, ckpt_path, dataset, args.device))
        save_cache(all_probs, members)

    # 确定要搜索的权重模板
    given = tuple(int(round(w)) for _, _, w in members)
    if all(w > 0 for w in given):
        templates = [given]           # 用户给了全正权重，只评估这一组
    elif m_num == 4:
        templates = WEIGHT_TEMPLATES_4
    elif m_num == 5:
        templates = WEIGHT_TEMPLATES_5
    elif m_num == 6:
        templates = WEIGHT_TEMPLATES_6
    elif m_num == 7:
        templates = WEIGHT_TEMPLATES_7
    elif m_num == 8:
        templates = WEIGHT_TEMPLATES_8
    elif m_num == 9:
        templates = WEIGHT_TEMPLATES_9
    else:
        # 成员数不是 4/5 时退化为只评估等权
        templates = [tuple(1 for _ in range(m_num))]

    # 网格搜索
    best = None
    rows = []
    for wvec in templates:
        counts = evaluate_combo(all_probs, wvec, n)
        for t in THRESHOLDS:
            metrics = compute_metrics(counts[t])
            rows.append((wvec, t, metrics))
            cand = (metrics['mIoU'], metrics['F1'])
            if best is None or cand > best[0]:
                best = (cand, wvec, t, metrics)

    # 输出每个模板在其最优阈值下的结果（便于查看趋势）
    print('\n各权重模板的最优阈值结果：')
    header = ['权重组合', '阈值', 'F1', 'mIoU', 'Crack IoU', 'Acc', 'Pr', 'Re']
    print('  '.join('%12s' % h for h in header))
    seen = set()
    for wvec in templates:
        sub = [r for r in rows if r[0] == wvec]
        wb = max(sub, key=lambda r: (r[2]['mIoU'], r[2]['F1']))
        key = wvec
        if key in seen:
            continue
        seen.add(key)
        m = wb[2]
        line = [str(wvec), '%.2f' % wb[1]] + ['%.2f' % m[k] for k in header[2:]]
        print('  '.join('%12s' % c for c in line))

    (_, bw, bt, bmetrics) = best
    print('\n===== mIoU 最高的组合 =====')
    print('权重=%s，阈值=%.2f' % (bw, bt))
    order = ['F1', 'mIoU', 'Crack IoU', 'Acc', 'Pr', 'Re']
    print('  '.join('%s=%.2f' % (k, bmetrics[k]) for k in order))

    # 在 mIoU≥80 的前提下，选择 Pr 最高的点（尽量保证精确率不下降）
    feasible = [r for r in rows if r[2]['mIoU'] >= 80.0]
    if feasible:
        rec = max(feasible, key=lambda r: (r[2]['Pr'], r[2]['mIoU']))
    else:
        rec = (bw, bt, bmetrics)
    wvec, t, metrics = rec

    # 打印推荐模板的完整阈值前沿
    print('\n推荐组合 权重=%s 的阈值前沿（核对 Pr/Re 权衡）：' % (wvec,))
    print('  '.join('%8s' % h for h in ['阈值', 'F1', 'mIoU', 'CrackIoU', 'Pr', 'Re']))
    for ww, tt, mm in [r for r in rows if r[0] == wvec]:
        print('  '.join('%8s' % c for c in [
            '%.2f' % tt, '%.2f' % mm['F1'], '%.2f' % mm['mIoU'],
            '%.2f' % mm['Crack IoU'], '%.2f' % mm['Pr'], '%.2f' % mm['Re']]))

    print('\n===== 最终采用（mIoU≥80 且 Pr 最高）=====')
    print('权重=%s，阈值=%.2f' % (wvec, t))
    print('  '.join('%s=%.2f' % (k, metrics[k]) for k in order))

    out_dir = osp.join(ROOT, 'work_dirs', 'ensemble')
    os.makedirs(out_dir, exist_ok=True)
    out_json = osp.join(out_dir, 'weighted_ensemble.json')
    payload = {
        'weights': wvec,
        'threshold': t,
        'members': [ck for _, ck, _ in members],
        'metrics': {k: round(v, 2) for k, v in metrics.items()},
        'max_miou_point': {
            'weights': bw, 'threshold': bt,
            'metrics': {k: round(v, 2) for k, v in bmetrics.items()}},
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print('结果已保存: %s' % out_json)


if __name__ == '__main__':
    main()
