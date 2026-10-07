# -*- coding: utf-8 -*-
"""数据准备脚本：把 Cityscapes 风格的 UAV-Crack 原始数据整理为 MMSeg 布局。

仅使用 Python 标准库（无需 Pillow），通过复制文件完成，不会修改原始数据集。

源数据结构：
    SRC/leftImg8bit/train/<X>/<stem>.jpg
    SRC/gtFine/train/<X>/<stem>.png        # 8 位灰度，取值 {0,1}
    SRC/leftImg8bit/val/*.jpg              # 无标注 -> 作为 test 推理集

将 1200 张“有标注”的训练图片按“同一 DJI 航片帧”分组后切分为 train/val
（同一航片帧的所有补丁进入同一子集，避免数据泄漏）。

输出结构：
    DST/images/{train,val,test}/*.jpg
    DST/annotations/{train,val}/*.png

示例：
    # Windows
    python scripts/prepare_crack_data.py \
        --src "C:/Users/rentianci/Desktop/数据集/UAV-Crack-dataset" \
        --dst "data/Crack"
    # WSL2
    python scripts/prepare_crack_data.py \
        --src "/mnt/c/Users/rentianci/Desktop/数据集/UAV-Crack-dataset" \
        --dst "data/Crack"
"""
import argparse
import os
import random
import re
import shutil


def main():
    """主函数：扫描配对 -> 按帧分组切分 -> 复制到目标目录。"""
    parser = argparse.ArgumentParser(description='构建 MMSeg 格式裂缝数据集')
    parser.add_argument('--src', required=True, help='原始 UAV-Crack-dataset 路径')
    parser.add_argument('--dst', required=True, help='输出数据根目录（data/Crack）')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    src, dst = args.src, args.dst

    # 创建输出目录
    for sub in [
            'images/train', 'images/val', 'images/test', 'annotations/train',
            'annotations/val']:
        os.makedirs(os.path.join(dst, sub), exist_ok=True)

    train_img_root = os.path.join(src, 'leftImg8bit', 'train')
    train_ann_root = os.path.join(src, 'gtFine', 'train')

    pairs = []  # (航片帧分组键, 图片路径, 掩码路径)
    for subset in sorted(os.listdir(train_img_root)):
        img_dir = os.path.join(train_img_root, subset)
        ann_dir = os.path.join(train_ann_root, subset)
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith('.jpg'):
                continue
            stem = os.path.splitext(fn)[0]
            img_path = os.path.join(img_dir, fn)
            ann_path = os.path.join(ann_dir, stem + '.png')
            if not os.path.exists(ann_path):
                print('警告：缺少掩码', ann_path)
                continue
            # 去掉结尾的 “_<补丁序号>_z”，得到所属原始航片帧标识
            match = re.match(r'^(.*)_\d+_z$', stem)
            frame = match.group(1) if match else stem
            pairs.append((frame, img_path, ann_path))

    # 按航片帧分组并随机切分
    frames = sorted(set(frame for frame, _, _ in pairs))
    random.Random(args.seed).shuffle(frames)
    n_val = int(round(len(frames) * args.val_ratio))
    val_frames = set(frames[:n_val])

    counts = {'train': 0, 'val': 0}
    for frame, img_path, ann_path in pairs:
        split = 'val' if frame in val_frames else 'train'
        base = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(dst, 'images', split, base))
        shutil.copy2(
            ann_path,
            os.path.join(dst, 'annotations', split,
                         os.path.splitext(base)[0] + '.png'))
        counts[split] += 1

    # 原始 val（无标注）作为 test 推理集
    n_test = 0
    val_dir = os.path.join(src, 'leftImg8bit', 'val')
    if os.path.isdir(val_dir):
        for fn in sorted(os.listdir(val_dir)):
            if fn.lower().endswith('.jpg'):
                shutil.copy2(
                    os.path.join(val_dir, fn),
                    os.path.join(dst, 'images', 'test', fn))
                n_test += 1

    print('航片帧总数：%d，其中验证帧：%d' % (len(frames), n_val))
    print('有标注训练图片：%d' % counts['train'])
    print('有标注验证图片：%d' % counts['val'])
    print('无标注测试图片：%d' % n_test)
    print('完成，输出目录 ->', dst)


if __name__ == '__main__':
    main()
