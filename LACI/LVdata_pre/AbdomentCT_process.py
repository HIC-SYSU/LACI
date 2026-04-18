"""
AbdomenCT 数据预处理脚本
采用 nnUNet 3d_lowres 配置:
- Target Spacing: 1.89mm x 1.89mm x 1.89mm (各向同性)
- Normalization: CTNormalization
- 输出: h5 文件格式 (keys: image, label)

输出文件:
- processed_data/h5_file/*.h5
- all_data.txt (所有数据路径)
- train.txt (800个训练数据)
- test.txt (200个测试数据)

使用方法:
    cd /data/xingshihanxiao/Pyproject/open_data/AbdomenCT
    python preprocess_abdomenct.py --num_workers 4
"""

import os
import argparse
import numpy as np
import nibabel as nib
import h5py
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========================= nnUNet 配置参数 =========================
CONFIG = {
    # 来自 nnUNet 3d_lowres 自动计算的配置
    "target_spacing": (1.89, 1.89, 1.89),  # mm, 各向同性

    # 数据分割
    "num_train": 800,
    "num_test": 200,
}


def load_nifti(filepath):
    """加载 NIfTI 文件，返回数据和元信息"""
    nii = nib.load(filepath)
    data = nii.get_fdata().astype(np.float32)
    spacing = np.array(nii.header.get_zooms()[:3])
    return data, spacing


def resample_volume(volume, original_spacing, target_spacing, order=3):
    """
    重采样 3D 体积数据到目标 spacing

    Args:
        volume: 3D numpy array
        original_spacing: 原始 spacing
        target_spacing: 目标 spacing
        order: 插值阶数 (0=最近邻, 1=线性, 3=三次)
    """
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    resampled = zoom(volume, zoom_factors, order=order)
    return resampled


def ct_normalization(image, clip_lower, clip_upper, mean, std):
    """
    nnUNet 的 CT 专用归一化
    1. Clip 到 percentile 范围
    2. Z-score 归一化
    """
    image = np.clip(image, clip_lower, clip_upper)
    image = (image - mean) / std
    return image


def compute_dataset_statistics(image_dir, mask_dir):
    """
    计算数据集的强度统计量（用于归一化）
    使用全部数据
    """
    print("正在计算数据集统计量（使用全部数据）...")

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz') and not f.startswith('.')])
    print(f"共找到 {len(mask_files)} 个样本")

    all_foreground_values = []

    for mask_file in tqdm(mask_files, desc="计算统计量"):
        case_id = mask_file.replace('.nii.gz', '')
        image_file = f"{case_id}_0000.nii.gz"

        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(image_path):
            continue

        image, _ = load_nifti(image_path)
        mask, _ = load_nifti(mask_path)

        # 提取前景区域的强度值
        foreground_mask = mask > 0
        if foreground_mask.sum() > 0:
            foreground_values = image[foreground_mask]
            # 随机采样（避免内存溢出）
            if len(foreground_values) > 10000:
                indices = np.random.choice(len(foreground_values), 10000, replace=False)
                foreground_values = foreground_values[indices]
            all_foreground_values.extend(foreground_values.tolist())

    all_foreground_values = np.array(all_foreground_values)

    stats = {
        "percentile_00_5": float(np.percentile(all_foreground_values, 0.5)),
        "percentile_99_5": float(np.percentile(all_foreground_values, 99.5)),
        "mean": float(np.mean(all_foreground_values)),
        "std": float(np.std(all_foreground_values)),
        "min": float(np.min(all_foreground_values)),
        "max": float(np.max(all_foreground_values)),
        "median": float(np.median(all_foreground_values)),
        "num_samples": len(mask_files),
    }

    print("\n=== 数据集统计量 ===")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    return stats


def preprocess_single_case(args):
    """预处理单个样本"""
    case_id, image_path, mask_path, output_dir, config, stats = args

    try:
        # 加载数据
        image, spacing = load_nifti(image_path)
        mask, _ = load_nifti(mask_path)

        # 重采样到目标 spacing
        target_spacing = config["target_spacing"]
        image_resampled = resample_volume(image, spacing, target_spacing, order=3)
        mask_resampled = resample_volume(mask, spacing, target_spacing, order=0)  # 最近邻
        mask_resampled = np.round(mask_resampled).astype(np.uint8)

        # CT 归一化
        image_normalized = ct_normalization(
            image_resampled,
            stats["percentile_00_5"],
            stats["percentile_99_5"],
            stats["mean"],
            stats["std"]
        ).astype(np.float32)

        # 保存为 h5 文件
        h5_filename = f"{case_id}.h5"
        h5_path = os.path.join(output_dir, h5_filename)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('image', data=image_normalized, compression="gzip", compression_opts=4)
            f.create_dataset('label', data=mask_resampled, compression="gzip", compression_opts=4)
            f.attrs['case_id'] = case_id
            f.attrs['original_spacing'] = spacing.tolist()
            f.attrs['target_spacing'] = list(target_spacing)
            f.attrs['original_shape'] = list(image.shape)
            f.attrs['resampled_shape'] = list(image_resampled.shape)

        return case_id, h5_path, image_resampled.shape, True, None

    except Exception as e:
        return case_id, None, None, False, str(e)


def create_train_test_split(all_paths, output_dir, num_train=800, seed=42):
    """
    创建训练/测试划分，生成 txt 文件
    """
    np.random.seed(seed)

    # 随机打乱
    paths = all_paths.copy()
    np.random.shuffle(paths)

    train_paths = paths[:num_train]
    test_paths = paths[num_train:]

    # 写入 txt 文件
    all_txt = os.path.join(output_dir, "all_data.txt")
    train_txt = os.path.join(output_dir, "train.txt")
    test_txt = os.path.join(output_dir, "test.txt")

    with open(all_txt, 'w') as f:
        for p in all_paths:
            f.write(f"{p}\n")

    with open(train_txt, 'w') as f:
        for p in train_paths:
            f.write(f"{p}\n")

    with open(test_txt, 'w') as f:
        for p in test_paths:
            f.write(f"{p}\n")

    print(f"数据划分完成:")
    print(f"  all_data.txt: {len(all_paths)} 样本 -> {all_txt}")
    print(f"  train.txt: {len(train_paths)} 样本 -> {train_txt}")
    print(f"  test.txt: {len(test_paths)} 样本 -> {test_txt}")

    return train_paths, test_paths


def main():
    parser = argparse.ArgumentParser(description="AbdomenCT 数据预处理 (nnUNet 配置, h5 格式)")
    parser.add_argument("--num_workers", type=int, default=4, help="并行工作进程数")
    parser.add_argument("--output_subdir", type=str, default="processed_data", help="输出子目录名")

    args = parser.parse_args()

    # 路径设置
    script_dir = Path(__file__).parent
    image_dir = script_dir / "Image"
    mask_dir = script_dir / "Mask"
    output_dir = script_dir / args.output_subdir
    h5_output_dir = output_dir / "h5_file"
    h5_output_dir.mkdir(parents=True, exist_ok=True)

    config = CONFIG.copy()

    # ============ 步骤 1: 计算统计量 ============
    print("=" * 60)
    print("步骤 1/3: 计算数据集统计量")
    print("=" * 60)

    stats = compute_dataset_statistics(str(image_dir), str(mask_dir))

    # 保存统计量
    with open(output_dir / "dataset_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # ============ 步骤 2: 预处理数据 ============
    print("\n" + "=" * 60)
    print("步骤 2/3: 预处理数据 (重采样 + 归一化 -> h5)")
    print("=" * 60)

    # 获取所有有标签的样本
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz') and not f.startswith('.')])

    # 准备任务
    tasks = []
    for mask_file in mask_files:
        case_id = mask_file.replace('.nii.gz', '')
        image_file = f"{case_id}_0000.nii.gz"
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(image_path):
            tasks.append((case_id, image_path, mask_path, str(h5_output_dir), config, stats))

    print(f"共找到 {len(tasks)} 个有效样本")

    # 并行处理
    all_h5_paths = []
    shapes = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(preprocess_single_case, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="预处理中"):
            case_id, h5_path, shape, success, error = future.result()
            if success:
                all_h5_paths.append(h5_path)
                shapes.append(shape)
            else:
                print(f"警告: {case_id} 处理失败: {error}")

    # 排序路径
    all_h5_paths = sorted(all_h5_paths)

    # 保存处理信息
    info = {
        "config": config,
        "statistics": stats,
        "num_samples": len(all_h5_paths),
        "sample_shapes": {
            "min": [int(min(s[i] for s in shapes)) for i in range(3)],
            "max": [int(max(s[i] for s in shapes)) for i in range(3)],
            "mean": [float(np.mean([s[i] for s in shapes])) for i in range(3)],
        }
    }

    with open(output_dir / "preprocessing_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n预处理完成！共处理 {len(all_h5_paths)} 个样本")
    print(f"h5 文件目录: {h5_output_dir}")
    print(f"重采样后尺寸范围: {info['sample_shapes']}")

    # ============ 步骤 3: 创建训练/测试划分 ============
    print("\n" + "=" * 60)
    print("步骤 3/3: 创建训练/测试划分")
    print("=" * 60)

    create_train_test_split(
        all_h5_paths,
        str(script_dir),  # txt 文件放在数据根目录
        num_train=config["num_train"]
    )

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
