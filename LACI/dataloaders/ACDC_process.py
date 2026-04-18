# -*- coding: utf-8 -*-
"""
ACDC 3D preprocessing
功能：
1. 从 ACDC training 中读取每个 patient 的 ED / ES
2. 使用标签辅助定位裁剪（bbox + margin）
3. 只重采样 x/y spacing，保留原始 z spacing
4. x/y 最终 resize 到 196x196
5. z 最终统一到 48：
   - 若 z > 48：中心裁剪到 48
   - 若 z < 48：复制内部相邻层补齐
   - 不复制第一层和最后一层
6. 保存 h5 和 nii，分开放置

依赖：
pip install nibabel h5py numpy scipy tqdm
"""

import json
import h5py
import numpy as np
import nibabel as nib

from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm


# =========================
# 路径配置
# =========================
SRC_ROOT = Path("/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/ACDC/training")
DST_ROOT = Path("/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/ACDC/Processing_3D")

H5_ROOT = DST_ROOT / "h5"
NII_IMAGE_ROOT = DST_ROOT / "nii_images"
NII_LABEL_ROOT = DST_ROOT / "nii_labels"
META_ROOT = DST_ROOT / "meta"
LIST_ROOT = DST_ROOT / "lists"

# =========================
# 预处理参数
# =========================
OUT_SHAPE = (192, 192, 48)      # (X, Y, Z)
TARGET_XY_SPACING = (1.33, 1.33)

# 标签 bbox 外扩 margin（原始 voxel 单位）
MARGIN_X = 20
MARGIN_Y = 20
MARGIN_Z = 2

# 强度裁剪
LOW_P = 0.5
HIGH_P = 99.5

USE_LABEL_GUIDED_CROP = True


def parse_info_cfg(cfg_path: Path):
    ed = None
    es = None
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ED:"):
                ed = int(line.split(":")[1].strip())
            elif line.startswith("ES:"):
                es = int(line.split(":")[1].strip())
    if ed is None or es is None:
        raise ValueError(f"Cannot parse ED/ES from {cfg_path}")
    return ed, es


def frame_name(frame_id: int):
    return f"frame{frame_id:02d}"


def load_nifti(path: Path):
    nii = nib.load(str(path))
    data = nii.get_fdata()
    affine = nii.affine.copy()
    header = nii.header.copy()
    spacing = header.get_zooms()[:3]
    return data, affine, header, spacing


def get_label_bbox(label, margin_xyz=(20, 20, 2)):
    """
    根据 label > 0 得到前景 bbox，并加 margin
    label shape: (X, Y, Z)
    return: (x0, x1, y0, y1, z0, z1)
    """
    coords = np.where(label > 0)
    if len(coords[0]) == 0:
        return (0, label.shape[0], 0, label.shape[1], 0, label.shape[2])

    mx, my, mz = margin_xyz

    x0 = max(int(coords[0].min()) - mx, 0)
    x1 = min(int(coords[0].max()) + 1 + mx, label.shape[0])

    y0 = max(int(coords[1].min()) - my, 0)
    y1 = min(int(coords[1].max()) + 1 + my, label.shape[1])

    z0 = max(int(coords[2].min()) - mz, 0)
    z1 = min(int(coords[2].max()) + 1 + mz, label.shape[2])

    return (x0, x1, y0, y1, z0, z1)


def crop_with_bbox(arr, bbox):
    x0, x1, y0, y1, z0, z1 = bbox
    return arr[x0:x1, y0:y1, z0:z1]


def robust_intensity_normalize(img):
    """
    非零区域 z-score 标准化，先做轻微 percentile clipping
    """
    img = img.astype(np.float32)
    mask = img != 0

    if np.any(mask):
        vals = img[mask]
        lo = np.percentile(vals, LOW_P)
        hi = np.percentile(vals, HIGH_P)
        img = np.clip(img, lo, hi)

        vals = img[mask]
        mean = vals.mean()
        std = vals.std()
        if std < 1e-8:
            std = 1.0
        img[mask] = (img[mask] - mean) / std
    else:
        mean = img.mean()
        std = img.std()
        if std < 1e-8:
            std = 1.0
        img = (img - mean) / std

    return img.astype(np.float32)


def resample_xy_keep_z(image, label, old_spacing, target_xy=(1.33, 1.33)):
    """
    只重采样 x/y，z 不动
    image/label shape: (X, Y, Z)
    old_spacing: (sx, sy, sz)
    """
    sx, sy, sz = old_spacing
    tx, ty = target_xy

    zoom_x = sx / tx
    zoom_y = sy / ty
    zoom_z = 1.0

    image_rs = zoom(image, zoom=(zoom_x, zoom_y, zoom_z), order=1)  # linear
    label_rs = zoom(label, zoom=(zoom_x, zoom_y, zoom_z), order=0)  # nearest

    new_spacing = (tx, ty, sz)
    return image_rs.astype(np.float32), label_rs.astype(np.uint8), new_spacing


def resize_xy_to_target(image, label, out_xy=(192, 192)):
    """
    将 x/y resize 到目标大小，z 不变
    image: order=1
    label: order=0
    """
    out_x, out_y = out_xy
    x, y, z = image.shape

    zoom_x = out_x / x
    zoom_y = out_y / y

    image_out = zoom(image, zoom=(zoom_x, zoom_y, 1.0), order=1)
    label_out = zoom(label, zoom=(zoom_x, zoom_y, 1.0), order=0)

    return image_out.astype(np.float32), label_out.astype(np.uint8)


def adjust_z_with_internal_slice_copy(image, label, target_z=48):
    """
    调整 z 到 target_z

    规则：
    - 若 z == target_z: 直接返回
    - 若 z > target_z: 中心裁剪
    - 若 z < target_z:
        复制“内部层”补齐，不复制第一层和最后一层
        也就是从 [1, z-2] 范围中选层进行复制插入
    """
    x, y, z = image.shape

    if z == target_z:
        return image, label

    if z > target_z:
        start_z = max((z - target_z) // 2, 0)
        image = image[:, :, start_z:start_z + target_z]
        label = label[:, :, start_z:start_z + target_z]
        return image, label

    # z < target_z
    if z == 1:
        # 极端情况：只有一层，只能复制它
        img_out = np.repeat(image, target_z, axis=2)
        lab_out = np.repeat(label, target_z, axis=2)
        return img_out, lab_out

    if z == 2:
        # 只有两层时，无法避开首尾，只能交替复制
        idx = [0, 1]
        while len(idx) < target_z:
            idx.extend([0, 1])
        idx = idx[:target_z]
        return image[:, :, idx], label[:, :, idx]

    # 一般情况：z >= 3
    # 不复制第一层和最后一层，所以内部可复制层为 [1, ..., z-2]
    current_indices = list(range(z))
    internal_indices = list(range(1, z - 1))  # 不含 0 和 z-1

    needed = target_z - z

    # 尽量均匀地从内部挑层进行复制
    sampled = []
    for k in range(needed):
        pos = (k + 1) * len(internal_indices) / (needed + 1)
        pick = internal_indices[int(np.floor(pos))]
        sampled.append(pick)

    # 将这些复制层插入到原层后面，保持顺序
    insert_count = {i: 0 for i in range(z)}
    for s in sampled:
        insert_count[s] += 1

    final_indices = []
    for i in range(z):
        final_indices.append(i)
        # 复制插入到当前层后面
        if i in insert_count:
            final_indices.extend([i] * insert_count[i])

    final_indices = final_indices[:target_z]

    image_out = image[:, :, final_indices]
    label_out = label[:, :, final_indices]

    return image_out, label_out


def update_affine_for_crop_and_resample(orig_affine, bbox, old_shape_after_crop, new_shape_after_xy, new_spacing):
    """
    近似更新 affine：
    1) 根据 crop 起点平移 origin
    2) 根据 x/y 重采样更新 spacing
    3) 后续 x/y resize 到 196、z 复制/裁剪，本质是网络输入对齐，不再追求严格物理保真
       因此这里 affine 主要用于查看，不作为严格几何真值
    """
    affine = orig_affine.copy()
    x0, _, y0, _, z0, _ = bbox

    crop_origin_voxel = np.array([x0, y0, z0, 1.0], dtype=np.float64)
    new_origin_world = orig_affine @ crop_origin_voxel
    affine[:3, 3] = new_origin_world[:3]

    for i in range(3):
        sign = -1.0 if affine[i, i] < 0 else 1.0
        affine[i, i] = sign * float(new_spacing[i])

    return affine


def save_h5(h5_path: Path, image, label, meta: dict):
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("image", data=image.astype(np.float32), compression="gzip")
        f.create_dataset("label", data=label.astype(np.uint8), compression="gzip")
        f.attrs["meta"] = json.dumps(meta, ensure_ascii=False)


def save_nifti(path: Path, array, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(array, affine)
    nib.save(nii, str(path))


def process_one_case(patient_dir: Path, frame_id: int, phase_name: str):
    patient_id = patient_dir.name
    stem = frame_name(frame_id)

    img_path = patient_dir / f"{patient_id}_{stem}.nii.gz"
    lab_path = patient_dir / f"{patient_id}_{stem}_gt.nii.gz"

    if not img_path.exists():
        raise FileNotFoundError(img_path)
    if not lab_path.exists():
        raise FileNotFoundError(lab_path)

    image, affine, header, old_spacing = load_nifti(img_path)
    label, _, _, _ = load_nifti(lab_path)
    label = np.rint(label).astype(np.uint8)

    if image.ndim != 3 or label.ndim != 3:
        raise ValueError(f"Expected 3D volume, got image {image.shape}, label {label.shape}")

    original_shape = tuple(int(v) for v in image.shape)

    # 1) 标签辅助裁剪
    if USE_LABEL_GUIDED_CROP:
        bbox = get_label_bbox(label, margin_xyz=(MARGIN_X, MARGIN_Y, MARGIN_Z))
        image = crop_with_bbox(image, bbox)
        label = crop_with_bbox(label, bbox)
    else:
        bbox = (0, image.shape[0], 0, image.shape[1], 0, image.shape[2])

    cropped_shape = tuple(int(v) for v in image.shape)

    # 2) 只重采样 x/y，z 不动
    image_rs, label_rs, new_spacing = resample_xy_keep_z(
        image=image,
        label=label,
        old_spacing=old_spacing,
        target_xy=TARGET_XY_SPACING
    )
    resampled_shape = tuple(int(v) for v in image_rs.shape)

    # 3) 强度归一化
    image_rs = robust_intensity_normalize(image_rs)

    # 4) x/y 直接 resize 到 196x196
    image_xy, label_xy = resize_xy_to_target(
        image_rs, label_rs, out_xy=(OUT_SHAPE[0], OUT_SHAPE[1])
    )
    resized_xy_shape = tuple(int(v) for v in image_xy.shape)

    # 5) z 调整到 48：内部层复制，不复制首尾
    image_out, label_out = adjust_z_with_internal_slice_copy(
        image_xy, label_xy, target_z=OUT_SHAPE[2]
    )

    final_shape = tuple(int(v) for v in image_out.shape)
    assert final_shape == OUT_SHAPE, f"Final shape mismatch: {final_shape} vs {OUT_SHAPE}"

    # affine 仅用于导出查看
    new_affine = update_affine_for_crop_and_resample(
        orig_affine=affine,
        bbox=bbox,
        old_shape_after_crop=cropped_shape,
        new_shape_after_xy=resized_xy_shape,
        new_spacing=new_spacing
    )

    sample_name = f"{patient_id}_{phase_name}"

    h5_path = H5_ROOT / f"{sample_name}.h5"
    nii_img_path = NII_IMAGE_ROOT / f"{sample_name}.nii.gz"
    nii_lab_path = NII_LABEL_ROOT / f"{sample_name}.nii.gz"
    meta_path = META_ROOT / f"{sample_name}.json"

    meta = {
        "patient_id": patient_id,
        "phase": phase_name,
        "frame_id": int(frame_id),
        "src_image": str(img_path),
        "src_label": str(lab_path),
        "original_shape_xyz": list(original_shape),
        "cropped_shape_xyz": list(cropped_shape),
        "resampled_shape_xyz": list(resampled_shape),
        "resized_xy_shape_xyz": list(resized_xy_shape),
        "output_shape_xyz": list(final_shape),
        "original_spacing_xyz": [float(v) for v in old_spacing],
        "xy_resampled_spacing_xyz": [float(v) for v in new_spacing],
        "bbox_xyz": [int(v) for v in bbox],
        "z_process": "If z < 48, duplicate internal neighboring slices only; first and last slices are not duplicated. If z > 48, center crop.",
        "note": "x/y are resized to fixed size 196x196; z keeps original spacing before slice duplication/cropping."
    }

    save_h5(h5_path, image_out, label_out, meta)
    save_nifti(nii_img_path, image_out.astype(np.float32), new_affine)
    save_nifti(nii_lab_path, label_out.astype(np.uint8), new_affine)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return sample_name


def main():
    H5_ROOT.mkdir(parents=True, exist_ok=True)
    NII_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    NII_LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    META_ROOT.mkdir(parents=True, exist_ok=True)
    LIST_ROOT.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([
        p for p in SRC_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("patient")
    ])

    all_list = []
    ed_list = []
    es_list = []

    for patient_dir in tqdm(patient_dirs, desc="Processing ACDC"):
        cfg_path = patient_dir / "Info.cfg"
        if not cfg_path.exists():
            print(f"[WARN] Missing Info.cfg: {cfg_path}")
            continue

        try:
            ed, es = parse_info_cfg(cfg_path)

            ed_name = process_one_case(patient_dir, ed, "ED")
            es_name = process_one_case(patient_dir, es, "ES")

            all_list.extend([ed_name, es_name])
            ed_list.append(ed_name)
            es_list.append(es_name)

        except Exception as e:
            print(f"[ERROR] {patient_dir}: {e}")

    with open(LIST_ROOT / "all.list", "w", encoding="utf-8") as f:
        for name in all_list:
            f.write(name + "\n")

    with open(LIST_ROOT / "all_ED.list", "w", encoding="utf-8") as f:
        for name in ed_list:
            f.write(name + "\n")

    with open(LIST_ROOT / "all_ES.list", "w", encoding="utf-8") as f:
        for name in es_list:
            f.write(name + "\n")

    print(f"Done. Results saved to: {DST_ROOT}")


if __name__ == "__main__":
    main()