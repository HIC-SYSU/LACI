import os
import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage
from tqdm import tqdm

output_size = [112, 112, 80]

import os
import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage
from tqdm import tqdm

# ---------- 标签映射 ----------
def remap_labels(label_data):
    mapping = {205: 1, 500: 2, 600: 3, 420: 4, 550: 5, 820: 6}
    label_remap = np.zeros_like(label_data, dtype=np.uint8)
    for old, new in mapping.items():
        label_remap[label_data == old] = new
    return label_remap

def normalize_image(img):
    """归一化到 0 均值、1 方差"""
    return (img - np.mean(img)) / np.std(img)

def crop_with_margin(image, label, margin=25):
    """根据标签裁剪 + margin"""
    w, h, d = image.shape
    tempL = np.nonzero(label)
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    minx = max(minx - margin, 0)
    maxx = min(maxx + margin, w)
    miny = max(miny - margin, 0)
    maxy = min(maxy + margin, h)
    minz = max(minz - margin, 0)
    maxz = min(maxz + margin, d)

    return image[minx:maxx, miny:maxy, minz:maxz], \
           label[minx:maxx, miny:maxy, minz:maxz], \
           [(minx+maxx)//2, (miny+maxy)//2, (minz+maxz)//2]

def resample_to_isotropic(image, label, spacing, new_spacing=(1.0,1.0,1.0)):
    """重采样到 isotropic spacing"""
    zoom = np.array(spacing) / np.array(new_spacing)
    image_res = scipy.ndimage.zoom(image, zoom, order=1)
    label_res = scipy.ndimage.zoom(label, zoom, order=0)
    return image_res, label_res

def save_h5(image, label, save_path):
    image = image.astype(np.float32)
    label = label.astype(np.uint8)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")

def preprocess_train(image_paths, label_paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    centers = []

    for img_p, lbl_p in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        print(img_p)
        if img_p =='/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/MM-WHS 2017 Dataset/ct_train/ct_train_1001_image.nii.gz':
            img = nib.load(img_p)
            lbl = nib.load(lbl_p)

            image_data = img.get_fdata()
            label_data = lbl.get_fdata()

            # ------- 标签重映射 -------
            label_data = remap_labels(label_data)

            spacing = img.header.get_zooms()[:3]

            # 重采样
            image_res, label_res = resample_to_isotropic(image_data, label_data, spacing)

            # 裁剪 + margin
            image_crop, label_crop, center = crop_with_margin(image_res, label_res, margin=25)
            centers.append(center)

            # 归一化
            image_norm = normalize_image(image_crop)

            # ------- 生成 label_slice -------
            z_dim, y_dim, x_dim = label_crop.shape
            label_slice = np.full_like(label_crop, 255, dtype=np.uint8)

            # 在 30%-70% 区间随机取切片
            cz = np.random.randint(int(0.3 * z_dim), int(0.7 * z_dim))
            cy = np.random.randint(int(0.3 * y_dim), int(0.7 * y_dim))
            cx = np.random.randint(int(0.3 * x_dim), int(0.7 * x_dim))

            # 保留三个方向的切片
            label_slice[cz, :, :] = label_crop[cz, :, :]
            label_slice[:, cy, :] = label_crop[:, cy, :]
            label_slice[:, :, cx] = label_crop[:, :, cx]

            # 保存
            case_id = os.path.basename(img_p).replace('_image.nii.gz', '')
            save_path = os.path.join(save_dir, f"{case_id}.h5")
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('image', data=image_norm.astype(np.float32), compression="gzip")
                f.create_dataset('label', data=label_crop.astype(np.uint8), compression="gzip")
                f.create_dataset('label_slice', data=label_slice, compression="gzip")
                f.create_dataset('idx', data=np.array([cz, cy, cx], dtype=np.int32))

    return np.array(centers)

def preprocess_test(image_paths, save_dir, global_center, crop_size=output_size):
    os.makedirs(save_dir, exist_ok=True)

    for img_p in tqdm(image_paths):
        img = nib.load(img_p)
        image_data = img.get_fdata()
        spacing = img.header.get_zooms()[:3]

        # 没有 label，但还是保持 pipeline 一致：重采样 + 归一化
        image_res = scipy.ndimage.zoom(image_data, np.array(spacing)/1.0, order=1)

        image_norm = normalize_image(image_res)

        # 根据训练集全局中心裁剪
        cz, cy, cx = global_center
        sz, sy, sx = crop_size
        z1, z2 = cz - sz//2, cz + sz//2
        y1, y2 = cy - sy//2, cy + sy//2
        x1, x2 = cx - sx//2, cx + sx//2

        image_crop = image_norm[max(z1,0):z2, max(y1,0):y2, max(x1,0):x2]

        # 保存
        case_id = os.path.basename(img_p).replace('_image.nii.gz', '')
        save_path = os.path.join(save_dir, f"{case_id}.h5")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('image', data=image_crop.astype(np.float32), compression="gzip")



if __name__ == "__main__":
    ###### 1. 数据处理
    train_path = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/MM-WHS 2017 Dataset/ct_train"
    test_path  = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/MM-WHS 2017 Dataset/ct_test"
    output_path = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing_WHS/MMWHS"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_train = os.path.join(output_path, "train_h5")
    save_test  = os.path.join(output_path, "test_h5")

    # ---------------- 训练集 ----------------
    # 找到所有 image 文件
    train_images = sorted([f for f in os.listdir(train_path) if f.endswith("_image.nii.gz")])
    train_labels = [f.replace("_image.nii.gz", "_label.nii.gz") for f in train_images]

    centers = preprocess_train(
        [os.path.join(train_path, f) for f in train_images],
        [os.path.join(train_path, f) for f in train_labels],
        save_train
    )

    global_center = np.mean(centers, axis=0).astype(int)
    print("Global center (from train) =", global_center)

    # ---------------- 测试集 ----------------
    test_images = sorted([f for f in os.listdir(test_path) if f.endswith("_image.nii.gz")])

    preprocess_test(
        [os.path.join(test_path, f) for f in test_images],
        save_test,
        global_center
    )


    # ####### 2. data_split #######
    # # path = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing_WHS/MMWHS/test_h5"
    # # list = os.listdir(path)
    # # print(len(list))
    #
    # import os
    # import random
    # out = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing_WHS/MMWHS"
    #
    # # 输入 H5 文件路径
    # data_dir = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing_WHS/MMWHS/train_h5"
    #
    # # 输出 txt 文件路径
    # output_train = os.path.join(out, "train_labeled_list.txt")
    # output_test = os.path.join(out, "test_list.txt")
    #
    # # 随机种子（保证可复现）
    # random.seed(42)
    #
    # # 获取所有 h5 文件
    # all_h5 = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
    # all_h5.sort()  # 固定顺序，保证一致性
    #
    # print(f"共找到 {len(all_h5)} 个 H5 文件")
    #
    # # 随机划分 8 / 12
    # train_files = random.sample(all_h5, 8)
    # test_files = [f for f in all_h5 if f not in train_files]
    #
    # # 保存到 txt
    # with open(output_train, "w") as f:
    #     f.write("\n".join(train_files))
    # with open(output_test, "w") as f:
    #     f.write("\n".join(test_files))
    #
    # print(f"✅ 已生成: {output_train} (标记数据 8 个)")
    # print(f"✅ 已生成: {output_test} (测试数据 12 个)")
    #
    # data_dir = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing/MMWHS/test_h5"
    # all_h5 = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
    # all_h5.sort()
    # output_train = os.path.join(out, "train_unlabeled_list.txt")
    # with open(output_train, "w") as f:
    #     f.write("\n".join(all_h5))
    # print(f"✅ 已生成: {output_train} (标记数据 8 个)")


