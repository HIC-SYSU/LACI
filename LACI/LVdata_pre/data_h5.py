"""
不裁剪，预处理:归一化
"""

import random
import numpy as np
from tqdm import tqdm
import h5py
import nibabel as nib
import os

output_size = [256, 256, 128]

def covert_h5():
    image_CT_path = '/data/chenjinfeng/Data/CT_160/image_CT'
    label_CT_path = '/data/chenjinfeng/Data/CT_160/label_CT'
    norm_h5_path = '/data/chenjinfeng/Data/CT_160/norm_256_h5'
    # List the files in the directories
    image_CT_files = os.listdir(image_CT_path) if os.path.exists(image_CT_path) else []

    for item in tqdm(image_CT_files):
        image_data = nib.load(os.path.join(image_CT_path, item)).get_fdata()
        label_data = nib.load(os.path.join(label_CT_path, item)).get_fdata()

        image = (image_data - np.mean(image_data)) / np.std(image_data)
        image = image.astype(np.float32)

        # 处理文件名和保存
        h5_filename = os.path.join(norm_h5_path, item.replace('.nii.gz', '_norm.h5'))
        f = h5py.File(h5_filename, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label_data, compression="gzip")
        f.close()

covert_h5()



def h5_to_nii(h5_directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 列出所有H5文件
    h5_files = [f for f in os.listdir(h5_directory) if f.endswith('.h5')]

    for h5_file in tqdm(h5_files):
        h5_path = os.path.join(h5_directory, h5_file)

        # 读取H5文件
        with h5py.File(h5_path, 'r') as file:
            image = file['image'][:]
            label = file['label'][:]

        # 创建NIfTI图像
        image_nii = nib.Nifti1Image(image, affine=np.eye(4))
        label_nii = nib.Nifti1Image(label, affine=np.eye(4))

        # 输出文件路径
        image_output_path = os.path.join(output_directory, h5_file.replace('_norm.h5', '_image.nii.gz'))
        label_output_path = os.path.join(output_directory, h5_file.replace('_norm.h5', '_label.nii.gz'))

        # 保存为.nii.gz
        nib.save(image_nii, image_output_path)
        nib.save(label_nii, label_output_path)

# # 使用示例
# h5_directory = '/data/chenjinfeng/Data/CT_160/norm_h5'
# output_directory = '/data/chenjinfeng/Data/CT_160/h5_2_nii'
# h5_to_nii(h5_directory, output_directory)