import os
import h5py
import SimpleITK as sitk
import numpy as np

def convert_h5_to_nifti(h5_path, nifti_path):
    # 打开HDF5文件
    with h5py.File(h5_path, 'r') as f:
        # 读取图像和标签数据
        image = f['image'][:]
        label = f['label'][:]
        print('Image shape:', image.shape)

    # 将numpy数组转换为SimpleITK的图像格式
    sitk_image = sitk.GetImageFromArray(image)
    sitk_label = sitk.GetImageFromArray(label)

    # 保存图像和标签为NIfTI格式
    sitk.WriteImage(sitk_image, nifti_path.replace('.nii.gz', '_image.nii.gz'), True)
    sitk.WriteImage(sitk_label, nifti_path.replace('.nii.gz', '_label.nii.gz'), True)

def batch_convert_h5_to_nifti(h5_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历指定文件夹内所有的HDF5文件
    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5'):
            h5_path = os.path.join(h5_folder, filename)
            nifti_path = os.path.join(output_folder, filename.replace('.h5', '.nii.gz'))
            convert_h5_to_nifti(h5_path, nifti_path)
            print(f'Converted {filename} to NIfTI format')

if __name__ == '__main__':
    h5_folder = '/data/chenjinfeng/Data/CT_160/Xinan/h5_data'  # 指定HDF5文件所在的文件夹
    output_folder = '/data/chenjinfeng/Data/CT_160/Xinan/h5nii'  # 指定输出NIfTI文件的文件夹
    batch_convert_h5_to_nifti(h5_folder, output_folder)
