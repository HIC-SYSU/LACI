import glob
import nibabel as nib
import os
import numpy as np
label_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/Abdominal/Pancreas-CT/label/*'
save_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/Abdominal/Pancreas-CT/label_T/'
label_list = glob.glob(label_path)
print(label_list)
for label_file in label_list:
    name = label_file.split('/')[-1]
    label_img = nib.load(label_file)
    label_data = label_img.get_fdata()
    print(f'label_data.shape: {label_data.shape}')
    # # 创建一个空数组，用于存储转置后的切片
    # transposed_slices = np.empty((label_data.shape[0], label_data.shape[1], label_data.shape[2]))
    # 
    # # 遍历每个切片，对其进行转置，并存储
    # for i in range(label_data.shape[2]):
    #     transposed_slices[:, :, i] = label_data[:, :, i].T
    # 
    # transposed_slices = transposed_slices[:, :, ::-1]
    # print(f'transposed_slices.shape: {transposed_slices.shape}')
    # 创建一个新的 Nifti1Image 对象，使用转置后的数据和原始的仿射矩阵
    
    label_data_flip = np.flip(label_data, axis=2) 
    new_img = nib.Nifti1Image(label_data_flip, affine=label_img.affine)
    nib.save(new_img, os.path.join(save_path, name))
    