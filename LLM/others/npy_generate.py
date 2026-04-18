import nibabel as nib
import numpy as np
from skimage.transform import resize

# 读取 NIfTI 图像
nii_file_path = '/data/chenjinfeng/Data/CT_160/Xinan/image_CT/1_d.nii.gz'
img = nib.load(nii_file_path)
img_data = img.get_fdata()

# 假设 img_data 的初始形状为 (x, y, z)
# 我们将其调整为 (32, 256, 256) 的形状

# 首先调整大小以适应 32x256x256（可以根据实际数据调整顺序和尺寸）
target_shape = (32, 256, 256)
resized_img_data = resize(img_data, target_shape, mode='reflect', anti_aliasing=True)

# 添加批次维度以形成 (1, 32, 256, 256) 的形状
final_img_data = np.expand_dims(resized_img_data, axis=0)

# 进行 Min-Max 归一化至 0-1 之间
min_val = final_img_data.min()
max_val = final_img_data.max()
normalized_img_data = (final_img_data - min_val) / (max_val - min_val)

# 将数据保存为 .npy 格式
npy_file_path = '/data/chenjinfeng/Data/CT_160/Xinan/image_CT_npy/1_d.npy'
np.save(npy_file_path, normalized_img_data)

print(f"Processed image saved to {npy_file_path} with shape {normalized_img_data.shape}")