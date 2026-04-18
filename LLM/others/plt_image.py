import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
#  ### 路径变量，需要根据实际情况调整
# image_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/M3D_data/016405/Axial_non_contrast/4.png'
# img = Image.open(image_path).convert('L')
# print(img.size)  # 输出图像大小
# img_resized = img.resize((80, 80), Image.BICUBIC)  # 使用双三次插值法进行缩放
# # 将图像转换为 numpy 数组，并重复 32 次
# # img_array = np.array(img_resized)
# # img_repeated = np.repeat(img_array[np.newaxis, np.newaxis, :, :], 32, axis=1)
# print(img_resized.size)
# # final_output = torch.from_numpy(img_repeated).unsqueeze(0).to(dtype=dtype, device=device)
#
# # 可视化原始图像和缩放后的图像
# plt.figure(figsize=(12, 6))
# # 显示原始图像
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')
#
# # 显示缩放后的图像
# plt.subplot(1, 2, 2)
# plt.imshow(img_resized, cmap='gray')
# plt.title('Resized Image (256x256)')
# plt.axis('off')
#
# plt.show()



# image_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/M3D_data/016405/processed_image.npy'
image_path = "/data/chenjinfeng/Data/CT_160/Xinan/image_CT_npy/1_d.npy"
image_np = np.load(image_path)
image_pt = torch.from_numpy(image_np)#.unsqueeze(0).to(dtype=dtype, device=device)
# final_output = image_pt
# 选择要显示的切片索引
# slice_index = 10  # 你可以根据具体情况修改这个索引
# # 使用matplotlib显示这个切片
# plt.subplot(2, 2, 1)
# plt.imshow(image_pt[0, 2].cpu().numpy(), cmap='gray')
# plt.title(f'Slice {slice_index}')
# plt.axis('off')
# # plt.show()
#
# plt.subplot(2, 2, 2)
# plt.imshow(image_pt[0, 12].cpu().numpy(), cmap='gray')
# plt.title(f'Slice {slice_index}')
# plt.axis('off')
# # plt.show()
#
#
# plt.subplot(2, 2, 3)
# plt.imshow(image_pt[0, 22].cpu().numpy(), cmap='gray')
# plt.title(f'Slice {slice_index}')
# plt.axis('off')
# # plt.show()
#
#
# plt.subplot(2, 2, 4)
# plt.imshow(image_pt[0, 31].cpu().numpy(), cmap='gray')
# plt.title(f'Slice {slice_index}')
# plt.axis('off')
# plt.show()

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(image_pt[0, 2*i].cpu().numpy(), cmap='gray')
    plt.title(f'Slice {2*i}')
    plt.axis('off')
plt.show()