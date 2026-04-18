# """
# 检查数据是否匹配
# """
#
# import os
# import nibabel as nib
# import numpy as np
#
# image_CT_path = '/data/chenjinfeng/Data/CT_160/image_CT'
# label_CT_path = '/data/chenjinfeng/Data/CT_160/label_CT'
#
# # List the files in the directories
# image_CT_files = os.listdir(image_CT_path) if os.path.exists(image_CT_path) else []
# label_CT_files = os.listdir(label_CT_path) if os.path.exists(label_CT_path) else []
#
# z_max = 0
# z_min = 500
# for i in image_CT_files:
#     image_data = nib.load(os.path.join(image_CT_path, i))
#     label_data = nib.load(os.path.join(label_CT_path, i))
#     _, _, z = image_data.shape
#     _, _, z1 = label_data.shape
#     if z!= z1:
#         print(i, z, z1)
#     print(image_data.shape, i)
# #     _,_,z = image_data.shape
# #     if z > z_max:
# #         z_max = z
# #     if z < z_min:
# #         z_min = z
# # print(z_max, z_min)
#
#

