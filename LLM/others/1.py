import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 设置数据类型和设备
# dtype = torch.float32  # 可根据需要更改数据类型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 加载 .h5 文件
# h5_file_path = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/0RZDK210BSMWAA6467LU/mri_norm2.h5"  # 替换为实际的 .h5 文件路径
# with h5py.File(h5_file_path, 'r') as h5_file:
#     image_np = h5_file['image'][:]
# # 将 NumPy 数组转换为 PyTorch 张量
# image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
# print("Original shape:", image_pt.shape)
# resized_tensor = F.interpolate(image_pt.unsqueeze(0), size=(256, 256, 80), mode='trilinear', align_corners=False) # 检查调整后的形状
# print("Resized spatial dimensions shape:", resized_tensor.shape)  # 输出应为 (1, 1, 256, 256, 80)
# # 进一步调整深度维度，从 80 到 32
# final_output = F.interpolate(resized_tensor, size=(256, 256, 32), mode='trilinear', align_corners=False).squeeze(0) # # 继续使用 interpolate，调整深度维度
# final_output = final_output.permute(0,3,1,2)
# # 检查最终输出形状
# print("Final output shape:", final_output.shape)



tensor = torch.Tensor(1, 70)
tensor1 = torch.Tensor(1, 60)
batch_size = 2
generated_texts = []
generated_texts.append(tensor)
generated_texts.append(tensor1)
def cosine_dissimilarity(text1, text2):
    # 将文本特征归一化
    text1 = text1.float()
    text2 = text2.float()
    text1 = F.normalize(text1, dim=-1)       
    text2 = F.normalize(text2, dim=-1)
    # 计算余弦相似度
    cosine_similarity = torch.sum(text1 * text2, dim=-1)
    # 不相似度 = 1 - 相似度
    return 1.0 - cosine_similarity

# 假设已经生成了文本特征 now stored in generated_texts
# 计算每对文本特征之间的不相似度
dissimilarities = []
for i in range(batch_size):
    for j in range(i + 1, batch_size):
        dissimilarity = cosine_dissimilarity(generated_texts[i], generated_texts[j])
        dissimilarities.append(dissimilarity)

# # 对每张图像进行映射处理
# for i in range(batch_size):
#     # 取出图像的深度、宽度和高度
#     depth, height, width = features_llm[i, 0].shape
#     # 创建一个与图像尺寸相同的不相似度映射
#     dissimilarity_map = dissimilarities[i].item() * torch.ones(depth, height, width)
#     # 将不相似度映射到图像的深度维度上
#     features_llm[i, 0] += dissimilarity_map