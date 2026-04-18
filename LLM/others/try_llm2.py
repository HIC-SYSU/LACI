
import os

# import contextlib
# from transformers import AutoModelForCausalLM
# model_path = "/data/chenjinfeng/code/VL/pre_weight/M3D"
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda:0")
# 
# 
# with open('M3D.txt', 'w') as f:
#     with contextlib.redirect_stdout(f):
#         print(model)

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 不在统一设备
device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.float16 # or bfloat16, float16, float32      # RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# # 加载模型和分词器
# model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D'
# proj_out_num = 256
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     torch_dtype=torch.float16,
#     device_map='auto',
#     trust_remote_code=True
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name_or_path,
#     model_max_length=512,
#     padding_side="right",
#     use_fast=False,
#     trust_remote_code=True
# )
#
# model = model.to(device=device)
# features = {}
#
# # 定义 Hook 函数
# def get_features_hook(name):
#     def hook(model, input, output):
#         features[name] = output.detach()  # 这里将输出特征存储到 features 字典中
#     return hook
#
# # 添加 hook 到指定层
# # 例如，访问 'model.layers.31.post_attention_layernorm' 层
# # layer = model.base_model.layers[31].post_attention_layernorm  # 访问模型中的某一层
# # layer.register_forward_hook(get_features_hook('post_attention_layernorm_31'))  # 注册钩子
# # layer = model.mm_projector
# # layer.register_forward_hook(get_features_hook('mm_projector'))  # 注册钩子
# layers_to_hook = [
#         'model.layers.1.post_attention_layernorm',
#         'model.layers.11.post_attention_layernorm',
#         'model.layers.21.post_attention_layernorm',
#         'model.layers.31.post_attention_layernorm',
#     ]
# features = {}
# for layer_name in layers_to_hook:
#     layer = dict([*model.named_modules()])[layer_name]
#     layer.register_forward_hook(partial(get_features, layer_name))
#
# # 构建输入
# question = 'Describe the size, lesions, and other details of the left atrium in this image'
# image_tokens = "<im_patch>" * proj_out_num
# input_txt = image_tokens + question
# input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
#
# # 假设 features_llm 是你的输入图像特征 (batch_size, channel, height, width, depth)
# features_llm = torch.randn(4, 1, 32, 256, 256).to(device=device, dtype=torch.float16)
# batch_size = features_llm.shape[0]
#
# # 存储生成的文本特征
# generated_texts = []
#
# # 生成并捕获特征
# for i in range(batch_size):
#     # 获取每张图像的特征
#     input_feature = features_llm[i:i + 1, :, :, :, :]  # 单张图像特征，保持维度
#
#     # 调用大语言模型进行生成，同时 hook 函数会捕获指定层的输出
#     generation, _ = model.generate(input_feature, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
#
# # 查看捕获的特征
# print(features['mm_projector'])
# print(features['mm_projector'].shape)



#
#     # 将生成的文本特征存储起来
#     generated_texts.append(generation)
#
# # import torch
# # import torch.nn.functional as F
# #
# # # 计算文本特征之间的不相似度（使用余弦相似度）
# # def cosine_dissimilarity(text1, text2):
# #     # 将文本特征归一化
# #     text1 = text1.float()
# #     text2 = text2.float()
# #     text1 = F.normalize(text1, dim=-1)
# #     text2 = F.normalize(text2, dim=-1)
# #     # 计算余弦相似度
# #     cosine_similarity = torch.sum(text1 * text2, dim=-1)
# #
# #     # 不相似度 = 1 - 相似度
# #     return 1.0 - cosine_similarity
# #
# # # 假设已经生成了文本特征 now stored in generated_texts
# # # 计算每对文本特征之间的不相似度
# # dissimilarities = []
# # for i in range(batch_size):
# #     for j in range(i + 1, batch_size):
# #         dissimilarity = cosine_dissimilarity(generated_texts[i], generated_texts[j])
# #         dissimilarities.append(dissimilarity)
# #
# # # 对每张图像进行映射处理
# # for i in range(batch_size):
# #     # 取出图像的深度、宽度和高度
# #     depth, height, width = features_llm[i, 0].shape
# #     # 创建一个与图像尺寸相同的不相似度映射
# #     dissimilarity_map = dissimilarities[i].item() * torch.ones(depth, height, width)
# #     # 将不相似度映射到图像的深度维度上
# #     features_llm[i, 0] += dissimilarity_map
# #
# # # 可视化映射结果，展示其中一张图像的某一层
# # for i in range(batch_size):
# #     plt.imshow(features_llm[i, 0, 16, :, :].numpy(), cmap='hot')  # 展示深度为16的图像层
# #     plt.title(f"Image {i+1} Dissimilarity Mapped")
# #     plt.colorbar()
# #     plt.show()


# 计算不相似位置的函数
# def calculate_top_dissimilar_positions(tensor1, tensor2, top_percentage=0.3):
#     # 计算逐元素差异
#     diff = tensor1 - tensor2
#     diff_abs = torch.abs(diff)
#     # 将差异展平为一维数组，并根据差异值排序
#     flattened_diff_abs = diff_abs.view(-1)
#     sorted_diff, indices = torch.sort(flattened_diff_abs, descending=True)
#     # 计算需要选取的不相似元素个数（前30%）
#     num_elements_to_select = int(top_percentage * flattened_diff_abs.numel())
#     # 选取前30%的位置
#     top_indices = indices[:num_elements_to_select]
#     mask = torch.zeros_like(diff)
#     mask.view(-1)[top_indices] =  True#torch.sign(diff.view(-1)[top_indices])
#     return mask, diff
#
# f_discribe = torch.randn(4, 1, 3, 3)
# batch_size = 4
# # 如果 size 为 2，计算两个特征的不相似区域并选取前30%
# if batch_size == 2:
#     top_dissimilar_mask, top_diff_values = calculate_top_dissimilar_positions(f_discribe[0], f_discribe[1], top_percentage=0.3)
#     print(f"Top 30% dissimilar regions between 1st and 2nd features:\n {top_dissimilar_mask}")
#     print(f"Top 30% difference values:\n {top_diff_values}")
#
# # 如果 size 为 4，计算 1 和 3，2 和 4 之间的不相似区域并选取前30%
# elif batch_size == 4:
#     top_dissimilar_mask_1_3, top_diff_values_1_3 = calculate_top_dissimilar_positions(f_discribe[0], f_discribe[2], top_percentage=0.3)
#     top_dissimilar_mask_2_4, top_diff_values_2_4 = calculate_top_dissimilar_positions(f_discribe[1], f_discribe[3], top_percentage=0.3)
#     f_discribe0_2 = top_dissimilar_mask_1_3 * f_discribe[2] + (1- top_dissimilar_mask_1_3)*f_discribe[0]      # ba保留了0中70%的
#     f_discribe2_0 = top_dissimilar_mask_1_3 * f_discribe[0] + (1 - top_dissimilar_mask_1_3) * f_discribe[2]
#     print(f'f_discribe2:\n {f_discribe[2]}, f_discribe0:\n {f_discribe[0]}')
#     print(f'mask:{top_dissimilar_mask_1_3}')
#     print(f'f_discribe2:\n {f_discribe0_2}, f_discribe0:\n {f_discribe2_0}')
#     # print(f"Top 30% dissimilar regions between 1st and 3rd features:\n {top_dissimilar_mask_1_3}")
#     # print(f"Top 30% difference values (1st vs 3rd):\n {top_diff_values_1_3}")
#     # print(f"Top 30% dissimilar regions between 2nd and 4th features:\n {top_dissimilar_mask_2_4}")
#     # print(f"Top 30% difference values (2nd vs 4th):\n {top_diff_values_2_4}")
#
# else:
#     print("Unsupported batch size for dissimilarity calculation.")


import torch
import torch.nn.functional as F

# 初始化 features 和 dismask
features = []
features.append(torch.randn(4, 16, 112, 112, 80))
features.append(torch.randn(4, 32, 56, 56, 40))
features.append(torch.randn(4, 64, 28, 28, 20))
features.append(torch.randn(4, 128, 14, 14, 10))
features.append(torch.randn(4, 256, 7, 7, 5))
# 初始化两个空列表来保存每个分割部分的结果
split_tensors_1 = []
split_tensors_2 = []
# 遍历列表，分割每个张量，并分别将结果添加到两个不同的列表中
for tensor in features:
    split_part_1, split_part_2 = torch.chunk(tensor, 2, dim=0)
    split_tensors_1.append(split_part_1)
    split_tensors_2.append(split_part_2)

# 输出分割后的张量尺寸
print("First set of split tensors:")
for tensor in split_tensors_1:
    print(tensor.size())

print("\nSecond set of split tensors:")
for tensor in split_tensors_2:
    print(tensor.size())

dis_masks = split_tensors_1
A_B_list = []
B_A_list = []
for dis_mask, a, b in zip(dis_masks, split_tensors_1, split_tensors_2):
    print(f'dis_mask: {dis_mask.shape}, a:{a.shape}, b:{b.shape}')
    A_B = (1 - dis_mask) * a + dis_mask * b
    B_A = (1 - dis_mask) * b + dis_mask * a
    A_B_list.append(A_B)
    B_A_list.append(B_A)
counterfactual_features = A_B_list + B_A_list
print(f'counterfactual_features: {counterfactual_features[0].shape}')
# A_B_tensor = torch.cat(A_B_list, dim=0)
# B_A_tensor = torch.cat(B_A_list, dim=0)
# counterfactual_features = torch.cat([A_B_tensor, B_A_tensor], dim=0)

# dismask = torch.randn(2, 1, 4096)  # 原始输入 (N, 1, 4096)
#
# # 假设我们希望将 4096 重新形状为 3D 形状 (16, 16, 16)
# dismask = dismask.view(2, 1, 16, 16, 16)
#
# # 动态调整 dismask 的形状以匹配每个 feature
# dis_masks = []
# for i, feature in enumerate(features):
#     # 获取当前 feature 的空间维度
#     _, C, D, H, W = feature.shape
#     # 将 dismask 插值到当前 feature 的空间大小
#     dis_mask_i = F.interpolate(dismask, size=(D, H, W), mode='trilinear', align_corners=False)
#     # 将通道数从 1 调整为与 feature 的通道数一致
#     dis_mask_i = dis_mask_i.expand(-1, C, -1, -1, -1)
#     # 保存调整后的 dismask
#     dis_masks.append(dis_mask_i)
#     print(f'dis_mask[{i}] shape: {dis_mask_i.shape}')

# import torch
# import torch.nn as nn
# # 初始化 features 和 dismask
# features = []
# features.append(torch.randn(4, 16, 112, 112, 80))
# features.append(torch.randn(4, 32, 56, 56, 40))
# features.append(torch.randn(4, 64, 28, 28, 20))
# features.append(torch.randn(4, 128, 14, 14, 10))
# features.append(torch.randn(4, 256, 7, 7, 5))
#
# dismask = torch.randn(2, 1, 4096)  # 原始输入 (N, 1, 4096)
#
# # 假设我们希望将 4096 重新形状为 3D 形状 (16, 16, 16)
# dismask = dismask.view(2, 1, 16, 16, 16)
#
#
# # 定义转置卷积模块，用于将 dismask 扩展到与 features 对应的大小
# class UpsampleWithConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(UpsampleWithConv, self).__init__()
#         self.trans_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
#
#     def forward(self, x):
#         return self.trans_conv(x)
#
#
# # 创建与 features 相匹配的转置卷积层
# upsamplers = [
#     UpsampleWithConv(1, 16, kernel_size=(2, 2, 2), stride=(7, 7, 5), padding=0),  # 对应 features[0]
#     UpsampleWithConv(1, 32, kernel_size=(2, 2, 2), stride=(4, 4, 3), padding=0),  # 对应 features[1]
#     UpsampleWithConv(1, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),  # 对应 features[2]
#     UpsampleWithConv(1, 128, kernel_size=(2, 2, 2), stride=(1, 1, 1), padding=0),  # 对应 features[3]
#     UpsampleWithConv(1, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)  # 对应 features[4]
# ]
#
# # 动态调整 dismask 的形状以匹配每个 feature
# dis_masks = []
# for i, feature in enumerate(features):
#     dis_mask_i = upsamplers[i](dismask)  # 使用转置卷积来调整大小
#     print(f'dis_mask[{i}] shape after transposed conv: {dis_mask_i.shape}')
#
#     # 将通道数从 1 调整为与 feature 的通道数一致（如果需要）
#     dis_masks.append(dis_mask_i)

# 现在 dis_masks 中的每个张量都已经通过卷积调整为与对应的 feature 大小匹配