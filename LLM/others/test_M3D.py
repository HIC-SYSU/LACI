############################################################### 1. 测试数据生成
# import numpy as np
# import os
# from PIL import Image
# import skimage.transform as sktrans
#
# def load_pngs_as_3d_array(path, target_depth=32, target_height=256, target_width=256):
#     # 获取路径下的所有PNG文件
#     files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')])
#     # 读取和处理每一个图像为灰度
#     images = [np.array(Image.open(f).convert('L')) for f in files]
#     # 将图像堆叠成3D数组
#     stack = np.stack(images, axis=0)
#     # 调整尺寸以匹配目标形状
#     resized_stack = sktrans.resize(stack, (target_depth, target_height, target_width), order=1, mode='reflect', cval=0, anti_aliasing=True)
#     print(f'resized stack: {resized_stack.shape}')
#     # 归一化处理
#     norm_stack = (resized_stack - np.min(resized_stack)) / (np.max(resized_stack) - np.min(resized_stack))
#     # 增加一个额外的维度以符合模型输入要求
#     norm_stack = norm_stack[np.newaxis, ...]
#     return norm_stack
#
# # 调用函数
# path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/M3D_data/016405/Axial_non_contrast'
# processed_image = load_pngs_as_3d_array(path)
# # 保存为.npy文件
# np.save('/data/chenjinfeng/40180-data/chenjinfeng/Data_download/M3D_data/016405/processed_image.npy', processed_image)

###############################################################2. 测试
import os
import numpy as np
import torch
# import bleach
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 不在统一设备
device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.float16 # or bfloat16, float16, float32      # RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'

# model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D'#'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.


model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params, trainable_params
# 现在调用上面的函数来计算参数量
# total_params, trainable_params = count_parameters(model)
# print(f'Total parameters: {total_params}')
# print(f'Trainable parameters: {trainable_params}')
# Total parameters: 6983026640
# Trainable parameters: 6983026640

# question = "Can you provide a caption consists of findings for this medical image?"
# question = "What is left atrium in this image? Please output the segmentation mask and describe the details of the left atrium?"
# question = "What is liver in this image? Please output the box."
# question = 'What is left atrium in this image? Can you describe the details of the left atrium in this medical image?'

image_tokens = "<im_patch>" * proj_out_num
# image_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/M3D_data/016405/processed_image.npy'
# # image_path = "/data/chenjinfeng/Data/CT_160/Xinan/image_CT_npy/1_d.npy"
# image_np = np.load(image_path)
# image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
# final_output = image_pt

from PIL import Image
import numpy as np

# 路径变量，需要根据实际情况调整
image_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/M3D_data/016405/Axial_non_contrast/4.png'
img = Image.open(image_path).convert('L')
img_resized = img.resize((256, 256), Image.BICUBIC)  # 使用双三次插值法进行缩放
# 将图像转换为 numpy 数组，并重复 32 次
img_array = np.array(img_resized)
img_repeated = np.repeat(img_array[np.newaxis, np.newaxis, :, :], 32, axis=1)
print(img_repeated.shape)
final_output = torch.from_numpy(img_repeated).unsqueeze(0).to(dtype=dtype, device=device)
import matplotlib.pyplot as plt
# 可视化原始图像和缩放后的图像
plt.figure(figsize=(12, 6))
# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示缩放后的图像
plt.subplot(1, 2, 2)
plt.imshow(img_resized, cmap='gray')
plt.title('Resized Image (256x256)')
plt.axis('off')

plt.show()


# input_str = 'Is the pancreas included in the image? '
# # input_str = bleach.clean(input_str)
# prompt = "<im_patch>" * proj_out_num + input_str
# input_id = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device=device)
# generation = model.generate(image_pt, input_id, seg_enable=False, max_new_tokens=256, do_sample=False, temperature=0.1)
# ## do_sample用于决定生成文本时是否采用随机采样，启用：模型在选择下一令牌时从概率分布中采样，而不是简单的选择最高概率令牌
# ## top_p与概率阈值相关的参数，用于实施核采样策略，这意味着模型在生成每个新令牌时，只从累计概率高于top_p的令牌中进行选择。 ## do_sample=False时，top_p参数无效。
# ## temperature参数用于控制生成的文本的随机性，值越高，生成的文本越随机。## do_sample=True时，temperature参数无效。
# output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
# print("output_str", output_str)



question1 = 'What is the findings of this image?'       ##Describe the findings of the medical image you see.'
input_txt1 = image_tokens + question1
input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
generation = model.generate(final_output, input_id1, seg_enable=False, max_new_tokens=256, do_sample=False)
generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
print('question', question1)
print('generated_texts:', generated_texts[0])
##############################################################################
# question1 = 'Describe the findings of the medical image you see.'       ##'
question1 = 'Describe the findings of the cardiac image you see.'
input_txt1 = image_tokens + question1
input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
generation = model.generate(final_output, input_id1, seg_enable=False, max_new_tokens=1000, do_sample=False)
generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
print('question', question1)
print('generated_texts:', generated_texts[0])

question1 = 'Describe the findings of the medical image you see.'
input_txt1 = image_tokens + question1
input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
generation = model.generate(final_output, input_id1, seg_enable=False, max_new_tokens=1000, do_sample=False)
generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
print('question', question1)
print('generated_texts:', generated_texts[0])


# print(f'input_id device:{input_id1.shape}')
# print(f'image_pt device:{final_output.shape}')
# print(f'dtype:{final_output.dtype}')
# print(f'model device:{model.device}')

# generation = model.generate(image_pt, input_id1, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generation, seg_logit = model.generate(final_output, input_id1, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
# print('question', question1)
# print('generated_texts:', generated_texts[0])
#
# question1 = 'What is the findings of this image?'
# input_txt1 = image_tokens + question1
# input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
# generation, seg_logit = model.generate(final_output, input_id1, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
#
# print('question', question1)
# print('generated_texts:', generated_texts[0])
#
# question1 = 'Describe this medical scan with findings.'
# input_txt1 = image_tokens + question1
# input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
# generation, seg_logit = model.generate(final_output, input_id1, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
#
# print('question', question1)
# print('generated_texts:', generated_texts[0])
#
# question1 = 'Can you summarize with findings the images presented?'
# input_txt1 = image_tokens + question1
# input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
# generation, seg_logit = model.generate(final_output, input_id1, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# print('question', question1)
# print('generated_texts:', generated_texts[0])
#
# question1 = 'Please provide a caption consists of findings for this medical image.'
# input_txt1 = image_tokens + question1
# input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
# generation, seg_logit = model.generate(final_output, input_id1, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# print('question', question1)
# print('generated_texts:', generated_texts[0])
#
#
# question1 = 'Can you provide a description consists of findings of this medical scan?'
# input_txt1 = image_tokens + question1
# input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
# generation, seg_logit = model.generate(final_output, input_id1, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# print('question', question1)
# print('generated_texts:', generated_texts[0])
#
# question1 = 'Please caption this scan with findings.'
# input_txt1 = image_tokens + question1
# input_id1 = tokenizer(input_txt1, return_tensors="pt")['input_ids'].to(device=device)
# generation, seg_logit = model.generate(final_output, input_id1, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# print('question', question1)
# print('generated_texts:', generated_texts[0])


