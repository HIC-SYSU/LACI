# import contextlib
#
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
# from PIL import Image
# import requests
#
# path_to_llava = '/data/chenjinfeng/code/VL/pre_weight/llava-v1.6-mistral-7b-hf'
# processor = LlavaNextProcessor.from_pretrained(path_to_llava)       # , local_files_only=True
# model = LlavaNextForConditionalGeneration.from_pretrained(path_to_llava, torch_dtype=torch.float16, low_cpu_mem_usage=True)
# model.to("cuda:0")
#
# with open('LlavaNextForConditionalGeneration_architecture.txt', 'w') as f:
#     with contextlib.redirect_stdout(f):
#         print(model)
# # prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = "[INST] <image>\nThe people is tall [/INST]"
#
# # print("image:", image.size, "prompt:", prompt)
# inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")     # 预处理图像和文本数据,并将其转换为模型可接受的形式
# # print(inputs.keys())        # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_sizes'])
#
# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)
# print(output)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
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
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 不在统一设备
device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.float16 # or bfloat16, float16, float32      # RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D'#'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
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
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

model = model.to(device=device)

# question = "Can you provide a caption consists of findings for this medical image?"
# question = "What is left atrium in this image? Please output the segmentation mask and describe the details of the left atrium?"
# question = "What is liver in this image? Please output the box."
# question = 'What is left atrium in this image? Can you describe the details of the left atrium in this medical image?'
question = 'Describe the size, lesions, and other details of the left atrium in this image'
image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

##################################################################################image1
import h5py
import numpy as np
import torch
import torch.nn as nn
# 设置数据类型和设备
dtype = torch.float32  # 可根据需要更改数据类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载 .h5 文件
h5_file_path = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/0RZDK210BSMWAA6467LU/mri_norm2.h5"  # 替换为实际的 .h5 文件路径
with h5py.File(h5_file_path, 'r') as h5_file:
    image_np = h5_file['image'][:]
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())        # 归一化
# 将 NumPy 数组转换为 PyTorch 张量
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
# print("Original shape:", image_pt.shape)
resized_tensor = F.interpolate(image_pt.unsqueeze(0), size=(256, 256, 80), mode='trilinear', align_corners=False) # 检查调整后的形状
# print("Resized spatial dimensions shape:", resized_tensor.shape)  # 输出应为 (1, 1, 256, 256, 80)
# 进一步调整深度维度，从 80 到 32
final_output = F.interpolate(resized_tensor, size=(256, 256, 32), mode='trilinear', align_corners=False) # # 继续使用 interpolate，调整深度维度
final_output = final_output.permute(0,1,4,2,3).to(dtype=torch.float16)    # .cpu().numpy()        # # 转换为 NumPy 数组

# generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
generation1, seg_logit = model.generate(final_output, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

generated_texts = tokenizer.batch_decode(generation1, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
print(generation1.shape)
print('question', question)
print('generated_texts:', generated_texts[0])
##################################################################################image2
h5_file_path = "/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/CBIJFVZ5L9BS0LKWE8YL/mri_norm2.h5"  # 替换为实际的 .h5 文件路径
with h5py.File(h5_file_path, 'r') as h5_file:
    image_np = h5_file['image'][:]
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())        # 归一化
# 将 NumPy 数组转换为 PyTorch 张量
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
# print("Original shape:", image_pt.shape)
resized_tensor = F.interpolate(image_pt.unsqueeze(0), size=(256, 256, 80), mode='trilinear', align_corners=False) # 检查调整后的形状
# print("Resized spatial dimensions shape:", resized_tensor.shape)  # 输出应为 (1, 1, 256, 256, 80)
# 进一步调整深度维度，从 80 到 32
final_output = F.interpolate(resized_tensor, size=(256, 256, 32), mode='trilinear', align_corners=False) # # 继续使用 interpolate，调整深度维度
final_output = final_output.permute(0,1,4,2,3).to(dtype=torch.float16)    # .cpu().numpy()        # # 转换为 NumPy 数组

# generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
generation2, seg_logit = model.generate(final_output, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

generated_texts = tokenizer.batch_decode(generation2, skip_special_tokens=True)
print(generation2.shape)
print('generated2_texts:', generated_texts[0])
# generation1 = generation1.float()
# generation2 = generation2.float()
if generation1.size(1)<generation2.size(1):
    generation1 = F.interpolate(generation1.float().unsqueeze(1), size=generation2.size(1), mode='linear', align_corners=False).squeeze(0)
    difference = generation1.long() - generation2
    different = torch.abs(difference)
    different = torch.round(different)
    print(different.shape)
    generated_diff_texts = tokenizer.batch_decode(different.to(dtype=torch.float16), skip_special_tokens=True)
    print('generated_diff_texts:', generated_diff_texts[0])
    generated_diff_texts = tokenizer.batch_decode((generation2 + difference).to(dtype=torch.float16), skip_special_tokens=True)
    print('generated_diff2_texts:', generated_diff_texts[0])

else:
    print(generation2.dtype)
    generation2 = F.interpolate(generation2.float().unsqueeze(1), size=generation1.size(1), mode='linear', align_corners=False).squeeze(0)#.to(torch.long)
    print(generation2.dtype)
    difference = generation1 - generation2.long()
    different = torch.abs(difference)
    different = torch.round(different)
    print(different.shape)
    generated_diff_texts = tokenizer.batch_decode(different.long(), skip_special_tokens=True)
    print('generated_diff_texts:', generated_diff_texts[0])

    is_equal = torch.allclose(generation1 - difference, generation2, atol=1e-5)
    print("Are generation1 - difference and generation2 approximately equal?", is_equal)
    # 打印差异值
    print("Difference between generation1 - difference and generation2:", (generation1 - difference) - generation2)

    count = generation1 - difference
    generated_diff_texts = tokenizer.batch_decode(count, skip_special_tokens=True)
    print('generated_diff1_texts:', generated_diff_texts[0])



# difference = difference>0




