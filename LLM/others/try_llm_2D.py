import contextlib

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import os
import nibabel as nib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path_to_llava = '/data/chenjinfeng/code/VL/pre_weight/llava-v1.6-mistral-7b-hf'
processor = LlavaNextProcessor.from_pretrained(path_to_llava)       # , local_files_only=True
model = LlavaNextForConditionalGeneration.from_pretrained(path_to_llava, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda")

# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params, trainable_params
# ## 现在调用上面的函数来计算参数量 75亿
# total_params, trainable_params = count_parameters(model)
# print(f'Total parameters: {total_params}')
# print(f'Trainable parameters: {trainable_params}')


with open('LlavaNextForConditionalGeneration_architecture.txt', 'w') as f:
    with contextlib.redirect_stdout(f):
        print(model)
#############   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = "[INST] <image>\nThe people is tall [/INST]"
# inputs = processor(prompt, image, return_tensors="pt").to("cuda")     # 预处理图像和文本数据,并将其转换为模型可接受的形式
# # print(inputs.keys())        # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_sizes'])
# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)
# print(output)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)

#############   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### 2D输入
# image_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/M3D_data/016405/Axial_non_contrast/4.png'      ## image:(1003, 938)
# image = Image.open(image_path)
# print(f'image:{image.size}')
### 3D输入某个切片
image_path = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/Abdominal/Pancreas-CT/data/PANCREAS_0001.nii.gz'   ## image:(512, 512, 240)
image_data = nib.load(image_path).get_fdata()
slice_num = image_data.shape[2]//2
image_slice = image_data[:,:,slice_num]
image = Image.fromarray(image_slice.astype('uint8'))  # 确保数据类型正确
print(f'image:{image.size}')
# ### 将3D压缩成一个2D特征


prompt = "[INST] <image>\nDescribe the findings of the cardiac image you see. [/INST]"
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=1000)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)

#
# prompt = "[INST] <image>\nDescribe the findings of the MR pancreas image you see. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
# # print("image:", image.size, "prompt:", prompt)
#
#
# prompt = "[INST] <image>\nWhat is the findings of this MR pancreas image? [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
#
#
# prompt = "[INST] <image>\nDescribe this MR pancreas image with findings. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
#
#
# prompt = "[INST] <image>\nCan you summarize with findings the MR pancreas image presented? [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
#
# prompt = "[INST] <image>\nPlease provide a caption consists of findings for this MR pancreas image. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
#
# prompt = "[INST] <image>\nCan you provide a description consists of findings of this MR pancreas image? [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
#
# prompt = "[INST] <image>\nPlease caption this MR pancreas image with findings. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)



# prompt = "[INST] <image>\nDescribe the findings of the medical image you see. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
# # print("image:", image.size, "prompt:", prompt)
# 
# 
# prompt = "[INST] <image>\nWhat is the findings of this image? [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
# 
# 
# prompt = "[INST] <image>\nDescribe this medical scan with findings. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
# 
# 
# prompt = "[INST] <image>\nCan you summarize with findings the images presented? [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
# 
# prompt = "[INST] <image>\nPlease provide a caption consists of findings for this medical image. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
# 
# prompt = "[INST] <image>\nCan you provide a description consists of findings of this medical scan? [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)
# 
# prompt = "[INST] <image>\nPlease caption this scan with findings. [/INST]"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=1000)
# response = processor.decode(output[0], skip_special_tokens=True)
# print(response)


