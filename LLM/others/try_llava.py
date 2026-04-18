import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda')  # 'cpu', 'cuda'
dtype = torch.float16  # or bfloat16, float16, float32

model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D'  # 'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
image_path = "/data/chenjinfeng/Data/CT_160/Xinan/image_CT_npy/1_d.npy"

llm_model = AutoModelForCausalLM.from_pretrained(
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

llm_model = llm_model.to(device=device)

# Freezing model parameters
for param in llm_model.parameters():
    param.requires_grad = False


total_params = sum(p.numel() for p in llm_model.parameters())
trainable_params = sum(p.numel() for p in llm_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


# 打印所有模块名称以确保你找到了正确的模块
for name, module in llm_model.named_modules():  # 打印所有模块名称
    print(name)

features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# 添加多个层的钩子
layers_to_hook = [
    'model.vision_tower.vision_tower.blocks.5',       #第0层：transformer编码器块，每一层都会进一步处理图像特征，特别是最后几层特征包含较高的语义信息
    'model.vision_tower.vision_tower.blocks.5.norm2',       # 经过所有编码块处理后的归一化层，相当于提取最终的图像特征，这些特征经常用于下游任务，如分类或生成
    'model.layers.0.post_attention_layernorm',
    'model.mm_projector.projector.2',
    'model.mm_projector'
]

for layer_name in layers_to_hook:
    if layer_name in dict(llm_model.named_modules()):
        layer = dict([*llm_model.named_modules()])[layer_name]
        layer.register_forward_hook(get_features(layer_name))
        print(f"Hooking layer {layer_name}")
    else:
        print(f"Layer {layer_name} not found in the model")

# # Example input
# question = "What is liver in this image? Please output the segmentation mask."
# image_tokens = "<im_patch>" * proj_out_num
# input_txt = image_tokens + question
# input_ids = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
# image_np = np.load(image_path)
# image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)       # input_ids: torch.Size([1, 272]), image_pt: torch.Size([1, 1, 32, 256, 256])
# # Forward pass to extract features and generate text and mask
# print(f'input_ids: {input_ids.shape}, image_pt: {image_pt.shape}')
# output = model(input_ids=input_ids, images=image_pt)
#
# # 获取特征
# if 'model.vision_tower.vision_tower.blocks.5' in features:
#     block_5_features = features['model.vision_tower.vision_tower.blocks.5']
#     print(f'block_5_features shape: {block_5_features.shape}')
# else:
#     print('block_5_features not found in features')
#
# if 'model.vision_tower.vision_tower.blocks.5.norm2' in features:
#     norm2_features = features['model.vision_tower.vision_tower.blocks.5.norm2']
#     print(f'norm2_features shape: {norm2_features.shape}')
# else:
#     print('norm2_features not found in features')
#
#
# if 'model.layers.0.post_attention_layernorm' in features:
#     norm2_features = features['model.layers.0.post_attention_layernorm']
#     print(f'norm2_features shape: {norm2_features.shape}')
# else:
#     print('norm2_features not found in features')
# if 'model.layers.0' in features:
#     layers_0_features = features['model.layers.0']
#     print(f'model.layers.0: {layers_0_features.shape}')
# else:
#     print('layers_0_features not found in features')

# # 生成文本和分割掩码
# generation, seg_logit = model.generate(input_ids=input_ids, images=image_pt, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
# generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
#
# print('question', question)
# print('generated_texts', generated_texts[0])
#
# image = sikt.GetImageFromArray(image_np)
# ssv.display(image)
# seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
# ssv.display(seg)

