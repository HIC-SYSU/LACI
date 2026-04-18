import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# Freezing model parameters
for param in model.parameters():
    param.requires_grad = False


for name, module in model.named_modules():  # 打印所有模块名称
    print(name)

features = {}
#
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# 添加多个层的钩子
layers_to_hook = [
    'model.layers.1.post_attention_layernorm',
    'model.layers.11.post_attention_layernorm',
    'model.layers.21.post_attention_layernorm',
    'model.layers.31.post_attention_layernorm',
]

for layer_name in layers_to_hook:
    if layer_name in dict(model.named_modules()):
        layer = dict([*model.named_modules()])[layer_name]
        layer.register_forward_hook(get_features(layer_name))
        print(f"Hooking layer {layer_name}")
    else:
        print(f"Layer {layer_name} not found in the model")

# Example input
question = "What is liver in this image? Please output the segmentation mask."
image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_ids = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

image_np = np.load(image_path)
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

# Forward pass to extract features and generate text and mask
output = model(input_ids=input_ids, images=image_pt)


##获取特征
if 'model.layers.1.post_attention_layernorm' in features:
    v1_proj_features = features['model.layers.1.post_attention_layernorm']
    print(f'v1_proj_features shape: {v1_proj_features.shape}')
else:
    print('v1_proj_features not found in features')

# if 'model.layers.11.post_attention_layernorm' in features:
#     v11_proj_features = features['model.layers.11.post_attention_layernorm']
#     print(f'v11_proj_features shape: {v11_proj_features.shape}')
# else:
#     print('v11_proj_features not found in features')
#
# if 'model.layers.21.post_attention_layernorm' in features:
#     v21_proj_features = features['model.layers.21.post_attention_layernorm']
#     print(f'v21_proj_features shape: {v21_proj_features.shape}')
# else:
#     print('v21_proj_features not found in features')
#
# if 'model.layers.31.post_attention_layernorm' in features:
#     v31_proj_features = features['model.layers.31.post_attention_layernorm']
#     print(f'v31_proj_features shape: {v31_proj_features.shape}')
# else:
#     print('v31_proj_features not found in features')


# generation, seg_logit = model.generate(image_pt, input_ids, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
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