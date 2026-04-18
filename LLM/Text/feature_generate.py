import os
from torchvision import transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm
####################################################dtype = torch.float16
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_process.pancreas import Pancreas

model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D'#'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

llm_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True)
features = {}

# 定义 Hook 函数
def get_features_hook(name):
    def hook(llm_model, input, output):
        features[name] = output.detach()  # 这里将输出特征存储到 features 字典中

    return hook


# 添加 hook 到指定层
layer_name = 'model.mm_projector.projector'
layer = eval(f'llm_model.{layer_name}')
layer.register_forward_hook(get_features_hook('mm_projector.projector'))  # 注册钩子

question = 'Describe the size, lesions, and other details of the left atrium in this image'
image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].cuda()  # .to(device=device)


features_dict = {}


train_data_path = '/data/chenjinfeng/code/semi_supervised/All_code/dataset'


### 数据集
patch_size =  [112, 112, 80]
db_train = Pancreas(base_dir=train_data_path,
                            train_flod='train0.txt',  # todo change training flod
                            transform=transforms.Compose([RandomCrop(patch_size), ToTensor()]))

file_path = '/data/chenjinfeng/code/VL/demo/demo/features_dict_pancreas.pth'
if not os.path.exists(file_path):
    os.makedirs(file_path)

for i_batch, sampled_batch in enumerate(tqdm(db_train)):
    volume_batch = sampled_batch['image'].cuda()
    name = sampled_batch['name']
    # print(f'ids:{name}')
    # print(volume_batch.shape)
    volume_batch = volume_batch.unsqueeze(1)
    resized_feature = F.interpolate(volume_batch.permute(0, 1, 4, 2, 3), size=(32, 256, 256), mode='trilinear', align_corners=False)
    generation, _ = llm_model.generate(resized_feature.to(dtype=torch.float16), input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
    features_dict[name] = features['mm_projector.projector']
torch.save(features_dict, file_path)
print(f"Tensor saved to {file_path}")