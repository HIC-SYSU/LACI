import h5py
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
####################################################dtype = torch.float16
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D'#'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

### 加载数据集
######################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class LVHeart(Dataset):
    """ LV Dataset """
    def __init__(self, base_dir=None, split='train',train_flod=None, common_transform=None):
        self._base_dir = base_dir
        # self.h5_dir = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set'
        self.common_transform = common_transform
        self.sample_list = []
        print(train_flod)
        if split=='train':
            with open(self._base_dir+train_flod, 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("total {} unlabel_samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)
        # return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label, 'name':image_name}
        if self.common_transform:
            sample = self.common_transform(sample)
        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf, 'name': name}
        else:
            return {'image': image, 'label': label, 'name': name}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'name': sample['name'],
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'name': sample['name']}


train_data_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/'
db_train = LVHeart(base_dir=train_data_path,
                           split='train',
                           train_flod='train_all.txt',  # todo change training flod
                           common_transform=transforms.Compose([
                               ToTensor()]))


# 第一步：预先提取所有特征
file_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/LV_features_dict.pth'
text_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/LV_text.txt'

# if not os.path.isfile(file_path):
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

question = 'Describe the size, shape, brightness, lesions and other details of the left ventricle and myocardium in this image'
image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].cuda()  # .to(device=device)

features_dict = {}
generated_texts_dict = {}
with open(text_path, 'w', encoding='utf-8') as f:
    for i_batch, sampled_batch in enumerate(tqdm(db_train)):
        volume_batch = sampled_batch['image'].cuda()
        name = sampled_batch['name']
        # print(f'ids:{name}')
        # print(volume_batch.shape)
        volume_batch = volume_batch.unsqueeze(1)
        resized_feature = F.interpolate(volume_batch.permute(0, 1, 4, 2, 3), size=(32, 256, 256), mode='trilinear', align_corners=False)
        generation, _ = llm_model.generate(resized_feature.to(dtype=torch.float16), input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
        features_dict[name] = features['mm_projector.projector']
        # features_dict[name] = generation
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
        # 将每个生成的文本与图像名称一起写入文件
        for text in generated_texts:
            f.write(f"{name}: {text}\n")  # 格式化为 "图像名称: 生成的文本"
        print(name, generation.shape, generated_texts)
torch.save(features_dict, file_path)


