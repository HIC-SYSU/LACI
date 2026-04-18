import os
import time
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D'
proj_out_num = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(sample['label']).long(),
            'name': sample['name']
        }


class LAHeart(Dataset):
    def __init__(self, base_dir=None, split='train', train_flod=None, common_transform=None):
        self._base_dir = base_dir
        self.common_transform = common_transform

        if split == 'train':
            with open(os.path.join(self._base_dir, train_flod), 'r') as f:
                self.image_list = [item.strip() for item in f.readlines()]
        else:
            raise ValueError(f'Unsupported split: {split}')

        print(f"total {len(self.image_list)} samples")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        with h5py.File(os.path.join(image_name, "mri_norm_new.h5"), 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        sample = {'image': image, 'label': label, 'name': image_name}
        if self.common_transform:
            sample = self.common_transform(sample)
        return sample


def build_loader():
    train_data_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/'
    db_train = LAHeart(
        base_dir=train_data_path,
        split='train',
        train_flod="All_data.txt",
        common_transform=transforms.Compose([ToTensor()])
    )
    loader = DataLoader(
        db_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader


def load_m3d():
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    torch.cuda.synchronize()
    load_time = time.perf_counter() - t0
    return llm_model, tokenizer, load_time


def run_extraction_cost_only():
    stats_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/LA_extraction_cost_only.json'

    loader = build_loader()
    llm_model, tokenizer, model_load_time = load_m3d()

    features = {}

    def get_features_hook(name):
        def hook(module, inputs, output):
            features[name] = output.detach()
        return hook

    layer_name = 'model.mm_projector.projector'
    layer = eval(f'llm_model.{layer_name}')
    hook_handle = layer.register_forward_hook(get_features_hook('mm_projector.projector'))

    question = "Describe the left atrium in this cardiac MRI image, including its shape, boundary, intensity, and surrounding structures."
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device)

    per_case_times = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    extraction_start = time.perf_counter()

    with torch.no_grad():
        for _, sampled_batch in enumerate(tqdm(loader, desc='Extracting embeddings only')):
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)

            if volume_batch.ndim == 4:
                volume_batch = volume_batch.unsqueeze(1)

            torch.cuda.synchronize()
            case_t0 = time.perf_counter()

            resized_feature = F.interpolate(
                volume_batch.permute(0, 1, 4, 2, 3),
                size=(32, 256, 256),
                mode='trilinear',
                align_corners=False
            )

            _generation, _ = llm_model.generate(
                resized_feature.to(dtype=torch.float16),
                input_id,
                seg_enable=True,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=1.0
            )

            _ = features['mm_projector.projector']

            torch.cuda.synchronize()
            case_time = time.perf_counter() - case_t0
            per_case_times.append(case_time)

    torch.cuda.synchronize()
    extraction_total_time = time.perf_counter() - extraction_start
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    hook_handle.remove()

    stats = {
        'num_cases': len(per_case_times),
        'model_load_time_sec': model_load_time,
        'extraction_total_time_sec': extraction_total_time,
        'extraction_avg_time_per_case_sec': float(np.mean(per_case_times)) if per_case_times else 0.0,
        'extraction_std_time_per_case_sec': float(np.std(per_case_times)) if per_case_times else 0.0,
        'extraction_peak_gpu_mem_mb': peak_gpu_mem_mb,
    }

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print('\n=== Extraction Cost Only ===')
    print(f"num_cases: {stats['num_cases']}")
    print(f"model_load_time_sec: {stats['model_load_time_sec']:.4f}")
    print(f"extraction_total_time_sec: {stats['extraction_total_time_sec']:.4f}")
    print(f"extraction_avg_time_per_case_sec: {stats['extraction_avg_time_per_case_sec']:.4f}")
    print(f"extraction_std_time_per_case_sec: {stats['extraction_std_time_per_case_sec']:.4f}")
    print(f"extraction_peak_gpu_mem_mb: {stats['extraction_peak_gpu_mem_mb']:.2f}")
    print(f"saved stats to: {stats_path}")

    return stats


if __name__ == '__main__':
    run_extraction_cost_only()