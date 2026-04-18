import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py

from scipy import ndimage
import random
import itertools
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb
from torchvision import transforms
from torch.utils.data import DataLoader


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = case
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


# class LAHeart(Dataset):
#     """ LA Dataset """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []
#
#         train_path = self._base_dir+'/train.list'
#         test_path = self._base_dir+'/test.list'
#
#         if split=='train':
#             with open(train_path, 'r') as f:
#                 self.image_list = f.readlines()
#         elif split == 'test':
#             with open(test_path, 'r') as f:
#                 self.image_list = f.readlines()
#
#         self.image_list = [item.replace('\n','') for item in self.image_list]
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
#         # h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


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


class RandomCrop_LRV(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label, label_slice = sample['image'], sample['label_full'], sample['label3']
        name = sample['name']
        # print(label_slice.shape)
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        # print(label.shape)
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label_slice = np.pad(label_slice, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label_slice = label_slice[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf, 'label_slice': label_slice, "name": name}
        else:
            return {'image': image, 'label': label, 'label_slice': label_slice, "name": name}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomColorJitter(object):
    def __init__(self, color=(0.04, 0.04, 0.04, 0.01), p=0.1) -> None:
        self.color = color
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(low=0, high=1, size=1) > self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            for j in range(image.shape[0]):
                for t in range(image.shape[-1]):
                    image[j, :, :, :, t] = ColorJitter(
                        brightness=self.color[0],
                        contrast=self.color[1],
                        saturation=self.color[2],
                        hue=self.color[3])((image[j, :, :, :, t]))

            return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()} # , 'name': sample['name']
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()} # , 'name': sample['name']


class ToTensor_LRV(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        if 'name' not in sample:
            return {
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(sample['label']).long(),
                'label_slice': torch.from_numpy(sample['label_slice']).long()
            }
        else:
            name = sample['name']
            return {
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(sample['label']).long(),
                'label_slice': torch.from_numpy(sample['label_slice']).long(),
                'name': name
            }


class BatchSampler(Sampler):
    """Iterate on a sets of indices."""

    def __init__(self, primary_indices, batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        return (
            primary_batch
            for (primary_batch)
            in grouper(primary_iter, self.primary_batch_size)
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class TwoStreamBatchSampler_u(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_eternally(self.primary_indices)
        secondary_iter = iterate_once(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class ThreeStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + primary_batch
            for (primary_batch, secondary_batch, primary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



######################
class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train',train_flod=None, common_transform=None):
        self._base_dir = base_dir
        # self.h5_dir = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set'
        self.common_transform = common_transform
        self.sample_list = []
        print(train_flod)
        if split=='train':
            with open(self._base_dir+'/LA/'+train_flod, 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]

        print("total {} train_sample".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)
        # return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(image_name+"/mri_norm_new.h5", 'r')
        h5f = h5py.File(image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label, 'name':image_name}
        if self.common_transform:
            sample = self.common_transform(sample)
        return sample


class Pancreas(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir=None, train_flod=None, num=None, transform=None):
        self._base_dir = base_dir
        # self.h5_dir = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/Abdominal/Pancreas-CT'
        self.transform = transform
        self.sample_list = []

        with open(self._base_dir + '/Pancreas/Flods/' + train_flod, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]

        print("total {} unlabel_samples".format(len(self.image_list)))

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        # sample = {'image': image, 'label': label}
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class LVHeart(Dataset):
    """ LV Dataset """
    def __init__(self, base_dir=None, split='train',train_flod=None, common_transform=None):
        self._base_dir = base_dir
        # self.h5_dir = '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set'
        self.common_transform = common_transform
        self.sample_list = []
        print(train_flod)
        if split=='train':
            with open(self._base_dir+'/LV_112/'+train_flod, 'r') as f:
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

class LRVHeart(Dataset):
    """ LRV Dataset """
    def __init__(self, base_dir=None, split='train_labeled.txt', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        with open(os.path.join(self._base_dir, f"LRV_112/train_800_labeled_{split}"), 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('_norm.h5\n', '_slice_norm.h5') for item in self.image_list]
        # self.image_list = [item.replace('jinan_LRV_norm', 'jinan_LRV_norm_m15') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/" + image_name, 'r')
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_slice = h5f['label_slice'][:]
        slice_idx = h5f['slice_index'][:]
        h5f.close()
        sample = {'image': image, 'label_full': label, "label3": label_slice}
        # print(slice_idx)
        # print(f"idx={idx}, image.shape={image.shape}, label.shape={label.shape}")

        # # 创建一个全零的标签数组，形状与 label 相同
        # sparse_label = np.zeros_like(label, dtype=np.uint8)
        #
        # # # 分别取出 slice_idx 的三个方向索引
        # # axial_idx = slice_idx[0]    # 对应于 depth (D) 轴
        # # coronal_idx = slice_idx[1]  # 对应于 height (H) 轴
        # sagittal_idx = slice_idx[2] # 对应于 width (W) 轴
        #
        # # 复制三个方向的切片到稀疏标签中
        # # sparse_label[axial_idx, :, :] = label[axial_idx, :, :]
        # # sparse_label[:, coronal_idx, :] = label[:, coronal_idx, :]
        # sparse_label[:, :, sagittal_idx] = label[:, :, sagittal_idx]
        #
        # # print(f"idx={idx}, image.shape={image.shape}, label.shape={label.shape}, label_slice.shape={label_slice.shape}")
        # sample = {'image': image, 'label_full': label, 'label': sparse_label, "label3": label_slice}
        if self.transform:
            sample = self.transform(sample)
        # sample["slice_idx"] = slice_idx
        return sample


class LRVHeart_un(Dataset):
    """ LRV Dataset """
    def __init__(self, base_dir=None, split='train_unlabeled.txt', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        with open(os.path.join(self._base_dir, f"LRV_112/train_800_unlabeled_{split}"), 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('_norm.h5\n', '_slice_norm.h5') for item in self.image_list]
        # self.image_list = [item.replace('jinan_LRV_norm', 'jinan_LRV_norm_m15') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/" + image_name, 'r')
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        # label = h5f['label'][:]
        # label_slice = h5f['label_slice'][:]
        # new_label = np.zeros_like(image, dtype=np.uint8)
        # label = new_label
        # label_slice = new_label
        label = np.full_like(image, 255)
        label_slice = np.full_like(image, 255)
        # print(f"idx={idx}, image.shape={image.shape}, label.shape={label.shape}")
        h5f.close()

        sample = {'image': image, 'label_full': label, "label3": label_slice}
        if self.transform:
            sample = self.transform(sample)
        return sample
# train_data_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods'
# patch_size = [280, 280, 200]
# db_train = LVHeart(base_dir=train_data_path,
#                            split='train',
#                            train_flod='train.txt',  # todo change training flod
#                            common_transform=transforms.Compose([
#                                RandomCrop(patch_size),
#                                ToTensor()]))
#
# labeled_idxs = list(range(32))
# unlabeled_idxs = list(range(32, 323))
# batch_sampler = TwoStreamBatchSampler_u(labeled_idxs, unlabeled_idxs, 4, 2)
# trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
# for iteration, sampled_batch in enumerate(trainloader):
#     volume_batch, label_batch, name_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['name']
#     print(volume_batch.shape, label_batch.shape, name_batch)
class LRVHeart_pre(Dataset):
    """ LRV Dataset """
    def __init__(self, base_dir=None, split='train_labeled.txt', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        with open("/data/chenjinfeng/40180-data/chenjinfeng/Data_download/preprocessing/MMWHS_MR/train_pre.txt", 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        # self.image_list = [item.replace('jinan_LRV_norm', 'jinan_LRV_norm_m15') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        print(image_name)
        # h5f = h5py.File(self._base_dir + "/" + image_name, 'r')
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_slice = h5f['label_slice'][:]
        h5f.close()
        sample = {'image': image, 'label_full': label, "label3": label_slice}

        if self.transform:
            sample = self.transform(sample)
        # sample["slice_idx"] = slice_idx
        return sample


class LRVHeart_un_pre(Dataset):
    """ LRV Dataset """
    def __init__(self, base_dir=None, split='train_unlabeled.txt', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        with open("/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing/MMWHS_MR/train_pre.txt", 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        # self.image_list = [item.replace('_norm.h5\n', '_slice_norm.h5') for item in self.image_list]
        # self.image_list = [item.replace('jinan_LRV_norm', 'jinan_LRV_norm_m15') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/" + image_name, 'r')
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        # label = h5f['label'][:]
        # label_slice = h5f['label_slice'][:]
        # new_label = np.zeros_like(image, dtype=np.uint8)
        # label = new_label
        # label_slice = new_label
        label = np.full_like(image, 255)
        label_slice = np.full_like(image, 255)
        # print(f"idx={idx}, image.shape={image.shape}, label.shape={label.shape}")
        h5f.close()

        sample = {'image': image, 'label_full': label, "label3": label_slice}
        if self.transform:
            sample = self.transform(sample)
        return sample

class LRVImageCHDHeart(Dataset):
    """ LRV Dataset """
    def __init__(self, base_dir=None, split='train_labeled.txt', num=None, transform=None, keep_labels_012=False):
        self._base_dir = base_dir
        self.transform = transform
        self.keep_labels_012 = keep_labels_012
        self.sample_list = []

        with open(os.path.join(self._base_dir, f"{split}"), 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # image_name = image_name.replace(
        #     '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/preprocessing/MMWHS_MR/train_h5',
        #     "/data/xingshihanxiao/Pyproject/Contrast/datalist/data_mmwhs_shijie/train_h5"
        # )

        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_slice = h5f['label_slice'][:]
        slice_idx = h5f['idx'][:]
        h5f.close()

        # 只保留 0,1,2，其他类别置为 0
        if self.keep_labels_012:
            label = np.where(np.isin(label, [0, 1, 2]), label, 0).astype(np.uint8)
            label_slice = np.where(np.isin(label_slice, [0, 1, 2]), label_slice, 0).astype(np.uint8)

        # unique_labels = np.unique(label)
        # print(image_name, unique_labels)

        sample = {
            'image': image,
            'label_full': label,
            'label3': label_slice,
            'name': image_name
        }

        if self.transform:
            sample = self.transform(sample)
        return sample



class LRVImageCHDHeart_un(Dataset):
    """LRV unlabeled dataset"""
    def __init__(self, base_dir=None, split='train_unlabeled.txt', num=None,
                 transform=None, keep_labels_012=False):
        self._base_dir = base_dir
        self.transform = transform
        self.keep_labels_012 = keep_labels_012
        self.sample_list = []

        with open(os.path.join(self._base_dir, f"{split}"), 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_name = image_name.replace(
            '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/preprocessing/MMWHS_MR/train_h5',
            "/data/xingshihanxiao/Pyproject/Contrast/datalist/data_mmwhs_shijie/train_h5"
        )

        with h5py.File(image_name, 'r') as h5f:
            image = h5f['image'][:]

        # 无标签数据：这里只是占位
        # 如果后续不会参与 supervised CE / one-hot，可以保留 255
        # 如果后续可能进入 one_hot / CE，建议改成 0
        label = np.full(image.shape, 255, dtype=np.uint8)
        label_slice = np.full(image.shape, 255, dtype=np.uint8)

        # unique_labels = np.unique(label)
        # print(image_name, unique_labels)

        sample = {
            'image': image,
            'label_full': label,
            'label3': label_slice,
            'name': image_name
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

class LRVACDC(Dataset):
    """ LRV Dataset """
    def __init__(self, base_dir=None, split='train_labeled.txt', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        with open(os.path.join(self._base_dir, f"{split}"), 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_slice = h5f['label_slice'][:]
        # print(f"image shape: {image.shape}, label shape: {label.shape}, label slice: {label_slice.shape}")
        slice_idx = int(h5f['z_idx'][()])
        h5f.close()
        sample = {'image': image, 'label_full': label, "label3": label_slice}

        if self.transform:
            sample = self.transform(sample)
        return sample


class LRVACDC_un(Dataset):
    """ LRV Dataset """
    def __init__(self, base_dir=None, split='train_unlabeled.txt', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        with open(os.path.join(self._base_dir, f"{split}"), 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        # self.image_list = [item.replace('jinan_LRV_norm', 'jinan_LRV_norm_m15') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/" + image_name, 'r')
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['image'][:]
        label_slice = h5f['image'][:]
        h5f.close()

        sample = {'image': image, 'label_full': label, "label3": label_slice}
        if self.transform:
            sample = self.transform(sample)
        return sample



class AbdomenCT(Dataset):
    """ AbdomenCT Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/train.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/test.txt', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # print(image_name)
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label, 'name':image_name}
        if self.transform:
            sample = self.transform(sample)
        return sample
