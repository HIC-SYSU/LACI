import random

import numpy as np
from tqdm import tqdm
import h5py
import nrrd
import glob
import os


output_size =[112, 112, 80]

def covert_h5():
    listt = glob('../data/LA/2018LA_Seg_Training Set/*/lgemri.nrrd')
    for item in tqdm(listt):
        image, img_header = nrrd.read(item)
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        f = h5py.File(item.replace('lgemri.nrrd', 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()


def covert_h5_new():
    listt = glob('/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/*/lgemri.nrrd')
    for item in tqdm(listt):
        image, img_header = nrrd.read(item)
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        f = h5py.File(item.replace('lgemri.nrrd', 'mri_norm_new.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()


def split_data():
    all_path = glob.glob(
        '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/*')
    print("Total number of cases:", len(all_path))

    text_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/'

    # 检查目录是否存在
    if not os.path.exists(text_path):
        os.makedirs(text_path)
    else:
        print("Directory already exists. Stopping execution.")
        return

    # 保存训练和测试路径到TXT文件
    for i in range(5):
        # 随机选择训练集路径
        select_path_train = random.sample(all_path, 80)
        # 剩余的路径作为测试集
        remaining_list = [item for item in all_path if item not in select_path_train]
        select_path_test = random.sample(remaining_list, 20)

        train_file_path = os.path.join(text_path, f'train{i}.txt')
        test_file_path = os.path.join(text_path, f'test{i}.txt')

        # 写入训练路径
        with open(train_file_path, 'w') as f:
            for item in select_path_train:
                f.write(item + '\n')

        # 写入测试路径
        with open(test_file_path, 'w') as f:
            for item in select_path_test:
                f.write(item + '\n')

        print(f'Fold {i} files saved: {train_file_path}, {test_file_path}')

def all_data():
    all_path = glob.glob(
        '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/*')
    print("Total number of cases:", len(all_path))

    text_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/'

    # # 检查目录是否存在
    # if not os.path.exists(text_path):
    #     os.makedirs(text_path)
    # else:
    #     print("Directory already exists. Stopping execution.")
    #     return


    train_file_path = os.path.join(text_path, f'All_data.txt')

    # 写入训练路径
    with open(train_file_path, 'w') as f:
        for item in all_path:
            f.write(item + '\n')


        print(f' files saved: {train_file_path}')

if __name__ == '__main__':
    # covert_h5()
    # covert_h5_new()     ## 预处理没有改变
    all_data()