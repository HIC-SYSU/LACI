import glob
import os
import random

def split_data():
    all_path = glob.glob('/data/chenjinfeng/Data/CT_160/norm_112_80_h5/*')
    num = len(all_path)
    print("Total number of cases:", num)

    text_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/'

    # # 检查目录是否存在
    # if not os.path.exists(text_path):
    #     os.makedirs(text_path)
    # else:
    #     print("Directory already exists. Stopping execution.")
    #     return

    # 随机选择训练集路径
    select_path_train = random.sample(all_path, 369)
    # 剩余的路径作为测试集
    remaining_list = [item for item in all_path if item not in select_path_train]
    select_path_test = random.sample(remaining_list, 92)

    train_file_path = os.path.join(text_path, f'train.txt')
    test_file_path = os.path.join(text_path, f'test.txt')

    # 写入训练路径
    with open(train_file_path, 'w') as f:
        for item in select_path_train:
            f.write(item + '\n')

    # 写入测试路径
    with open(test_file_path, 'w') as f:
        for item in select_path_test:
            f.write(item + '\n')


def split_data_all():
    all_path = glob.glob('/data/chenjinfeng/Data/CT_160/norm_112_80_h5/*')
    text_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/'
    num = len(all_path)
    print("Total number of cases:", num)

    # # 检查目录是否存在
    if not os.path.exists(text_path):
        os.makedirs(text_path)
    # else:
    #     print("Directory already exists. Stopping execution.")
    #     return

    # 随机选择训练集路径
    select_path_train = random.sample(all_path, num)
    train_file_path = os.path.join(text_path, f'train_all.txt')

    # 写入训练路径
    with open(train_file_path, 'w') as f:
        for item in select_path_train:
            f.write(item + '\n')

split_data()