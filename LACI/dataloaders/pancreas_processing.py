import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd

output_size =[112, 112, 80]

# def covert_h5():
#     listt = glob('/data/chenjinfeng/40180-data/chenjinfeng/Data_download/Abdominal/Pancreas-CT/')
#     for item in tqdm(listt):
#         image, img_header = nrrd.read(item)
#         label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
#         label = (label == 255).astype(np.uint8)
#         w, h, d = label.shape
#
#         tempL = np.nonzero(label)
#         minx, maxx = np.min(tempL[0]), np.max(tempL[0])
#         miny, maxy = np.min(tempL[1]), np.max(tempL[1])
#         minz, maxz = np.min(tempL[2]), np.max(tempL[2])
#
#         px = max(output_size[0] - (maxx - minx), 0) // 2
#         py = max(output_size[1] - (maxy - miny), 0) // 2
#         pz = max(output_size[2] - (maxz - minz), 0) // 2
#         minx = max(minx - np.random.randint(10, 20) - px, 0)
#         maxx = min(maxx + np.random.randint(10, 20) + px, w)
#         miny = max(miny - np.random.randint(10, 20) - py, 0)
#         maxy = min(maxy + np.random.randint(10, 20) + py, h)
#         minz = max(minz - np.random.randint(5, 10) - pz, 0)
#         maxz = min(maxz + np.random.randint(5, 10) + pz, d)
#
#         image = (image - np.mean(image)) / np.std(image)
#         image = image.astype(np.float32)
#         image = image[minx:maxx, miny:maxy]
#         label = label[minx:maxx, miny:maxy]
#         print(label.shape)
#         f = h5py.File(item.replace('lgemri.nrrd', 'mri_norm2.h5'), 'w')
#         f.create_dataset('image', data=image, compression="gzip")
#         f.create_dataset('label', data=label, compression="gzip")
#         f.close()

def covert_h5():
    # 遍历数据集路径
    listt = glob.glob('/data/chenjinfeng/40180-data/chenjinfeng/Data_download/Abdominal/Pancreas-CT/*/lgemri.nrrd')
    for item in tqdm(listt):
        # 读取图像和标签
        image, img_header = nrrd.read(item)
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)

        # 将HU值调整到软组织CT窗口 [-120, 240]
        image = np.clip(image, -120, 240)

        # 标准化HU值到范围 [0, 1] 以进行处理
        image = (image + 120) / 360  # 将 [-120, 240] 映射到 [0, 1]

        # 获取标签的形状
        w, h, d = label.shape

        # 查找非零标签的边界框
        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        # 计算边界框的填充
        px = max(96 - (maxx - minx), 0) // 2
        py = max(96 - (maxy - miny), 0) // 2
        pz = max(96 - (maxz - minz), 0) // 2

        # 对边界框进行随机扩展
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        # 图像归一化到零均值和单位标准差
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)

        # 裁剪图像和标签
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]

        print("Label shape after cropping:", label.shape)

        # 保存为HDF5文件
        f = h5py.File(item.replace('lgemri.nrrd', 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()