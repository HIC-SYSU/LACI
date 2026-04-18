import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
from skimage.measure import label
import os
import scipy.ndimage as ndi


from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm

def test_all_case_LV_PR(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                        save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0, id=0, exp="BCP"):
    """
    功能：
    1) 保存每个病例、每个类别的 PR 曲线为 .npy
    2) 在测试时同步累计所有病例的 y_true / y_score
    3) 最后直接计算并保存整体 PR 曲线（global PR）为 .npy

    保存内容：
    A. 每个病例：
       - {name}_class_1_precision.npy
       - {name}_class_1_recall.npy
       - {name}_class_2_precision.npy
       - {name}_class_2_recall.npy
       - {name}_class_3_precision.npy
       - {name}_class_3_recall.npy

    B. 整体：
       - overall_class_1_precision.npy
       - overall_class_1_recall.npy
       - overall_class_2_precision.npy
       - overall_class_2_recall.npy
       - overall_class_3_precision.npy
       - overall_class_3_recall.npy
    """
    from sklearn.metrics import precision_recall_curve
    import os
    import numpy as np
    import h5py
    from tqdm import tqdm

    loader = tqdm(image_list) if not metric_detail else image_list

    # 确保保存路径存在
    if save_result and test_save_path and not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    # -------------------------
    # 用于累计整体 PR 的原始数据
    # -------------------------
    overall_labels = {
        1: [],
        2: [],
        3: [],
    }
    overall_scores = {
        1: [],
        2: [],
        3: [],
    }

    # 处理每个图像
    for i_batch, image_path in enumerate(loader):
        with h5py.File(image_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        name = os.path.basename(image_path).split('.')[0]  # 去掉文件扩展名

        if preproc_fn:
            image = preproc_fn(image)

        prediction, score_map = test_single_case(
            model, image, stride_xy, stride_z, patch_size,
            num_classes=num_classes, exp=exp
        )

        if nms:
            prediction = getLargestCC_per_class(prediction, num_classes=num_classes)

        score_map = np.asarray(score_map)

        # -------------------------
        # 统一成 [C, N]
        # 兼容 [C, D, H, W] 和 [D, H, W, C]
        # -------------------------
        if score_map.ndim < 2:
            raise ValueError(f'Unexpected score_map shape: {score_map.shape}')

        if score_map.shape[0] == num_classes:
            score_map_flat = score_map.reshape(num_classes, -1)
        elif score_map.shape[-1] == num_classes:
            score_map_flat = np.moveaxis(score_map, -1, 0).reshape(num_classes, -1)
        else:
            raise ValueError(f'Cannot determine class axis from score_map shape {score_map.shape}')

        labels = label.reshape(-1)

        # =========================
        # class 1
        # =========================
        binary_labels_1 = (labels == 1).astype(np.uint8)
        probabilities_1 = score_map_flat[1, :]

        if binary_labels_1.sum() > 0:
            precision_1, recall_1, thresholds_1 = precision_recall_curve(binary_labels_1, probabilities_1)

            # 保存病例级 PR
            np.save(os.path.join(test_save_path, f"{name}_class_1_precision.npy"), precision_1)
            np.save(os.path.join(test_save_path, f"{name}_class_1_recall.npy"), recall_1)

            print(f"Saved PR for {name} class 1 in NPY format")

            # 累计整体 PR 原始数据
            overall_labels[1].append(binary_labels_1)
            overall_scores[1].append(probabilities_1)
        else:
            print(f"Skip {name} class 1: no positive voxels")

        # =========================
        # class 2
        # =========================
        binary_labels_2 = (labels == 2).astype(np.uint8)
        probabilities_2 = score_map_flat[2, :]

        if binary_labels_2.sum() > 0:
            precision_2, recall_2, thresholds_2 = precision_recall_curve(binary_labels_2, probabilities_2)

            # 保存病例级 PR
            np.save(os.path.join(test_save_path, f"{name}_class_2_precision.npy"), precision_2)
            np.save(os.path.join(test_save_path, f"{name}_class_2_recall.npy"), recall_2)

            print(f"Saved PR for {name} class 2 in NPY format")

            # 累计整体 PR 原始数据
            overall_labels[2].append(binary_labels_2)
            overall_scores[2].append(probabilities_2)
        else:
            print(f"Skip {name} class 2: no positive voxels")

        # =========================
        # class 3 = union(1, 2)
        # =========================
        binary_labels_3 = np.isin(labels, [1, 2]).astype(np.uint8)
        probabilities_3 = score_map_flat[1, :] + score_map_flat[2, :]

        if binary_labels_3.sum() > 0:
            precision_3, recall_3, thresholds_3 = precision_recall_curve(binary_labels_3, probabilities_3)

            # 保存病例级 PR
            np.save(os.path.join(test_save_path, f"{name}_class_3_precision.npy"), precision_3)
            np.save(os.path.join(test_save_path, f"{name}_class_3_recall.npy"), recall_3)

            print(f"Saved PR for {name} class 3 in NPY format")

            # 累计整体 PR 原始数据
            overall_labels[3].append(binary_labels_3)
            overall_scores[3].append(probabilities_3)
        else:
            print(f"Skip {name} class 3: no positive voxels")

    # =========================================================
    # 循环结束后，计算并保存整体 PR 曲线（global PR）
    # =========================================================
    for class_id in [1, 2, 3]:
        if len(overall_labels[class_id]) == 0:
            print(f"Skip overall class {class_id}: no valid cases")
            continue

        y_true_all = np.concatenate(overall_labels[class_id], axis=0)
        y_score_all = np.concatenate(overall_scores[class_id], axis=0)

        precision_all, recall_all, thresholds_all = precision_recall_curve(y_true_all, y_score_all)

        np.save(os.path.join(test_save_path, f"overall_class_{class_id}_precision.npy"), precision_all)
        np.save(os.path.join(test_save_path, f"overall_class_{class_id}_recall.npy"), recall_all)

        print(
            f"Saved overall PR for class {class_id} | "
            f"num_voxels={len(y_true_all)} | "
            f"num_points={len(precision_all)}"
        )


def test_all_MCF_PR(vnet, resnet, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                    save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    """
    只计算并保存 PR 曲线数据：
    1) 保存每个病例的 PR 为 .npy
    2) 同时累计并保存整体 PR 为 .npy
    3) 不保存图像，不计算分割指标

    保存文件：
    A. 每个病例：
       - {name}_class_1_precision.npy
       - {name}_class_1_recall.npy
       - {name}_class_2_precision.npy
       - {name}_class_2_recall.npy
       - {name}_class_3_precision.npy
       - {name}_class_3_recall.npy

    B. 整体：
       - overall_class_1_precision.npy
       - overall_class_1_recall.npy
       - overall_class_2_precision.npy
       - overall_class_2_recall.npy
       - overall_class_3_precision.npy
       - overall_class_3_recall.npy
    """
    from sklearn.metrics import precision_recall_curve
    import os
    import numpy as np
    import h5py
    from tqdm import tqdm

    loader = tqdm(image_list) if not metric_detail else image_list

    if save_result and test_save_path is not None and not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    # 用于累计整体 PR 原始数据
    overall_labels = {
        1: [],
        2: [],
        3: [],
    }
    overall_scores = {
        1: [],
        2: [],
        3: [],
    }

    for i_batch, image_path in enumerate(loader):
        print(image_path)
        image_path = image_path.replace(
            '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing/MMWHS_MR/train_h5',
            "/data/xingshihanxiao/Pyproject/Contrast/datalist/data_mmwhs_shijie/train_h5"
        )
        print(image_path)

        with h5py.File(image_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        name = os.path.basename(image_list[i_batch])
        if name.endswith('.h5'):
            name = name[:-3]

        if preproc_fn is not None:
            image = preproc_fn(image)

        prediction, score_map = test_single_MCF(
            vnet, resnet, image,
            stride_xy, stride_z, patch_size,
            num_classes=num_classes
        )

        if nms:
            prediction = getLargestCC_per_class(prediction, num_classes=num_classes)

        # -------------------------
        # score_map 统一处理为 [C, N]
        # -------------------------
        score_map = np.asarray(score_map)

        if score_map.ndim < 2:
            raise ValueError(f'Unexpected score_map shape: {score_map.shape}')

        if score_map.shape[0] == num_classes:
            score_map_flat = score_map.reshape(num_classes, -1)
        elif score_map.shape[-1] == num_classes:
            score_map_flat = np.moveaxis(score_map, -1, 0).reshape(num_classes, -1)
        else:
            raise ValueError(
                f'Cannot determine class axis from score_map shape {score_map.shape} with num_classes={num_classes}'
            )

        labels = label.reshape(-1)

        # =========================
        # class 1
        # =========================
        binary_labels_1 = (labels == 1).astype(np.uint8)
        probabilities_1 = score_map_flat[1, :]

        if binary_labels_1.sum() > 0:
            precision_1, recall_1, thresholds_1 = precision_recall_curve(binary_labels_1, probabilities_1)

            np.save(os.path.join(test_save_path, f"{name}_class_1_precision.npy"), precision_1)
            np.save(os.path.join(test_save_path, f"{name}_class_1_recall.npy"), recall_1)

            print(f"Saved PR for {name} class 1 in NPY format")

            overall_labels[1].append(binary_labels_1)
            overall_scores[1].append(probabilities_1)
        else:
            print(f"Skip {name} class 1: no positive voxels")

        # =========================
        # class 2
        # =========================
        binary_labels_2 = (labels == 2).astype(np.uint8)
        probabilities_2 = score_map_flat[2, :]

        if binary_labels_2.sum() > 0:
            precision_2, recall_2, thresholds_2 = precision_recall_curve(binary_labels_2, probabilities_2)

            np.save(os.path.join(test_save_path, f"{name}_class_2_precision.npy"), precision_2)
            np.save(os.path.join(test_save_path, f"{name}_class_2_recall.npy"), recall_2)

            print(f"Saved PR for {name} class 2 in NPY format")

            overall_labels[2].append(binary_labels_2)
            overall_scores[2].append(probabilities_2)
        else:
            print(f"Skip {name} class 2: no positive voxels")

        # =========================
        # class 3 = union(1,2)
        # =========================
        binary_labels_3 = np.isin(labels, [1, 2]).astype(np.uint8)
        probabilities_3 = score_map_flat[1, :] + score_map_flat[2, :]

        if binary_labels_3.sum() > 0:
            precision_3, recall_3, thresholds_3 = precision_recall_curve(binary_labels_3, probabilities_3)

            np.save(os.path.join(test_save_path, f"{name}_class_3_precision.npy"), precision_3)
            np.save(os.path.join(test_save_path, f"{name}_class_3_recall.npy"), recall_3)

            print(f"Saved PR for {name} class 3 in NPY format")

            overall_labels[3].append(binary_labels_3)
            overall_scores[3].append(probabilities_3)
        else:
            print(f"Skip {name} class 3: no positive voxels")

    # =========================================================
    # 循环结束后，计算并保存整体 PR 曲线（global PR）
    # =========================================================
    for class_id in [1, 2, 3]:
        if len(overall_labels[class_id]) == 0:
            print(f"Skip overall class {class_id}: no valid cases")
            continue

        y_true_all = np.concatenate(overall_labels[class_id], axis=0)
        y_score_all = np.concatenate(overall_scores[class_id], axis=0)

        precision_all, recall_all, thresholds_all = precision_recall_curve(y_true_all, y_score_all)

        np.save(os.path.join(test_save_path, f"overall_class_{class_id}_precision.npy"), precision_all)
        np.save(os.path.join(test_save_path, f"overall_class_{class_id}_recall.npy"), recall_all)

        print(
            f"Saved overall PR for class {class_id} | "
            f"num_voxels={len(y_true_all)} | "
            f"num_points={len(precision_all)}"
        )
def getLargestCC(segmentation):
    labels = label(segmentation)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC

def var_all_case(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4):
    loader = tqdm(image_list)
    dice_MYO = 0.0
    dice_LV = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = [0, 0]
        else:
            dice_1 = metric.binary.dc(prediction==1, label==1)
            dice_2 = metric.binary.dc(prediction==2, label==2)
        dice_MYO += dice_1
        dice_LV += dice_2
    avg_dice_MYO = dice_MYO / len(image_list)
    avg_dice_LV = dice_LV / len(image_list)
    print('average metric is {},{}'.format(avg_dice_MYO, avg_dice_LV))
    return avg_dice_MYO, avg_dice_LV


def getLargestCC_per_class(prediction, num_classes):
    """
    对每个类别提取最大连通区域，并合并结果。
    :param prediction: 预测的分割结果，形状为 [depth, height, width]
    :param num_classes: 类别总数
    :return: 仅包含每个类别最大连通区域的分割结果
    """
    largest_cc_prediction = np.zeros_like(prediction)

    for cls in range(1, num_classes):  # 假设 0 为背景类，忽略
        # 对每个类别生成二值掩码
        binary_mask = (prediction == cls)

        # 标记连通区域
        labeled_mask, num_features = ndi.label(binary_mask)

        if num_features > 0:
            # 计算每个连通区域的大小
            sizes = ndi.sum(binary_mask, labeled_mask, range(1, num_features + 1))
            # 获取最大连通区域的标签
            largest_cc_label = np.argmax(sizes) + 1
            # 保留最大连通区域
            largest_cc_prediction[labeled_mask == largest_cc_label] = cls

    return largest_cc_prediction


import numpy as np

metric_names = ['Dice', 'Jc', 'HD', 'ASD']


def test_all_case(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0, exp="BCP", nms=0, FLAGS=None):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metrics = {}  # 存储每个样本的指标
    ith = 0

    for i_batch, image_path in enumerate(loader):
        print(image_path)
        # image_path = image_path.replace(
        #     '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/preprocessing/MMWHS_MR/train_h5',
        #     "/data/xingshihanxiao/Pyproject/Contrast/datalist/data_mmwhs_shijie/train_h5")
        print(image_path)
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        if preproc_fn is not None:
            image = preproc_fn(image)

        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes, exp=exp)

        # 写入 performance.txt 文件
        if num_classes == 5:
            name = os.path.basename(image_list[i_batch])
            Dice, Jc, HD, ASD = calculate_metrics_perclass(prediction, label[:], num_classes)
            with open(test_save_path + '../performance.txt', 'a') as f:
                f.write(
                    f'{name}, {Dice[0]:.6f}, {Dice[1]:.6f}, {Dice[2]:.6f}, {Dice[3]:.6f}, {Jc[0]:.6f}, {Jc[1]:.6f}, {Jc[2]:.6f},  {Jc[3]:.6f},  '
                    f'{HD[0]:.6f}, {HD[1]:.6f}, {HD[2]:.6f}, {HD[3]:.6f}, {ASD[0]:.6f}, {ASD[1]:.6f}, {ASD[2]:.6f}, {ASD[3]:.6f}\n')
            # 存储每个样本的度量
            total_metrics[name] = {
                'class_1': {'Dice': Dice[0], 'Jc': Jc[0], 'HD': HD[0], 'ASD': ASD[0]},
                'class_2': {'Dice': Dice[1], 'Jc': Jc[1], 'HD': HD[1], 'ASD': ASD[1]},
                # 'class_1_2': {'Dice': Dice[2], 'Jc': Jc[2], 'HD': HD[2], 'ASD': ASD[2]}
                'class_3': {'Dice': Dice[2], 'Jc': Jc[2], 'HD': HD[2], 'ASD': ASD[2]},
                'class_4': {'Dice': Dice[3], 'Jc': Jc[3], 'HD': HD[3], 'ASD': ASD[3]},
            }
        elif num_classes == 2:

            name = os.path.basename(os.path.dirname(image_path))
            Dice, Jc, HD, ASD = calculate_metrics_perclass(prediction, label[:], num_classes)
            # with open(test_save_path + '../performance.txt', 'a') as f:
            #     f.write(
            #         f"{name}, {Dice[0]:.6f}, {Jc[0]:.6f}, {HD[0]:.6f},{ASD[0]:.6f}\n")
            # # 存储每个样本的度量
            total_metrics[name] = {
                'class_1': {'Dice': Dice[0], 'Jc': Jc[0], 'HD': HD[0], 'ASD': ASD[0]},
            }

        elif num_classes == 3:
            name = os.path.basename(image_list[i_batch])
            Dice, Jc, HD, ASD = calculate_metrics_perclass_union(prediction, label[:], num_classes)
            with open(test_save_path + '../performance.txt', 'a') as f:
                f.write(
                    f"{name}, {Dice[0]:.6f}, {Dice[1]:.6f}, {Jc[0]:.6f}, {Jc[1]:.6f}, {HD[0]:.6f}, {HD[1]:.6f},{ASD[0]:.6f}, {ASD[1]:.6f}\n")
            # 存储每个样本的度量
            total_metrics[name] = {
                'class_1': {'Dice': Dice[0], 'Jc': Jc[0], 'HD': HD[0], 'ASD': ASD[0]},
                'class_2': {'Dice': Dice[1], 'Jc': Jc[1], 'HD': HD[1], 'ASD': ASD[1]},
                'class_1_2': {'Dice': Dice[2], 'Jc': Jc[2], 'HD': HD[2], 'ASD': ASD[2]}
            }
        elif num_classes == 4:
            name = os.path.basename(image_list[i_batch])
            Dice, Jc, HD, ASD = calculate_metrics_perclass(prediction, label[:], num_classes)
            with open(test_save_path + '../performance.txt', 'a') as f:
                f.write(
                    f'{name}, {Dice[0]:.6f}, {Dice[1]:.6f}, {Dice[2]:.6f}, {Jc[0]:.6f}, {Jc[1]:.6f}, {Jc[2]:.6f},  '
                    f'{HD[0]:.6f}, {HD[1]:.6f}, {HD[2]:.6f}, {ASD[0]:.6f}, {ASD[1]:.6f}, {ASD[2]:.6f}\n')
            # 存储每个样本的度量
            total_metrics[name] = {
                'class_1': {'Dice': Dice[0], 'Jc': Jc[0], 'HD': HD[0], 'ASD': ASD[0]},
                'class_2': {'Dice': Dice[1], 'Jc': Jc[1], 'HD': HD[1], 'ASD': ASD[1]},
                'class_3': {'Dice': Dice[2], 'Jc': Jc[2], 'HD': HD[2], 'ASD': ASD[2]},
            }



        if save_result:
            nib.save(nib.Nifti1Image(prediction[:].astype(np.float32), np.eye(4)), test_save_path + name + "_pred.nii.gz")
            # nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + name + "_img.nii.gz")
            nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)), test_save_path + name + "_score.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + name + "_gt.nii.gz")
        ith += 1

    if num_classes == 5:
        # 计算每个类别和每个度量的均值和标准差
        avg_std_metrics = {cls: {metric: {} for metric in metric_names} for cls in ['class_1', 'class_2', "class_3", "class_4"]}
        with open(test_save_path + '../performance.txt', 'a') as f:
            for cls in ['class_1', 'class_2', "class_3", "class_4"]:
                for metric in metric_names:
                    # 提取每个样本的该类别和该度量的值
                    metric_values = [total_metrics[sample][cls][metric] for sample in total_metrics]
                    # 计算均值和标准差
                    avg_metric = np.mean(metric_values)
                    std_metric = np.std(metric_values)
                    # 存储均值和标准差
                    avg_std_metrics[cls][metric]['mean'] = avg_metric
                    avg_std_metrics[cls][metric]['std'] = std_metric
                    # 输出均值和标准差到控制台
                    print(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}")
                    # 写入 performance.txt 文件
                    f.write(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}\n")
        return avg_std_metrics

    if num_classes == 2:
        # 计算每个类别和每个度量的均值和标准差
        avg_std_metrics = {cls: {metric: {} for metric in metric_names} for cls in ['class_1']}
        with open(test_save_path + '../performance.txt', 'a') as f:
            for cls in ['class_1']:
                for metric in metric_names:
                    # 提取每个样本的该类别和该度量的值
                    metric_values = [total_metrics[sample][cls][metric] for sample in total_metrics]
                    # 计算均值和标准差
                    avg_metric = np.mean(metric_values)
                    std_metric = np.std(metric_values)
                    # 存储均值和标准差
                    avg_std_metrics[cls][metric]['mean'] = avg_metric
                    avg_std_metrics[cls][metric]['std'] = std_metric
                    # 输出均值和标准差到控制台
                    print(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}")
                    # 写入 performance.txt 文件
                    f.write(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}\n")
        return avg_std_metrics

    if num_classes == 3:
        # 计算每个类别和每个度量的均值和标准差
        avg_std_metrics = {cls: {metric: {} for metric in metric_names} for cls in ['class_1', 'class_2', "class_1_2"]}
        with open(test_save_path + '../performance.txt', 'a') as f:
            for cls in ['class_1', 'class_2', "class_1_2"]:
                for metric in metric_names:
                    # 提取每个样本的该类别和该度量的值
                    metric_values = [total_metrics[sample][cls][metric] for sample in total_metrics]
                    # 计算均值和标准差
                    avg_metric = np.mean(metric_values)
                    std_metric = np.std(metric_values)
                    # 存储均值和标准差
                    avg_std_metrics[cls][metric]['mean'] = avg_metric
                    avg_std_metrics[cls][metric]['std'] = std_metric
                    # 输出均值和标准差到控制台
                    print(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}")
                    # 写入 performance.txt 文件
                    f.write(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}\n")
        return avg_std_metrics
    if num_classes == 4:
        # 计算每个类别和每个度量的均值和标准差
        avg_std_metrics = {cls: {metric: {} for metric in metric_names} for cls in ['class_1', 'class_2', "class_3"]}
        with open(test_save_path + '../performance.txt', 'a') as f:
            for cls in ['class_1', 'class_2', "class_3"]:
                for metric in metric_names:
                    # 提取每个样本的该类别和该度量的值
                    metric_values = [total_metrics[sample][cls][metric] for sample in total_metrics]
                    # 计算均值和标准差
                    avg_metric = np.mean(metric_values)
                    std_metric = np.std(metric_values)
                    # 存储均值和标准差
                    avg_std_metrics[cls][metric]['mean'] = avg_metric
                    avg_std_metrics[cls][metric]['std'] = std_metric
                    # 输出均值和标准差到控制台
                    print(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}")
                    # 写入 performance.txt 文件
                    f.write(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}\n")
        return avg_std_metrics





def test_all_case_pre(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0, exp="BCP", nms=0, FLAGS=None):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metrics = {}  # 存储每个样本的指标
    ith = 0

    for i_batch, image_path in enumerate(loader):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        # label = h5f['label'][:]
        name = os.path.basename(image_list[i_batch])

        if preproc_fn is not None:
            image = preproc_fn(image)

        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes, exp=exp)

        if save_result:
            nib.save(nib.Nifti1Image(prediction[:].astype(np.float32), np.eye(4)), test_save_path + name + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + name + "_img.nii.gz")
            nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)), test_save_path + name + "_score.nii.gz")
            # nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + name + "_gt.nii.gz")
        ith += 1



def test_all_MCF(vnet, resnet, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metrics = {}  # 存储每个样本的指标
    ith = 0

    for i_batch, image_path in enumerate(loader):
        print(image_path)
        image_path = image_path.replace(
            '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing/MMWHS_MR/train_h5',
            "/data/xingshihanxiao/Pyproject/Contrast/datalist/data_mmwhs_shijie/train_h5")
        print(image_path)
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        # label_slice = h5f['label_slice'][:]
        name = os.path.basename(image_list[i_batch])

        if preproc_fn is not None:
            image = preproc_fn(image)

        # prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        prediction, score_map = test_single_MCF(vnet, resnet, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if nms:
            prediction = getLargestCC_per_class(prediction, num_classes=num_classes)

        if np.sum(prediction) == 0:
            Dice, Jc, HD, ASD = (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)
        else:
            Dice, Jc, HD, ASD = calculate_metrics_perclass_union(prediction, label[:], num_classes)

        # 打印详细度量信息
        with open(test_save_path + '../performance.txt', 'a') as f:
            f.write(
                f"{name}, {Dice[0]:.6f}, {Dice[1]:.6f}, {Dice[2]:.6f}, {Jc[0]:.6f}, {Jc[1]:.6f}, {Jc[2]:.6f}, {HD[0]:.6f}, {HD[1]:.6f}, {HD[2]:.6f}, {ASD[0]:.6f}, {ASD[1]:.6f}, {ASD[2]:.6f}\n")

        # 写入 performance.txt 文件
        with open(test_save_path + '../performance.txt', 'a') as f:
            f.write(
                f"{name}, {Dice[0]:.6f}, {Dice[1]:.6f}, {Dice[2]:.6f} "
                f"{Jc[0]:.6f}, {Jc[1]:.6f}, {Jc[2]:.6f}"
                f"{HD[0]:.6f}, {HD[0]:.6f}, {HD[2]:.6f}"
                f"{ASD[0]:.6f}, {ASD[1]:.6f}, {ASD[2]:.6f}\n")

        # 存储每个样本的度量
        total_metrics[name] = {
            'class_1': {'Dice': Dice[0], 'Jc': Jc[0], 'HD': HD[0], 'ASD': ASD[0]},
            'class_2': {'Dice': Dice[1], 'Jc': Jc[1], 'HD': HD[1], 'ASD': ASD[1]},
            'class_3': {'Dice': Dice[2], 'Jc': Jc[2], 'HD': HD[2], 'ASD': ASD[2]}
        }

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + name + "_pred.nii.gz")
            # nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + name + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + name + "_gt.nii.gz")
            nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)), test_save_path + name + "_score.nii.gz")
            # nib.save(nib.Nifti1Image(label_slice[:].astype(np.float32), np.eye(4)), test_save_path + name + "_slice.nii.gz")
        ith += 1

    # 计算每个类别和每个度量的均值和标准差
    avg_std_metrics = {cls: {metric: {} for metric in metric_names} for cls in ['class_1', 'class_2', 'class_3']}
    with open(test_save_path + '../performance.txt', 'a') as f:
        for cls in ['class_1', 'class_2', 'class_3']:
            for metric in metric_names:
                # 提取每个样本的该类别和该度量的值
                metric_values = [total_metrics[sample][cls][metric] for sample in total_metrics]

                # 计算均值和标准差
                avg_metric = np.mean(metric_values)
                std_metric = np.std(metric_values)

                # 存储均值和标准差
                avg_std_metrics[cls][metric]['mean'] = avg_metric
                avg_std_metrics[cls][metric]['std'] = std_metric

                # 输出均值和标准差到控制台
                print(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}")

                # 写入 performance.txt 文件
                f.write(f"{cls} {metric} - Mean: {avg_metric:.6f}, Std: {std_metric:.6f}\n")

    return avg_std_metrics


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=4, exp=None):
    w, h, d = image.shape
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    # y1, features = model(test_patch)
                    # y = y1[0, :, :, :, :].cpu().data.numpy()
                    # if exp=="AU_MT":
                    #     y1, _ = model(test_patch)      ## ACMT, AU,VNet

                    if exp=="UAMT":
                        y1 = model(test_patch, text_features=None)  ## UAMT
                    elif exp=="Vnet" or exp=="AUMT" or exp=="LACI" or exp=="LACI_1" or exp=="LACI_1_1" or exp=="LACI_1_1_FastFF" or exp=="LACI_review" or exp=="ACMT"  or exp=="AU_MT" or exp=="LACI_review_try2" or exp=="LACI_review_try3":
                        y1 = model(test_patch)  ## UAMT
                    # elif exp=="UPRC":
                    #     y1, _, _, _ = model(test_patch)
                    # elif exp=="LeFed":
                    #     y, y1, _, _, _ = model(test_patch, [])      ## LeFed
                    elif exp=="TAC":
                        y1, _, _, _, _, _, _, _, _, _ = model(test_patch)   ## TAC
                    # elif exp=="comwin":
                    #     y1 = model(test_patch,[])   ## comwin
                    elif exp=="BCP" or exp=="BCP_flod0" or exp=="VNet":
                        y1, features = model(test_patch)
                        # # print(features[-1].shape)
                        # y = y1[0, :, :, :, :].cpu().data.numpy()

                    y = F.softmax(y1, dim=1).cpu().data.numpy()[0]     ## (1,3,112,112,80) - - (1, 112, 112, 80)
                    # print(y1.shape, y.shape)
                    # elif exp=="MCF":
                    # y = y1[0,:,:,:,:].cpu().data.numpy()        ## MCF
                    # y = y1.cpu().data.numpy()
                    # print(y.shape)
                # Update score map and count map
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1

    # # Normalize the score map by the count map
    score_map /= np.maximum(cnt, 1)  # Avoid division by zero, apply broadcasting
    # # Generate the final label map
    if num_classes!=2:
        label_map = logits_to_label_3d(score_map)
    elif num_classes==2:
        label_map = one_hot2_3d(score_map)
    # label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_single_MCF(vnet,resnet=None, image=None, stride_xy=None, stride_z=None, patch_size=None, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)

    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                if vnet is not None:
                    y1 = vnet(test_patch)
                    y  = F.softmax(y1, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0,:,:,:,:]
                if resnet is not None:
                    y2 = resnet(test_patch)
                    y2 = F.softmax(y2, dim=1)
                    y2 = y2.cpu().data.numpy()
                    if vnet is not None:
                        y = (y+y2[0, :, :, :, :])/2
                    else:
                        y = y2[0, :, :, :, :]
                # Update score map and count map
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def one_hot2_3d(predict):  # shape = [s, h, w]
    predict[predict > 0.5] = 1
    predict[predict < 0.5] = 0
    # d, s, h, w = predict.shape
    [a, d, w, h] = predict.shape
    predict_3D = np.zeros(( d, w, h))
    for t in range(d):
        predict_3D[t,:,:][predict[0,t,:,:] == 1] = 0
        predict_3D[t,:,:][predict[1,t,:,:] == 1] = 1
        # predict_3D[t,:,:][predict[2,t,:,:] == 1] = 2
        # predict_3D[t, :, :][predict[3, t, :, :] == 1] = 3
    return predict_3D


def logits_to_label_3d(predict):
    """
    predict: numpy array, shape [C, D, H, W]
             multi-class probabilities/logits
    return: label map, shape [D, H, W]
    """
    predict = np.asarray(predict)
    pred_3d = np.argmax(predict, axis=0).astype(np.uint8)
    return pred_3d


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

def calculate_metrics_perclass(pred, gt, num_classes):
    """
    pred: ndarray, predicted segmentation [H, W] or [D, H, W]
    gt: ndarray, ground truth segmentation [H, W] or [D, H, W]
    num_classes: int, total number of classes including background
    """
    import medpy.metric.binary as mmb

    # 创建字典来存储每个类别的度量结果
    metrics = {
        'dice': [],
        'jc': [],
        'hd95': [],
        'asd': []
    }

    # 遍历所有前景类别（从1开始跳过背景）
    for i in range(1, num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)

        # 判断是否两个都是空掩码
        if not pred_i.any() and not gt_i.any():
            print(f"Class {i} is empty in both pred and gt. Skip.")
            continue

        # 如果只有一个是空的，dice=0, hd95/asd=999
        if not pred_i.any() or not gt_i.any():
            print(f"Class {i} missing in pred or gt. Set Dice=0, HD/ASD=999.")
            dice = 0.0
            jc = 0.0
            hd = 999.0
            asd = 999.0
        else:
            # 正常计算
            dice = mmb.dc(pred_i, gt_i)
            jc = mmb.jc(pred_i, gt_i)
            hd = mmb.hd95(pred_i, gt_i)
            asd = mmb.asd(pred_i, gt_i)

        # 保存每类的指标
        metrics['dice'].append(dice)
        metrics['jc'].append(jc)
        metrics['hd95'].append(hd)
        metrics['asd'].append(asd)
    print(f"metrics['dice'], metrics['jc'], metrics['hd95'], metrics['asd']:{metrics['dice'], metrics['jc'], metrics['hd95'], metrics['asd']}")

    return metrics['dice'], metrics['jc'], metrics['hd95'], metrics['asd']

def calculate_metrics_perclass_union(pred, gt, num_classes):
    """
    pred: ndarray, predicted segmentation [H, W] or [D, H, W]
    gt: ndarray, ground truth segmentation [H, W] or [D, H, W]
    num_classes: int, total number of classes including background
    """
    import medpy.metric.binary as mmb

    metrics = {
        'dice': [],
        'jc': [],
        'hd95': [],
        'asd': []
    }

    # -----------------------------
    # 1. 逐类计算指标 (class 1, 2, ...)
    # -----------------------------
    for i in range(1, num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)

        if not pred_i.any() and not gt_i.any():
            # 两者都是空
            dice = jc = 1.0
            hd = asd = 0.0
        elif not pred_i.any() or not gt_i.any():
            # 一个空，一个不空
            dice = jc = 0.0
            hd = asd = 999.0
        else:
            # 正常计算
            dice = mmb.dc(pred_i, gt_i)
            jc = mmb.jc(pred_i, gt_i)
            hd = mmb.hd95(pred_i, gt_i)
            asd = mmb.asd(pred_i, gt_i)

        metrics['dice'].append(dice)
        metrics['jc'].append(jc)
        metrics['hd95'].append(hd)
        metrics['asd'].append(asd)

    # -----------------------------
    # 2. 计算 union 类别  (class 1 + class 2)
    # -----------------------------
    if num_classes > 2:
        pred_union = np.logical_or(pred == 1, pred == 2)
        gt_union   = np.logical_or(gt == 1, gt == 2)

        if not pred_union.any() and not gt_union.any():
            dice_u = jc_u = 1.0
            hd_u = asd_u = 0.0
        elif not pred_union.any() or not gt_union.any():
            dice_u = jc_u = 0.0
            hd_u = asd_u = 999.0
        else:
            dice_u = mmb.dc(pred_union, gt_union)
            jc_u = mmb.jc(pred_union, gt_union)
            hd_u = mmb.hd95(pred_union, gt_union)
            asd_u = mmb.asd(pred_union, gt_union)

        # 把 union 也加入列表
        metrics['dice'].append(dice_u)
        metrics['jc'].append(jc_u)
        metrics['hd95'].append(hd_u)
        metrics['asd'].append(asd_u)

    print("Dice:", metrics['dice'])
    print("Jaccard:", metrics['jc'])
    print("HD95:", metrics['hd95'])
    print("ASD:", metrics['asd'])

    return metrics['dice'], metrics['jc'], metrics['hd95'], metrics['asd']


# def calculate_metrics_perclass(pred, gt, num_classes):
#     # 创建字典来存储每个类别的度量结果
#     metrics = {
#         'dice': [],
#         'jc': [],
#         'hd95': [],
#         'asd': []
#     }
#
#     # 遍历所有类别
#     for i in range(1,num_classes):
#         # 为每个类别生成二进制掩码
#         pred_i = (pred == i)
#         gt_i = (gt == i)
#
#         # 计算每个度量
#         dice = metric.binary.dc(pred_i, gt_i)
#         jc = metric.binary.jc(pred_i, gt_i)
#         hd = metric.binary.hd95(pred_i, gt_i)
#         asd = metric.binary.asd(pred_i, gt_i)
#
#         # 将结果添加到字典
#         metrics['dice'].append(dice)
#         metrics['jc'].append(jc)
#         metrics['hd95'].append(hd)
#         metrics['asd'].append(asd)
#
#         # 计算标签 1 和标签 2 的联合指标
#         pred_combined = (pred == 1) | (pred == 2)  # 联合预测类别 1 和类别 2
#         gt_combined = (gt == 1) | (gt == 2)  # 联合真实类别 1 和类别 2
#
#         # 如果联合类别的预测和实际值都为空，则跳过
#         if pred_combined.any() or gt_combined.any():
#             dice_combined = metric.binary.dc(pred_combined, gt_combined)
#             jc_combined = metric.binary.jc(pred_combined, gt_combined)
#             hd_combined = metric.binary.hd95(pred_combined, gt_combined)
#             asd_combined = metric.binary.asd(pred_combined, gt_combined)
#
#             # 将联合类别的指标添加到字典
#             metrics['dice'].append(dice_combined)
#             metrics['jc'].append(jc_combined)
#             metrics['hd95'].append(hd_combined)
#             metrics['asd'].append(asd_combined)
#
#     return metrics['dice'], metrics['jc'], metrics['hd95'], metrics['asd']


def test_all_case_MCF_s(vnet, resnet, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    metric_dice = []
    metric_jac = []
    metric_hd = []
    metric_asd = []
    with open(test_save_path + "/test_log.txt", "a") as f:
        for i_batch, image_path in enumerate(tqdm(image_list)):
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            name = os.path.basename(image_list[i_batch])
            # name = os.path.basename(os.path.dirname(image_list[i_batch]))

            label = label == 1

            if preproc_fn is not None:
                image = preproc_fn(image)
            prediction, score_map = test_single_MCF(vnet, resnet, image, stride_xy, stride_z, patch_size,
                                                     num_classes=num_classes)

            if np.sum(prediction) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])
            # print(single_metric)
            total_metric += np.asarray(single_metric)
            metric_dice.append(single_metric[0])
            metric_jac.append(single_metric[1])
            metric_hd.append(single_metric[2])
            metric_asd.append(single_metric[3])
            # print(str(cnt) + ", {}, {}, {}. {}".format(single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
            f.writelines(
                "{},{},{},{},{}\n".format(name, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

            if save_result:
                nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                         test_save_path + name + "_pred.nii.gz")
                nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + name + "_img.nii.gz")
                nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + name + "_gt.nii.gz")

        avg_metric = total_metric / len(image_list)
        std = [np.std(metric_dice), np.std(metric_jac), np.std(metric_hd), np.std(metric_asd)]
        std_error = [np.std(metric_dice) / math.sqrt(len(image_list)), np.std(metric_jac) / math.sqrt(len(image_list)),
                     np.std(metric_hd) / math.sqrt(len(image_list)), np.std(metric_asd) / math.sqrt(len(image_list))]
        ##标准偏差表示数据分布情况，即数据集中的值相对于其平均值的离散程度，反映了数据点偏离平均值的程度。
        ## 标准误差表示统计估计量可能的误差范围，用于表达估计量的可靠性。
        f.writelines("Mean metrics,{},{},{},{}\n".format(avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3]))
        f.close()
    print("Testing end")
    print(f'avg_metric: {avg_metric}, std: {std}, std_error: {std_error}')
    return avg_metric, std, std_error


def test_single_case_MCF_s(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y, y1, _, _, _ = net(test_patch, [])
                    # ensemble
                    # y1 = (y + y1)/2
                    # y = F.softmax(y1, dim=1)
                    # strong
                    y = F.softmax(y, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]

                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


from scipy.ndimage import zoom
from matplotlib import pyplot as plt
def test_all_case_gradcam(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metrics = {}  # 存储每个样本的指标
    ith = 0
    activations = []

    for i_batch, image_path in enumerate(loader):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        name = os.path.basename(image_list[i_batch])
        print(name)

        if preproc_fn is not None:
            image = preproc_fn(image)

        combined_activation = test_single_case_cam(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        activations.append((image, combined_activation))  # 保存原图和激活图
    # 确保有两张图像
    assert len(activations) == 2, "There should be exactly two activations for comparison."

    # 获取两张原始图像和激活图
    image1, activation1 = activations[0]
    image2, activation2 = activations[1]

    # 计算目标形状（取最大形状）
    target_shape = np.max([image1.shape, image2.shape, activation1.shape, activation2.shape], axis=0)

    # 插值原始图像到目标形状
    image1_resized = resize_to_target(image1, target_shape)
    image2_resized = resize_to_target(image2, target_shape)

    # 插值激活图到目标形状
    activation1_resized = resize_to_target(activation1, target_shape)
    activation2_resized = resize_to_target(activation2, target_shape)

    # 计算激活差异
    activation_diff = np.abs(activation1_resized - activation2_resized)

    # 可视化
    visualize_activation_diff(image1_resized, activation1_resized, activation_diff,
                              title="Image 1 with Activation Difference")
    visualize_activation_diff(image2_resized, activation2_resized, activation_diff,
                              title="Image 2 with Activation Difference")
        # slice_index = image.shape[-1] // 2  # 中间切片
        # original_slice = image[:, :, slice_index]
        # activation_slice = combined_activation[:, :, slice_index]
        #
        # # 可视化原图
        # plt.figure(figsize=(18, 6))
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_slice, cmap="gray")
        # # plt.title("Original Image")
        # plt.axis("off")
        #
        # # # 可视化激活图
        # # plt.subplot(1, 3, 2)
        # # plt.imshow(activation_slice, cmap="jet")
        # # plt.colorbar()
        # # plt.title("Activation Map")
        # # plt.axis("off")
        #
        # # 激活图叠加在原图上
        # plt.subplot(1, 2, 2)
        # plt.imshow(original_slice, cmap="gray")  # 显示灰度原图
        # plt.imshow(activation_slice, cmap="jet", alpha=0.5)  # 叠加激活图，设置透明度
        # plt.colorbar()
        # # plt.title("Activation Overlay on Original Image")
        # plt.axis("off")
        #
        # plt.tight_layout()
        # plt.show()
    return combined_activation


def resize_to_target(image, target_shape):
    """
    对 3D 图像插值到目标形状
    Args:
        image (numpy.ndarray): 输入图像，形状为 [H, W, D]
        target_shape (tuple): 目标形状
    Returns:
        numpy.ndarray: 插值后的图像
    """
    scale = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, scale, order=1)  # 使用线性插值



def visualize_activation_diff(image, activation, activation_diff, title="Activation Overlay"):
    """
    可视化单张图像及其激活差异特征
    Args:
        image (numpy.ndarray): 原始图像
        activation (numpy.ndarray): 激活图
        activation_diff (numpy.ndarray): 差异特征图
        title (str): 显示标题
    """
    # 选择中间切片
    slice_index = image.shape[-1] // 2
    original_slice = image[:, :, slice_index]
    activation_slice = activation[:, :, slice_index]
    activation_diff_slice = activation_diff[:, :, slice_index]

    # 归一化激活图和差异图到 [0, 1]
    activation_slice = (activation_slice - np.min(activation_slice)) / (
        np.max(activation_slice) - np.min(activation_slice) + 1e-8)
    activation_diff_slice = (activation_diff_slice - np.min(activation_diff_slice)) / (
        np.max(activation_diff_slice) - np.min(activation_diff_slice) + 1e-8)

    # 可视化
    plt.figure(figsize=(18, 6))

    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(original_slice, cmap="gray")
    # plt.title("Original Image")
    plt.axis("off")


    from scipy.ndimage import gaussian_filter
    plt.subplot(2, 2, 2)
    # activation_diff_slice = 1 / (1 + np.exp(-0.5 * (activation_diff_slice)))
    sigma = 6  # 可以调整 sigma 的值，控制模糊程度
    smoothed_activation_diff_slice = gaussian_filter(activation_diff_slice, sigma=sigma)

    plt.imshow(smoothed_activation_diff_slice, cmap="jet", alpha=0.8)
    # plt.title("Original Image")
    plt.axis("off")

    # 原始激活图
    plt.subplot(2, 2, 3)
    plt.imshow(original_slice, cmap="gray")
    plt.imshow(activation_slice, cmap="jet", alpha=0.5)
    plt.colorbar()
    # plt.title("Activation Map")
    plt.axis("off")

    # 激活差异图叠加
    plt.subplot(2, 2, 4)
    plt.imshow(original_slice, cmap="gray")
    plt.imshow(smoothed_activation_diff_slice, cmap="jet", alpha=0.5)
    plt.colorbar()
    # plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def test_single_case_cam(model, image, stride_xy, stride_z, patch_size, num_classes=3):
    w, h, d = image.shape
    downsample_ratio = 16
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    features_map = np.zeros(
        (256, image.shape[0] // downsample_ratio, image.shape[1] // downsample_ratio,
         image.shape[2] // downsample_ratio),
        dtype=np.float32
    )
    cnt = np.zeros(image.shape).astype(np.float32)
    cnt_f = np.zeros_like(features_map, dtype=np.float32)  # 累计计数器

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1, features = model(test_patch)
                    # print(features[-1].shape)
                    y = y1[0, :, :, :, :].cpu().data.numpy()
                    f = features[-1][0, :, :, :, :].cpu().data.numpy()  # 特征图块

                # Update score map and count map
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1
                # 计算特征图块的插入位置
                fx, fy, fz = xs // downsample_ratio, ys // downsample_ratio, zs // downsample_ratio
                fs = patch_size[0] // downsample_ratio, patch_size[1] // downsample_ratio, patch_size[
                    2] // downsample_ratio

                # 累积低分辨率特征图块
                features_map[:, fx:fx + fs[0], fy:fy + fs[1], fz:fz + fs[2]] += f
                cnt_f[fx:fx + fs[0], fy:fy + fs[1], fz:fz + fs[2]] += 1
    # # Normalize the score map by the count map
    score_map /= np.maximum(cnt, 1)  # Avoid division by zero, apply broadcasting
    features_map /= np.maximum(cnt_f, 1)  # 同样防止除以 0
    # # Generate the final label map
    label_map = one_hot2_3d(score_map)
    # label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    # return label_map, score_map

    # 假设你想对某类别的概率图和特征图进行融合
    # 上采样特征图到原始图像分辨率
    features_map_upsampled = F.interpolate(
        torch.tensor(features_map).unsqueeze(0),
        size=image.shape,
        mode="trilinear",
        align_corners=True
    ).squeeze(0).numpy()

    # 选择类别 1 和类别 2
    target_classes = [1, 2]
    weights = [2, 2]  # 对类别1和类别2的权重，可以调整
    # 提取目标类别的概率图并加权求和
    prob_map = sum(w * score_map[cls] for w, cls in zip(weights, target_classes))
    # 结合特征图和概率图
    combined_activation = prob_map * np.sum(features_map_upsampled, axis=0)  # 点乘融合
    # 归一化
    combined_activation = (combined_activation - np.min(combined_activation)) / (
            np.max(combined_activation) - np.min(combined_activation) + 1e-8)
    # 使用 Sigmoid 归一化
    # combined_activation = 1 / (1 + np.exp(-0.01*combined_activation))

    return combined_activation


def combine_features_and_score_map(features, score_map, method='weighted_sum'):
    """
    将 encoder 的特征图与最后的概率图结合。
    Args:
        features (torch.Tensor): Encoder 的特征图，形状为 [B, C_f, H_f, W_f, D_f]
        score_map (np.ndarray): 概率图，形状为 [num_classes, H, W, D]
        method (str): 融合方法 ('weighted_sum', 'dot', 'concat')
    Returns:
        combined_activation_map (torch.Tensor): 融合后的激活图，形状为 [H, W, D]
    """
    # 确保特征图和概率图都在 torch.Tensor 格式中
    features = torch.tensor(features).float() if isinstance(features, np.ndarray) else features
    score_map = torch.tensor(score_map).float() if isinstance(score_map, np.ndarray) else score_map

    # Step 1: 插值特征图到与 score_map 相同的大小
    num_classes, H, W, D = score_map.shape
    _, C_f, H_f, W_f, D_f = features.shape

    # 上采样特征图到概率图的大小
    features_resized = F.interpolate(features, size=(H, W, D), mode='trilinear', align_corners=True)

    # Step 2: 融合方式
    if method == 'weighted_sum':
        # 加权融合：将特征图与概率图逐像素相加
        combined = torch.sum(features_resized, dim=1) + torch.max(score_map, dim=0)[0]
    elif method == 'dot':
        # 点乘融合：逐像素点乘特征图和概率图
        score_map_expanded = score_map.unsqueeze(1)  # 扩展到 [num_classes, 1, H, W, D]
        combined = torch.sum(features_resized * score_map_expanded, dim=1)
    elif method == 'concat':
        # 拼接融合：直接拼接特征图和概率图
        combined = torch.cat([features_resized, score_map.unsqueeze(1)], dim=1)
    else:
        raise ValueError("Unsupported method! Choose from 'weighted_sum', 'dot', or 'concat'.")

    return combined.cpu().numpy()  # 转回 NumPy 格式以便后续处理
