#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
from medpy import metric


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


import torch

def dice_metric_binary(pred, target):
    smooth = 1e-8
    num = pred.size(0)
    m1 = pred.reshape(num, -1)  # Flatten
    m2 = target.reshape(num, -1)  # Flatten
    intersection = m1 * m2
    # intersection1 = torch.dot(m1, m2)
    # print('intersection', intersection)
    # print('intersection1', intersection1)

    intersection = torch.sum(intersection, dim=1, keepdim=True)
    m1 = torch.sum(m1, dim=1, keepdim=True)
    m2 = torch.sum(m2, dim=1, keepdim=True)

    loss = (2. * intersection + smooth) / (m1 + m2 + smooth)
    # a = 2. * intersection + smooth
    # b = m1 + m2 + smooth
    # print('up', a)
    # print('down',b)
    # print('ab', a/b)
    return torch.mean(loss, dim=0, keepdim=False)

def dice_metric_multiclass(input, target):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_metric_binary(input[:, channel, ...], target[:, channel, ...])

    return dice / input.shape[1]

def dice_metric_singleclass_list(input, target):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dices = []
    dice_sum= 0
    for batch in range(input.shape[0]):
        dice = dice_metric_binary(input[batch, :, ...], target[batch, :, ...])
        dices.append(dice)
        dice_sum += dice

    return dices, dice_sum/input.shape[0]


import torch


import torch

def dice_metric_multiclass_list(input, target, epsilon=1e-6):
    """
    计算类别1和类别2的 Dice 系数并返回各自的 Dice 列表
    :param input: 模型输出的预测，形状为 [batch_size, num_classes, depth, height, width]
    :param target: one-hot 编码的标签，形状为 [batch_size, num_classes, depth, height, width]
    :param epsilon: 防止除以零的小数
    :return: 类别1和类别2的平均 Dice 和批次的 Dice 列表
    """
    assert input.size() == target.size(), "输入和目标尺寸不匹配"
    num_classes = input.shape[1]
    batch_size = input.shape[0]
    dices_per_class = [[] for _ in range(num_classes)]  # 初始化一个嵌套列表来存储每个类别的 Dice 系数

    for batch in range(batch_size):
        for cls in [1, 2]:  # 只计算类别1和类别2的 Dice 系数
            pred = input[batch, cls, ...]  # 获取当前 batch 中的第 cls 类预测
            true = target[batch, cls, ...]  # 获取当前 batch 中的第 cls 类真实标签

            # 计算交集和并集
            intersection = torch.sum(pred * true)
            union = torch.sum(pred) + torch.sum(true)
            dice = (2. * intersection + epsilon) / (union + epsilon)

            dices_per_class[cls].append(dice)  # 将当前类别的 Dice 添加到对应类别的列表中

    # 计算类别1和类别2的平均 Dice 系数
    mean_dice_1 = torch.mean(torch.stack(dices_per_class[1]))
    mean_dice_2 = torch.mean(torch.stack(dices_per_class[2]))

    return mean_dice_1, mean_dice_2, dices_per_class[1], dices_per_class[2]  # 只返回类别1和类别2的结果


# def dice_metric_multiclass_time(input, target):
#     dice = 0
#     for frame in range(input.shape[1]):
#         dice += dice_metric_multiclass(input[:, frame, ...], target[:, frame, ...])
#
#     return dice / input.shape[1]

def dice_metric_multiclass_time(input, target):
    dice = 0
    for frame in range(input.shape[2]):
        dice += dice_metric_multiclass(input[:, :, frame, ...], target[:, :, frame, ...])

    return dice / input.shape[2]

def to_one_hot_3d(tensor, n_classes=3):  # shape = [s, h, w]
    s, h, w = tensor.shape
    unique_labels = np.unique(tensor)
    print("Unique labels in tensor:", unique_labels)
    one_hot = np.zeros((n_classes, s, h, w))
    one_hot[0,:,:,:][tensor == 0] = 1
    one_hot[1,:,:,:][tensor == 1] = 1
    one_hot[2,:,:,:][tensor == 2] = 1
    return one_hot

def one_hot2_3d(predict):  # shape = [s, h, w]
    # d, s, h, w = predict.shape
    [a, d, w, h] = predict.shape
    predict_3D = np.zeros(( d, w, h))

    for t in range(d):
        predict_3D[t,:,:][predict[0,t,:,:] == 1] = 0
        predict_3D[t,:,:][predict[1,t,:,:] == 1] = 1
        predict_3D[t,:,:][predict[2,t,:,:] == 1] = 2
    return predict_3D

if __name__ == '__main__':
    a = torch.tensor([[1,1,0],
                      [0,1,1]])
    b = torch.tensor([[1,0,0],
                      [0,1,0]])
    r = dice_metric_binary(a,b)
    # print(r)

    a = torch.tensor([[0, 0, 1],
                      [0, 0, 0]])
    b = torch.tensor([[0, 0, 0],
                      [0, 0, 0]])
    r = dice_metric_binary(a, b)
    print('r0: ', r)