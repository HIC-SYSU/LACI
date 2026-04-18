"""
✅ 按照验证集方法进行测试，并保存结果。测试集的处理和验证集一样，否则需要进行数据空间转换
"""

import torch.nn as nn
from monai.transforms import LoadImaged
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from torch.nn import DataParallel

from Networks.UXNet_3D.network_backbone import UXNET, UXNET_text
from monai.networks.nets import UNETR, SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
# from monai.inferers import sliding_window_inference  # , sliding_window_inference_text
# from monai.inferers.utils import sliding_window_inference_text
from monai.data import CacheDataset, DataLoader, decollate_batch
import SimpleITK as sitk
import torch
from torch.utils.tensorboard import SummaryWriter
from data_process.load_datasets_transforms_3D import data_loader, data_transforms, infer_post_transforms

import os
import numpy as np
from tqdm import tqdm
import argparse

from lib.utils.sliding_window_inference import sliding_window_inference, sliding_window_inference_text

parser = argparse.ArgumentParser(description='3D UX-Net hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/data/chenjinfeng/Data/CT_160/Xinan', required=False,
                    help='Root folder of all your images and labels')
## without text
# parser.add_argument('--output', type=str, default='/data/chenjinfeng/code/3DUX-Net/Train_out/xinan', required=False, help='Output folder for both tensorboard and the best model')
# parser.add_argument('--pretrained_weights', default='/data/chenjinfeng/code/3DUX-Net-40100/Train_out/Xinan/best_metric_model.pth', help='Path of pretrained weights')
# parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
# parser.add_argument('--dataset', type=str, default='left ventricle', required=False, help='Datasets: {left ventricle, left ventricle text}, Fyi: You can add your dataset here')
# parser.add_argument('--network', type=str, default='3DUXNET', help='Network models: {3DUXNET,3DUXNET_text}')
## with
# parser.add_argument('--output', type=str, default='/data/chenjinfeng/code/VL/demo/output/weight_xinan_text_U',
#                     required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--pretrained_weights',
                    default='/data/chenjinfeng/code/VL/demo/output/weight_xinan_text_U/best_metric_model.pth',
                    help='Path of pretrained weights')
parser.add_argument('--pretrain', default=True, help='Have pretrained weights or not')
parser.add_argument('--dataset', type=str, default='left ventricle', required=False,
                    help='Datasets: {left ventricle}, Fyi: You can add your dataset here')
parser.add_argument('--network', type=str, default='3DUXNET_text', help='Network models: {3DUXNET,3DUXNET_text}')

## Input model & training hyperparameters
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--batch_size', type=int, default='2', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=400000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=200, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='1', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.005, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# print('Used GPU: {}'.format(args.gpu))
gpu_ids = [int(gpu_id) for gpu_id in args.gpu.split(',')]
chosen_gpu_id = gpu_ids[0] if torch.cuda.is_available() and gpu_ids else 0
device = torch.device(f"cuda:{chosen_gpu_id}" if torch.cuda.is_available() else "cpu")
print(args.dataset)
test_samples, out_classes = data_loader(args)
print("test_samples:{}".format(len(test_samples["images"])))

val_files = [
    {"image": image_name, "label": label_name, "text": text_name}
    for image_name, label_name, text_name in
    zip(test_samples['images'], test_samples['labels'], test_samples['texts'])]

print(val_files)

set_determinism(seed=0)

test_transforms = data_transforms(args)

## Train Pytorch Data Loader and Caching

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(data=val_files, transform=test_transforms, cache_rate=args.cache_rate,
                      num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

## Load Networks
# device = torch.device("cuda:0")
if torch.cuda.is_available() and len(gpu_ids) > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.network == '3DUXNET':
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    )
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    print(device)
elif args.network == '3DUXNET_text':
    model = UXNET_text(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    )
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    print(device)

print("device:{}".format(device))
print('=====' * 50)
if args.pretrain == True:
    print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
    model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)


def cal_dice(prediction, label, num=3):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float64)
        label_tmp = label_tmp.astype(np.float64)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
               (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def one_hot2_3d(predict):  # shape = [s, h, w]
    # d, s, h, w = predict.shape
    [a, d, w, h] = predict.shape
    predict_3D = np.zeros((d, w, h))

    for t in range(d):
        predict_3D[t, :, :][predict[0, t, :, :].cpu() == 1] = 0
        predict_3D[t, :, :][predict[1, t, :, :].cpu() == 1] = 1
        predict_3D[t, :, :][predict[2, t, :, :].cpu() == 1] = 2
    return predict_3D

save_path_text = os.path.join(args.root, 'Train_out/3DUXNet_text')
def validation_text(epoch_iterator_val):
    # model_feat.eval()
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels, val_texts = (batch["image"].to(device), batch["label"].to(device), batch["text"].to(device))
            # val_labels = val_labels.unsqueeze(0)
            val_outputs = sliding_window_inference_text(val_inputs, val_texts,(128, 128, 96), 2, model)
            val_labels_list = decollate_batch(val_labels)       # 将batch样本分开
            print(F'val_labels:{val_labels.shape}')
            print(F'val_labels_list:{val_labels_list[0].shape}')
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]     # 将标签转为one-hot形式
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]     # 将pred进行argmax再转为one-hot形式
            # print(f'val_outputs_list:{len(val_outputs_list)}')  # 1
            # print(F'val_outputs_list_shape:{val_outputs_list[0].shape}')    # torch.Size([3, 512, 512, 217])
            # print(F'val_outputs_list:{val_output_convert[0]}')
            # print(F'val_labels_list:{val_labels_convert[0]}')
            print(f'val_inputs_shape:{val_inputs.shape}')
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice))

            ##
            name = os.path.basename(val_files[step]['image'])
            label = one_hot2_3d(val_labels_convert[0])
            pred = one_hot2_3d(val_output_convert[0])
            print(f'label:{label.shape}, pred:{pred.shape}')
            d = cal_dice(pred, label)
            print(f'dice:{d}')
            image = val_inputs.squeeze().permute(2, 1, 0).cpu().numpy()      #
            image = sitk.GetImageFromArray(image)

            # label = sitk.GetImageFromArray(label)
            # pred = sitk.GetImageFromArray(pred)
            label = sitk.GetImageFromArray(np.transpose(label, (2, 1, 0)))  # label is Numpy
            pred = sitk.GetImageFromArray(np.transpose(pred, (2, 1, 0)))
            sitk.WriteImage(image, os.path.join(save_path_text, 'img_' + name))
            sitk.WriteImage(label, os.path.join(save_path_text, 'label_' + name))
            sitk.WriteImage(pred, os.path.join(save_path_text, 'pred_' + name))

        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return dice_vals, mean_dice_val

max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
dice_vals, mean_dice_val = validation_text(epoch_iterator_val)
print(f'dice_vals:{dice_vals}, Mean dice val:{mean_dice_val}')