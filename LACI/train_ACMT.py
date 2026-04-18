"""
AC_MT原始代码，更改了数据路径，✅

For LA segmentation, we adopt the same 3D V-Net backbone (Milletari et al., 2016) as in Yu et al. (2019), Li et al. (2020b), Luo et al. (2021). Consistently, we randomly crop patches of 112 × 112 × 80 voxels as the input and use sliding window strategy with stride of 18 × 18 × 4 voxels for prediction. 𝑡max is set to 6,000.
For brain tumor segmentation, we follow (Xu et al., 2022c) that uses the 3D U-Net backbone (Çiçek et al., 2016) for a fair comparison. We randomly crop patches of 96 × 96 × 96 voxels as the input and use sliding window strategy with stride of 64 × 64 × 64 voxels for prediction. 𝑡max is set to 20,000.

删除了验证集
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
# CL
import cleanlab
from utils import losses
from utils.utils_BCP import sigmoid_rampup
from dataloaders.dataset import *
from networks.vnet_ACMT import VNet


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LA", help='dataset')
parser.add_argument('--exp', type=str,  default="ACMT_flod0", help='model_name')
parser.add_argument('--trainlist', type=str,  default="train0.txt", help='model_name')
parser.add_argument('--labeled_num', type=int,  default=8, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='2', help='num_classes')
parser.add_argument('--total_sample', type=int,  default=80, help='maximum samples to train')
##
parser.add_argument('--model', type=str, default='vnet_ACMT', help='model_name')
parser.add_argument('--max_iterations', type=int, default=8000, help='maximum epoch number to train')       ##6k
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
# CL
parser.add_argument('--CL_type', type=str, default='both', help='CL implement type')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    max_iterations = args.max_iterations
    num_classes = args.num_classes
    trainlist = args.trainlist

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    if args.dataset == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()]))
    elif args.dataset == "LV":
        print(f'undefined dataset {args.dataset}')

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_sample))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.Dice_Loss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            noisy_ema_inputs = unlabeled_volume_batch + noise
            ema_inputs = unlabeled_volume_batch

            outputs = model(volume_batch)       ## student正常输入
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_onehot = torch.argmax(outputs_soft, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)  ## teacher正常输入
                ema_output_soft = torch.softmax(ema_output, dim=1)
                noisy_ema_output = ema_model(noisy_ema_inputs)      ## teacher含噪音输入
                noisy_ema_output_soft = torch.softmax(noisy_ema_output, dim=1)
			
      			# S-Err. (using Cleanlab v2)
            # still a bit slow
            outputs_onehot_np = outputs_onehot[args.labeled_bs:].cpu().detach().numpy()
            ema_output_soft_np = ema_output_soft.cpu().detach().numpy()
            ema_output_soft_np_accumulated = np.swapaxes(ema_output_soft_np, 1, 2)      ## 交换轴
            ema_output_soft_np_accumulated = np.swapaxes(ema_output_soft_np_accumulated, 2, 3)
            ema_output_soft_np_accumulated = np.swapaxes(ema_output_soft_np_accumulated, 3, 4)
            ema_output_soft_np_accumulated = ema_output_soft_np_accumulated.reshape(-1, num_classes)
            ema_output_soft_np_accumulated = np.ascontiguousarray(ema_output_soft_np_accumulated)       ## 确保内存布局是连续的

            outputs_onehot_np_accumulated = outputs_onehot_np.reshape(-1).astype(np.uint8)
			
            assert outputs_onehot_np_accumulated.shape[0] == ema_output_soft_np_accumulated.shape[0]
			
            CL_type = args.CL_type
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = losses.softmax_mse_loss(outputs[args.labeled_bs:], noisy_ema_output)  # (batch, 2, 112,112,80)

            try:
                ## 通过Cleanlab库的find_label_issues函数识别潜在的标签错误或噪声，这是提高模型对噪声鲁棒性的一种策略。
	            # 应用噪声信息构建掩码：根据识别出的噪声信息，构建一个掩码（mask），用于在计算一致性损失时忽略噪声标签。
	            # 计算一致性损失：使用掩码加权的方法计算一致性损失，即只在模型预测与EMA输出一致的部分计算损失。
                #print('Potential Noise Num:', cleanlab.count.num_label_issues(outputs_onehot_np_accumulated, ema_output_soft_np_accumulated))                
                if CL_type in ['both']:
                    noise = cleanlab.filter.find_label_issues(outputs_onehot_np_accumulated, ema_output_soft_np_accumulated, filter_by='both', n_jobs=1)
                elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                    noise = cleanlab.filter.find_label_issues(outputs_onehot_np_accumulated, ema_output_soft_np_accumulated, filter_by=CL_type, n_jobs=1)
            
                confident_maps_np = noise.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(np.uint8)
                confident_maps = torch.from_numpy(confident_maps_np).cuda(outputs_soft.device.index)

                mask = (confident_maps == 1).float() 
                mask = torch.unsqueeze(mask, 1) 
                mask = torch.cat((mask, mask), 1)
                
                # ambiguity-selective CR
                consistency_loss = torch.sum(mask*consistency_dist)/(torch.sum(mask)+1e-16)
            
            except Exception as e:
                print('fail to identify errors in this batch; change to typical CR loss')
                consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - noisy_ema_output_soft) ** 2)

            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            # if iter_num % 20 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
            #         3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label',
            #                      grid_image, iter_num)
            #
            #     image = label_batch[0, :, :, 20:61:10].unsqueeze(
            #         0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label',
            #                      grid_image, iter_num)

            # The LA benchmark reports the results of val set
            # if iter_num > 1 and iter_num % 1 == 0:
            # if iter_num > 500 and iter_num % 50 == 0:
            #     model.eval()
            #     if args.dataset == "LA":
            #         with open('/data/chenjinfeng/code/semi_supervised/All_code/dataset/LA/Flods/test0.list', 'r') as f:  # todo change test flod
            #             image_list = f.readlines()
            #         image_list = ['/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/' + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
            #     elif args.dataset == "LV":
            #         print(f'undefined')
            #     avg_metric = test_all_case(
            #         model, image_list, num_classes=num_classes, patch_size=args.patch_size,
            #         stride_xy=18, stride_z=4)
            #     if avg_metric[:, 0].mean() > best_performance:
            #         best_performance = avg_metric[:, 0].mean()
            #         save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
            #         save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
            #         torch.save(model.state_dict(), save_mode_path)
            #         torch.save(model.state_dict(), save_best)
            #
            #     writer.add_scalar('info/val_dice_score', avg_metric[0, 0], iter_num)
            #     writer.add_scalar('info/val_hd95', avg_metric[0, 1], iter_num)
            #     logging.info('iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
            #     model.train()
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(args.dataset, args.exp, args.labeled_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
