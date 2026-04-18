"""zmax——inter= 30k"""

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
# from torchvision.utils import make_grid
from tqdm import tqdm


from utils import losses
from utils.utils_BCP import sigmoid_rampup
from dataloaders.dataset import *
from networks.unet_3D_UPRC import unet_3D_dv_semi
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LV", help='dataset')
parser.add_argument('--exp', type=str,  default="UPRC", help='model_name')
parser.add_argument('--trainlist', type=str,  default="train.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=32, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='3', help='num_classes')
parser.add_argument('--max_samples', type=int,  default=369, help='maximum samples to train')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80], help='patch size of network input')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')

parser.add_argument('--model', type=str,
                    default='unet_3D_dv_semi', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
# parser.add_argument('--labeled_num', type=int, default=18, help='labeled data')
# parser.add_argument('--total_labeled_num', type=int, default=180, help='total labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=400.0, help='consistency_rampup')
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
    num_classes = args.num_classes
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    trainlist = args.trainlist
    max_iterations = args.max_iterations

    net = unet_3D_dv_semi(n_classes=num_classes, in_channels=1)
    model = net.cuda()

    if args.dataset == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()]))
    elif args.dataset == "LV":
        db_train = LVHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=args.trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.Dice_Loss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4,  = model(volume_batch)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)      # 将输出转化为概率
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
            outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)

            ## 监督每一个中间层的输出
            loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs], label_batch[:args.labeled_bs])       # 本身已经包含softmax操作，如传入概率可能会造成数值不稳定
            loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_ce_aux4 = ce_loss(outputs_aux4[:args.labeled_bs], label_batch[:args.labeled_bs])

            loss_dice_aux1 = dice_loss(outputs_aux1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))       # 基于概率的损失
            loss_dice_aux2 = dice_loss(outputs_aux2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux3 = dice_loss(outputs_aux3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux4 = dice_loss(outputs_aux4_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            supervised_loss = (loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4 + loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4)/8

            preds = (outputs_aux1_soft + outputs_aux2_soft+outputs_aux3_soft+outputs_aux4_soft)/4       # 通过平均化减少单个分支输出的偏差和不稳定性


            ## ## 不确定性估计
            variance_aux1 = torch.sum(kl_distance(torch.log(outputs_aux1_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)        # KL散度度量该输出与平均预测结之间的差异 - - 求和
            exp_variance_aux1 = torch.exp(-variance_aux1)           # 方差越大，权重因子越小，表示模型对这个输出信心较低

            variance_aux2 = torch.sum(kl_distance(torch.log(outputs_aux2_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(torch.log(outputs_aux3_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            variance_aux4 = torch.sum(kl_distance(torch.log(outputs_aux4_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)

            consistency_weight = get_current_consistency_weight(iter_num//150)

                ### 不确定性修正 - - 公式5
            consistency_dist_aux1 = (preds[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2        # 方差定义一致性距离，该距离度量了辅助输出和平均预测之间的距离，与上述的区别是？？
            consistency_loss_aux1 = torch.mean(consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (preds[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (preds[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_dist_aux4 = (preds[args.labeled_bs:] - outputs_aux4_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux4 = torch.mean(consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)

            consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, supervised_loss: %f' %
                (iter_num, loss.item(), supervised_loss.item()))

            # if iter_num % 20 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     image = torch.argmax(outputs_aux1_soft, dim=1, keepdim=True)[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label', grid_image, iter_num)
            #
            #     image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label', grid_image, iter_num)

            if iter_num > 0 and iter_num % 500 == 0:
                save_mode_path = os.path.join(snapshot_path, 'UPRC_iter_{}.pth'.format(iter_num))
                torch.save(model.state_dict(), save_mode_path)
                # model.eval()
                # avg_metric = test_all_case(model, args.root_path, test_list="val.txt", num_classes=num_classes, patch_size=args.patch_size, stride_xy=64, stride_z=64)
                # if avg_metric[:, 0].mean() > best_performance:
                #     best_performance = avg_metric[:, 0].mean()
                #     save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                #     save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                #     torch.save(model.state_dict(), save_mode_path)
                #     torch.save(model.state_dict(), save_best)
                # for cls in range(1, num_classes):
                #     writer.add_scalar('info/val_cls_{}_dice_score'.format(cls), avg_metric[cls - 1, 0], iter_num)
                #     writer.add_scalar('info/val_cls_{}_hd95'.format(cls), avg_metric[cls - 1, 1], iter_num)
                # writer.add_scalar('info/val_mean_dice_score', avg_metric[:, 0].mean(), iter_num)
                # writer.add_scalar('info/val_mean_hd95', avg_metric[:, 1].mean(), iter_num)
                # logging.info('iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
                # model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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

    snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(args.dataset, args.exp, args.labelnum)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
