from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
from scipy import ndimage as ndi
from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.utils_BCP import context_mask, mix_loss, update_ema_variables, sigmoid_rampup

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LV", help='dataset')
parser.add_argument('--exp', type=str,  default="BCP", help='model_name')
parser.add_argument('--trainlist', type=str,  default="train.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=16, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='3', help='num_classes')
parser.add_argument('--max_samples', type=int,  default=369, help='maximum samples to train')
parser.add_argument('--model', type=str, default='VNet_BCP', help='model_name')       ## ✅
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')       ## ✅
parser.add_argument('--self_max_iteration', type=int,  default=6000, help='maximum self-train iteration to train')       ## ✅
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')       ## ✅
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')       ## ✅
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')       ## ✅
parser.add_argument('--patch_size', type=list, default=[112,112,80],help='patch size')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')       ## ✅
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')       ## ✅
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()


def get_cut_mask(out, thres=0.5, nms=0, cls_list=[1, 2]):
    probs = F.softmax(out, 1)
    masks_list = []
    b, c, h, w, d = probs.shape
    mask1 = torch.zeros(b, h, w, d).cuda()
    mask1[:, :, :, :] = 0  # Ensure the mask is initialized to 0

    for cls in cls_list:
        masks = (probs >= thres).type(torch.int64)
        masks = masks[:, cls, :, :].contiguous()
        if nms == 1:
            masks = LargestCC_pancreas(masks)
        masks_list.append(masks)
        mask1[masks[:, :, :, :] == 1] = cls
    unique_labels = torch.unique(mask1)
    # print(unique_labels, mask1.shape)
    return mask1


def get_cut_mask_0(out, thres=0.5, nms=0, cls=1):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, cls, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def get_cut_mask1(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    # print(out.shape, masks.shape, probs.shape)
    masks = masks[:, 1:, :, :, :].contiguous()
    # print(masks.shape)
    if nms == 1:
        masks = LargestCC_multi(masks)
    b, c, h, w, d = masks.shape
    mask1 = torch.zeros(b,h,w,d).cuda()
    mask1[:, :, :, :] = 0  # Ensure the mask is initialized to 0
    mask1[masks[:, 0, :, :, :] == 1] = 1  # Class 1 (first target class)
    mask1[masks[:, 1, :, :, :] == 1] = 2  # Class 2 (second target class)

    return mask1


def LargestCC_multi(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        largest_cc_prediction = np.zeros_like(n_prob)
        for cls in range(1, num_classes):  # 假设 0 为背景类，忽略
            # 对每个类别生成二值掩码
            binary_mask = (n_prob == cls)
            # 标记连通区域
            labeled_mask, num_features = ndi.label(binary_mask)

            if num_features > 0:
                # 计算每个连通区域的大小
                sizes = ndi.sum(binary_mask, labeled_mask, range(1, num_features + 1))
                # 获取最大连通区域的标签
                largest_cc_label = np.argmax(sizes) + 1
                # 保留最大连通区域
                largest_cc_prediction[labeled_mask == largest_cc_label] = cls
        batch_list.append(largest_cc_prediction)

    return torch.Tensor(batch_list).cuda()

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path
trainlist = args.trainlist

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = args.patch_size
num_classes = args.num_classes

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
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

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=3)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio, args.patch_size)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, features = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)     # ce损失直接使用logits。不需要softmax
            loss_dice = DICE(outputs, label_batch)      # dice损失需要输入概率分布，需要softmax
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))

            # if iter_num % 200 == 0:
            #     model.eval()
            #     image_path = '/data/chenjinfeng/code/semi_supervised/All_code/dataset/LA/Flods/test0.list'
            #     with open(image_path, 'r') as f:
            #         image_list = f.readlines()
            #     image_list = [
            #         '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/' + item.replace(
            #             '\n', '') + "/mri_norm2.h5" for item in image_list]
            # 
            #     dice_sample = test_3d_patch.var_all_case(model, image_list, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
            #     if dice_sample > best_dice:
            #         best_dice = round(dice_sample, 4)
            #         save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
            #         save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
            #         save_net_opt(model, optimizer, save_mode_path)
            #         save_net_opt(model, optimizer, save_best_path)
            #         # torch.save(model.state_dict(), save_mode_path)
            #         # torch.save(model.state_dict(), save_best_path)
            #         logging.info("save best model to {}".format(save_mode_path))
            #     writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
            #     writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
            #     model.train()
            if iter_num%1000==0:
                save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(iter_num))
                save_net_opt(model, optimizer, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
            param.detach_()   # ema_model set
    if args.dataset == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()
                           ]))
    elif args.dataset == "LV":
        db_train = LVHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=args.trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'iter_{args.pre_max_iteration}.pth')
    print('load pretrained model from {}'.format(pretrained_model))
    # "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/LV/BCP/16/pre_train/iter_2000.pth"
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)
    
    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1, cls_list=[1,2])
                plab_b = get_cut_mask(unoutput_b, nms=1)
                unique_labels_a, unique_b = torch.unique(lab_a), torch.unique(lab_b)
                print(unique_labels_a, unique_b)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio, args.patch_size)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
            outputs_l, _ = model(mixl_img)
            outputs_u, _ = model(mixu_img)
            unique = torch.unique(plab_a)
            # print(outputs_l.shape, lab_a.shape, plab_a.shape, loss_mask.shape, unique)
            loss_l = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)

            loss = loss_l + loss_u

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f'%(iter_num, loss, loss_l, loss_u))

            update_ema_variables(model, ema_model, 0.99)

             # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num%1000==0:
                save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(iter_num))
                # save_net_opt(model, optimizer, save_mode_path)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))


            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}/pre_train".format(args.dataset, args.exp,args.labelnum)
    self_snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}/self_train".format(args.dataset, args.exp, args.labelnum)
    print("Starting BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        train_log = os.path.join(snapshot_path, 'train.log')
        test_log = os.path.join(snapshot_path, 'test.log')
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if not os.path.exists(train_log):
            with open(train_log, 'w') as f:
                f.write("Iteration, loss, loss_dice, loss_ce\n")
        if not os.path.exists(test_log):
            with open(test_log, 'w') as f:
                f.write("Iteration, Dice Coefficient, Jaccard Index, Hausdorff Distance, Average Surface Distance\n")
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('/data/chenjinfeng/code/semi_supervised/All_code/code_all/train_BCP_LV.py', self_snapshot_path)
    # # -- Pre-Training
    # logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
