"""

"""

import sys
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from skimage.measure import label
from torch.utils.data import DataLoader

from utils.metrics import dice_metric_multiclass, calculate_metric_percase, dice_metric_multiclass_list, to_one_hot_3d
from networks.VNet_our import VNet_CF, VNet
from utils import losses, ramps, test_3d_patch
from dataloaders.dataset import *
from torch.utils.data import Subset


"""
num_all: 461 = 369(32 labeled+291 unlabeled+46 val)+92(test)
"""
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LV", help='dataset')
parser.add_argument('--exp', type=str,  default="DDT_flod0_adam", help='model_name')
parser.add_argument('--trainlist', type=str,  default="train.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=32, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='3', help='num_classes')
parser.add_argument('--max_samples', type=int,  default=369, help='maximum samples to train')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')       ## ✅
parser.add_argument('--self_max_iteration', type=int,  default=150000, help='maximum self-train iteration to train')       ## ✅
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')       ## ✅
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')       ## ✅
parser.add_argument('--patch_size', type=list, default=[112, 112, 80],help='patch size')
parser.add_argument('--deterministic', type=int,  default=1, help='确保实验的可重复性')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=2024, help='random seed')      ## 1337
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
# parser.add_argument('--file_path', type=str, default="/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_280_200/LV_features_dict.pth", help='LLM_TExt')
args = parser.parse_args()

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


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
def dice_loss_binary(pred, target):
    smooth = 1e-8
    num = pred.size(0)
    m1 = pred.reshape(num, -1)  # Flatten
    m2 = target.reshape(num, -1)  # Flatten
    intersection = m1 * m2
    intersection = torch.sum(intersection, dim=1, keepdim=True)
    m1 = torch.sum(m1, dim=1, keepdim=True)
    m2 = torch.sum(m2, dim=1, keepdim=True)
    loss = (2. * intersection + smooth) / (m1 + m2 + smooth)
    return 1 - torch.mean(loss, dim=0, keepdim=False)

def dice_loss_multiclass(input, target):
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_loss_binary(input[:, channel, ...], target[:, channel, ...])
    return dice / input.shape[1]

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)


if args.deterministic:
    cudnn.benchmark = False     ## 禁用cuDNN的自动调优：cuDNN的自动调优会尝试多种算法来找出最快的，但这可能会导致运行结果的不确定性
    cudnn.deterministic = True      ## 强制使用过确定性算法，确保可重复性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = args.patch_size
num_classes = args.num_classes


def pre_train(args, snapshot_path):
    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = VNet_CF(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()

    if args.dataset == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=args.trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()]))

        # # 创建一个包含前4个元素的子集
        # db_train = Subset(db_train, list(range(4)))

    elif args.dataset == "LV":
        db_train = LVHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=args.trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()]))

        file_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/LV_features_dict.pth"
        features_dict = torch.load(file_path)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        dice_score = 0
        epoch_loss = 0
        dice_avg_ = 0
        for iteration, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, name_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs], sampled_batch['name'][:args.labeled_bs]
            # print(name_batch, volume_batch.shape, label_batch.shape)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # unique_labels = torch.unique(label_batch)
            label_batch_onehot = F.one_hot(label_batch, num_classes).permute(0, 4, 1, 2, 3).float()
            # unique_labels = torch.unique(label_batch_onehot)

            f_describe = [features_dict[name].cuda().float() for name in name_batch]        ## os.path.basename(name)
            out_seg, out_seg_c, out_seg_cc = model(volume_batch, f_describe)

            outputs_soft1 = F.softmax(out_seg, dim=1)
            outputs_soft2 = F.softmax(out_seg_c, dim=1)
            outputs_soft3 = F.softmax(out_seg_cc, dim=1)

            # calculate the loss
            # supervised loss
            loss_sup1 = dice_loss_multiclass(outputs_soft1, label_batch_onehot)
            loss_sup2 = dice_loss_multiclass(outputs_soft1, label_batch_onehot)
            loss_sup3 = dice_loss_multiclass(outputs_soft1, label_batch_onehot)
            loss_sup = loss_sup1 + 0.5 * loss_sup2 + 0.5 * loss_sup3

            # unsupervised loss - - MSE loss
            # 使用 KL 散度作为无监督一致性损失
            # loss_cons_l = F.kl_div(outputs_soft1.log(), outputs_soft2, reduction="batchmean")
            # loss_cons_l1 = F.kl_div(outputs_soft1.log(), outputs_soft3, reduction="batchmean")
            loss_cons_l = mse_loss(outputs_soft1, outputs_soft2)
            loss_cons_l1 = mse_loss(outputs_soft1, outputs_soft3)
            loss_cons = loss_cons_l + loss_cons_l1

            ## CE
            # loss_CE1 = F.cross_entropy(out_seg, label_batch_onehot)
            # loss_CE2 = F.cross_entropy(out_seg_c, label_batch_onehot)
            # loss_CE3 = F.cross_entropy(out_seg_cc, label_batch_onehot)
            loss_CE1 = F.cross_entropy(out_seg, label_batch)
            loss_CE2 = F.cross_entropy(out_seg_c, label_batch)
            loss_CE3 = F.cross_entropy(out_seg_cc, label_batch)
            loss_CE = loss_CE1 + 0.5 * loss_CE2 + 0.5 * loss_CE3
            loss = loss_sup + loss_cons + loss_CE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # calculate the dice
            # mean_dices, MYo, LV = dice_metric_multiclass_list(outputs_soft1, label_batch_onehot)

            # dice_avg = torch.mean(torch.tensor(mean_dices))
            # dice_avg_ += dice_avg  # 累加每个批次的平均 Dice
            # print(dice_score)
            # dice_scores_str = ', '.join(f"{d.item():.3f}" for d in dice_score)
            logging.info('\titeration %d : loss: %03f, dice_loss: %03f, mse_loss: %03f, ce_loss: %03f' % (iter_num, loss, loss_sup, loss_cons, loss_CE))

            iter_num += 1
            writer.add_scalar('Pre/loss_dice', loss_sup, iter_num)
            writer.add_scalar('Pre/loss_mse', loss_cons, iter_num)
            writer.add_scalar('Pre/loss_all', loss, iter_num)

            epoch_loss += loss.item()

            if iter_num%1000==0:
                save_mode_path = os.path.join(snapshot_path, 'pre_iter_{}.pth'.format(iter_num))
                save_net_opt(model, optimizer, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= self_max_iterations:
                break

        # 计算平均损失并记录
        avg_epoch_loss = epoch_loss / len(trainloader)
        # avg_epoch_dice = dice_avg_ / (len(trainloader)/2)

        logging.info('Epoch %d : Average Loss: %03f, Average Dice: %03f' % (epoch, avg_epoch_loss, avg_epoch_dice))
        writer.add_scalar('Pre/avg_epoch_loss', avg_epoch_loss, epoch)
        # writer.add_scalar('Pre/avg_epoch_dice', avg_epoch_dice, epoch)

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args, pre_snapshot_path,snapshot_path):
    model = VNet_CF(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    pretrained_model = os.path.join(pre_snapshot_path, f'pre_iter_{args.pre_max_iteration}.pth')
    load_net(model, pretrained_model)
    if args.dataset == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           train_flod=args.trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor()]))

        # # 创建一个包含前4个元素的子集
        # db_train = Subset(db_train, list(range(4)))
    elif args.dataset == "LV":
        db_train = LVHeart(base_dir=train_data_path,
                       split='train',
                       train_flod=args.trainlist,  # todo change training flod
                       common_transform=transforms.Compose([
                           RandomCrop(patch_size),
                           ToTensor()]))

        file_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/LV_features_dict.pth"
        features_dict = torch.load(file_path)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        dice_score = 0
        epoch_loss = 0
        dice_avg_ = 0
        for iteration, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, name_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['name']
            # print(name_batch)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch_onehot = F.one_hot(label_batch, 3).permute(0, 4, 1, 2, 3).float()
            # unique_labels = np.unique(label_batch)
            # print("Unique labels in tensor:", unique_labels)
            f_describe = [features_dict[name].cuda().float() for name in name_batch]
            out_seg, out_seg_c, out_seg_cc = model(volume_batch, f_describe)
            # print(out_seg.shape, label_batch_onehot.shape)

            outputs_soft1 = F.softmax(out_seg, dim=1)
            outputs_soft2 = F.softmax(out_seg_c, dim=1)
            outputs_soft3 = F.softmax(out_seg_cc, dim=1)

            # calculate the loss
            # supervised loss
            loss_sup1 = dice_loss_binary(outputs_soft1[:args.labeled_bs],
                                         label_batch_onehot[:args.labeled_bs])
            loss_sup2 = dice_loss_binary(outputs_soft2[:args.labeled_bs],
                                         label_batch_onehot[:args.labeled_bs])
            loss_sup3 = dice_loss_binary(outputs_soft3[:args.labeled_bs],
                                         label_batch_onehot[:args.labeled_bs])
            loss_sup = loss_sup1 + 0.5 * loss_sup2 + 0.5 * loss_sup3

            # unsupervised loss - - MSE loss
            loss_cons_l = mse_loss(outputs_soft1[:args.labeled_bs], outputs_soft2[:args.labeled_bs])
            loss_cons_u = mse_loss(outputs_soft1[args.labeled_bs:], outputs_soft2[args.labeled_bs:])
            loss_cons_l1 = mse_loss(outputs_soft1[:args.labeled_bs], outputs_soft3[:args.labeled_bs])
            loss_cons_u1 = mse_loss(outputs_soft1[args.labeled_bs:], outputs_soft3[args.labeled_bs:])
            loss_cons = loss_cons_l + 0.5 * loss_cons_u + loss_cons_l1 + 0.5 * loss_cons_u1

            ## CE
            loss_CE1 = F.cross_entropy(out_seg[:args.labeled_bs], label_batch_onehot[:args.labeled_bs])
            loss_CE2 = F.cross_entropy(out_seg_c[:args.labeled_bs], label_batch_onehot[:args.labeled_bs])
            loss_CE3 = F.cross_entropy(out_seg_cc[:args.labeled_bs], label_batch_onehot[:args.labeled_bs])
            loss_CE = loss_CE1 + 0.5 * loss_CE2 + 0.5 * loss_CE3
            loss = loss_sup + loss_cons + loss_CE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # calculate the dice
            # dice_score, dice_avg = dice_metric_multiclass_list(outputs_soft1, label_batch_onehot)
            # dice_avg_ += dice_avg
            # print(dice_score)
            # dice_scores_str = ', '.join(f"{d.item():.3f}" for d in dice_score)
            logging.info(
                '\titeration %d : loss: %03f, dice_loss: %03f, mse_loss: %03f, ce_loss: %03f' % (
                iter_num, loss, loss_sup, loss_cons, loss_CE))

            iter_num += 1
            writer.add_scalar('Self/loss_dice', loss_sup, iter_num)
            writer.add_scalar('Self/loss_mse', loss_cons, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            epoch_loss += loss.item()

            # if iter_num % 1000 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 1000)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_

            if iter_num%1000==0:
                save_mode_path = os.path.join(snapshot_path, 'self_iter_{}.pth'.format(iter_num))
                save_net_opt(model, optimizer, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

                # model.eval()
                # if args.dataset == "LA":
                #     with open('/data/chenjinfeng/code/semi_supervised/All_code/dataset/LA/Flods/test0.list', 'r') as f:  # todo change test flod
                #         image_list = f.readlines()
                #     image_list = ['/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/' + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
                # elif args.dataset == "LV":
                #     print(f'undefined')
                # dice_sample = test_3d_patch.var_all_case(model, image_list, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                # if dice_sample > best_dice:
                #     best_dice = round(dice_sample, 4)
                #     save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                #     save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                #     # save_net_opt(model, optimizer, save_mode_path)
                #     # save_net_opt(model, optimizer, save_best_path)
                #     torch.save(model.state_dict(), save_mode_path)
                #     torch.save(model.state_dict(), save_best_path)
                #     logging.info("save best model to {}".format(save_mode_path))
                # writer.add_scalar('8_Var_dice/Dice', dice_sample, iter_num)
                # writer.add_scalar('8_Var_dice/Best_dice', best_dice, iter_num)
                # model.train()


            if iter_num >= self_max_iterations:
                break

        # 计算平均损失并记录
        avg_epoch_loss = epoch_loss / len(trainloader)
        # avg_epoch_dice = dice_avg_ / len(trainloader)

        logging.info('Epoch %d : Average Loss: %03f' % (epoch, avg_epoch_loss))
        writer.add_scalar('Self/avg_epoch_loss', avg_epoch_loss, epoch)
        # writer.add_scalar('Self/avg_epoch_dice', avg_epoch_dice, epoch)

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()



if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}/pre_train".format(args.dataset, args.exp, args.labelnum)
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
    shutil.copy('/data/chenjinfeng/code/semi_supervised/All_code/code_all/train_LV.py', self_snapshot_path)
    # # # -- Pre-Training
    # logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
