
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from scipy import ndimage as ndi
from skimage.measure import label

from code_all.networks.VNet_our_1 import VNet_CF
from utils import losses
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.utils_BCP import context_mask, mix_loss, update_ema_variables, sigmoid_rampup
import os
os.environ["TENSORBOARDX_DISABLE_COMET"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods',
                    help='Name of Experiment')  # todo change dataset path
parser.add_argument('--dataset', type=str, default="LV", help='dataset')
parser.add_argument('--exp', type=str, default="LACI", help='model_name')
parser.add_argument('--trainlist', type=str, default="train.txt", help='model_name')
parser.add_argument('--labelnum', type=int, default=32, help='trained samples')
parser.add_argument('--num_classes', type=int, default='3', help='num_classes')
parser.add_argument('--max_samples', type=int, default=369, help='maximum samples to train')
parser.add_argument('--model', type=str, default='VNet_CF', help='model_name')  ## ✅
parser.add_argument('--pre_max_iteration', type=int, default=1000, help='maximum pre-train iteration to train')  ## ✅
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')  ## ✅
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')  ## ✅
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')  ## ✅
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')  ## ✅
parser.add_argument('--patch_size', type=list, default=[112, 112, 80], help='patch size')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')  ## ✅
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')  ## ✅
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
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
    model = VNet_CF(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
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
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs].cuda()
            label_batch = sampled_batch['label'][:args.labeled_bs].cuda()
            label_batch = label_batch.long()  # CE 必须 long
            outputs = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = 0.5 * (loss_ce + loss_dice)

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))


            if iter_num % 1000 == 0:
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
    model = VNet_CF(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    ema_model = VNet_CF(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    for param in ema_model.parameters():
        param.detach_()  # ema_model set
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
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    pretrained_model = os.path.join(pre_snapshot_path, f'iter_{args.pre_max_iteration}.pth')
    print('Loading pretrained model from {}'.format(pretrained_model))
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)

    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_l, img_u = volume_batch[:args.labeled_bs], volume_batch[args.labeled_bs:]
            lab_l = label_batch[:args.labeled_bs]

            # ---------- 1) 有监督损失 ----------
            logits = model(volume_batch)
            logits_l = logits[:args.labeled_bs]
            logits_u = logits[args.labeled_bs:]
            loss_sup_ce = F.cross_entropy(logits_l, lab_l)
            loss_sup_dice = DICE(logits_l, lab_l)
            loss_sup = 0.5 * (loss_sup_ce + loss_sup_dice)
            # -----------------------------------

            # ---------- 2) Teacher 对无标签的预测 ----------
            with torch.no_grad():
                logits_u_teacher = ema_model(img_u)
                prob_u_teacher = F.softmax(logits_u_teacher, dim=1)
            prob_u_student = F.softmax(logits_u, dim=1)
            # ---------- 4) 一致性损失（soft MSE） ----------
            consistency_loss = F.mse_loss(prob_u_student, prob_u_teacher)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss_unsup = consistency_weight * consistency_loss

            loss = loss_sup + loss_unsup

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA 更新 teacher
            update_ema_variables(model, ema_model, 0.99)

            # 调度学习率
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            # # 日志记录
            # writer.add_scalar('Self/consistency_weight', consistency_weight, iter_num)
            # writer.add_scalar('Self/loss_sup', loss_sup.item(), iter_num)
            # writer.add_scalar('Self/loss_unsup', loss_unsup.item(), iter_num)
            # writer.add_scalar('Self/loss_all', loss.item(), iter_num)

            logging.info(
                'Self iteration %d : loss: %.4f, loss_sup: %.4f, loss_unsup: %.4f'
                % (iter_num, loss.item(), loss_sup.item(), loss_unsup.item())
            )

            # 保存模型
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}.pth'.format(iter_num))
                # 这里只存 state_dict，和你原来的习惯一致
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
    pre_snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}/pre_train".format(
        args.dataset, args.exp, args.labelnum)
    self_snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}/self_train".format(
        args.dataset, args.exp, args.labelnum)
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
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)



