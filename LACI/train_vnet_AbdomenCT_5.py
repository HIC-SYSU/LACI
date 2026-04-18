import os
import sys
import random
import logging
import argparse
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dataloaders.dataset import LRVHeart, LRVImageCHDHeart, RandomCrop_LRV, ToTensor_LRV, AbdomenCT, RandomCrop, \
    ToTensor
from utils import losses
from networks.net_factory import net_factory


# -----------------------------------------------------
# 1. 参数定义
# -----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/preprocessing/MMWHS', help='dataset root')
parser.add_argument('--dataset', type=str, default='Abdomenct', help='dataset name')
parser.add_argument('--exp', type=str, default='Vnet', help='experiment name')
parser.add_argument('--labelnum', type=int, default=40, help='number of labeled samples')
# parser.add_argument('--ratio_labeled', type=str,  default="5r.txt", help='model_name')
parser.add_argument('--num_classes', type=int, default=5, help='number of segmentation classes')
parser.add_argument('--model', type=str, default='VNet', help='model type (e.g. VNet)')
parser.add_argument('--max_iteration', type=int, default=6000, help='max training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01, help='base learning rate')
parser.add_argument('--patch_size', type=list, default=[112, 112, 80], help='training patch size')
parser.add_argument('--deterministic', type=int, default=1, help='use deterministic training')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
args = parser.parse_args()


# -----------------------------------------------------
# 2. 基础设置
# -----------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
train_data_path = args.root_path
num_classes = args.num_classes
base_lr = args.base_lr
max_iterations = args.max_iteration
patch_size = args.patch_size

# 随机性控制
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# 输出目录
snapshot_path = f"/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{args.dataset}/{args.exp}/{args.labelnum}/"
os.makedirs(snapshot_path, exist_ok=True)

# 日志文件
logging.basicConfig(filename=os.path.join(snapshot_path, 'train.log'),
                    level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s',
                    datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# -----------------------------------------------------
# 3. 工具函数
# -----------------------------------------------------
def save_net_opt(net, optimizer, path):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, path)


# -----------------------------------------------------
# 4. 监督训练函数
# -----------------------------------------------------
def train_supervised(args, snapshot_path):
    logging.info("Starting supervised V-Net training...")
    logging.info(f"Dataset: {args.dataset} | Model: {args.model} | Classes: {args.num_classes}")

    # 1️⃣ 模型构建
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = model.cuda()
    model.train()

    # # 2️⃣ 数据加载
    # if args.dataset in ["LRV_ImageCHD", "MMWHS", "MMWHS_MR_pre"]:
    #     labeled_ds = LRVImageCHDHeart(
    #         base_dir=train_data_path,
    #         split="train_pre.txt",
    #         transform=transforms.Compose([
    #             RandomCrop_LRV(patch_size),
    #             ToTensor_LRV(),
    #         ])
    #     )
    #     db_train = ConcatDataset([labeled_ds])
    if args.dataset == "LRV":
        labeled_ds = LRVHeart(
            base_dir=train_data_path,
            split=args.ratio_labeled,
            transform=transforms.Compose([
                RandomCrop_LRV(patch_size),
                ToTensor_LRV(),
            ])
        )
        db_train = ConcatDataset([labeled_ds])
    elif args.dataset == "LRV_ImageCHD" or args.dataset == "MMWHS":
        labeled_ds = LRVImageCHDHeart(base_dir=train_data_path,
                                      split="train_labeled_list.txt",
                                      transform=transforms.Compose([
                                          # RandomRotFlip(),
                                          RandomCrop_LRV(patch_size),
                                          ToTensor_LRV(),
                                      ]))
        db_train = ConcatDataset([labeled_ds])
    elif args.dataset == 'Abdomenct':
        db_train = AbdomenCT(base_dir="/data/xingshihanxiao/Pyproject/Data/open_data/AbdomenCT",
                             split='train',
                             # train_flod=args.trainlist,
                             transform=transforms.Compose([
                                 RandomCrop(patch_size),
                                 ToTensor()]))



    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Subset(db_train, list(range(args.labelnum)))
    trainloader = DataLoader(db_train,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True,
                             worker_init_fn=worker_init_fn)


    logging.info(f"Training set size: {len(trainloader.dataset)} samples")
    logging.info(f"Iterations per epoch: {len(trainloader)}")

    # 3️⃣ 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DICELoss(nclass=num_classes)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    # 4️⃣ 训练循环
    iter_num = 0
    best_loss = 1e9
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info(f"Total epochs: {max_epoch}, Total iterations: {max_iterations}")

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        epoch_loss = 0.0

        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].cuda(non_blocking=True)
            label_batch = sampled_batch['label'].cuda(non_blocking=True).long()
            unique_labels = torch.unique(label_batch).detach().cpu().numpy()
            # 前向传播
            outputs = model(volume_batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # preds = torch.argmax(outputs, dim=1)  # shape [B, H, W, D]
            # unique_preds = torch.unique(preds).detach().cpu().numpy()
            # print("Unique predicted classes in current batch:", unique_preds)

            # unique_outputs = torch.unique(outputs).detach().cpu().numpy()
            # print("Unique labels in current batch:", unique_labels, unique_outputs)

            # 计算损失
            probs = F.softmax(outputs, dim=1)
            loss_ce = ce_loss(outputs, label_batch)  # CE 仍用 logits
            loss_dice = dice_loss(probs, label_batch)  # Dice 用 softmax 后的概率
            loss = 0.5 * (loss_ce + loss_dice)

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # 日志记录
            iter_num += 1
            epoch_loss += loss.item()

            if iter_num % 20 == 0:
                logging.info(f"[Iter {iter_num}] total={loss.item():.4f} ce={loss_ce.item():.4f} dice={loss_dice.item():.4f}")

            writer.add_scalar('train/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iter_num)
            writer.add_scalar('train/loss_total', loss.item(), iter_num)

            # 模型保存
            if iter_num % 1000 == 0 or iter_num == max_iterations:
                save_path = os.path.join(snapshot_path, f'iter_{iter_num}.pth')
                save_net_opt(model, optimizer, save_path)
                logging.info(f"Model saved: {save_path}")

            if iter_num >= max_iterations:
                break

        avg_epoch_loss = epoch_loss / len(trainloader)
        logging.info(f"Epoch {epoch_num+1}/{max_epoch} | avg_loss={avg_epoch_loss:.4f}")

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    logging.info("✅ Training finished successfully.")


# -----------------------------------------------------
# 5. 主入口
# -----------------------------------------------------
if __name__ == "__main__":
    train_supervised(args, snapshot_path)
