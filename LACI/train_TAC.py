import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import losses
from utils.utils_BCP import sigmoid_rampup
from dataloaders.dataset import *
from networks.vnet_TAC import VNet


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LA", help='dataset')
parser.add_argument('--exp', type=str,  default="TAC_flod0", help='model_name')
parser.add_argument('--trainlist', type=str,  default="train0.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='2', help='num_classes')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80], help='patch size of network input')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

args = parser.parse_args()

max_iterations = 6000
snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(args.dataset, args.exp, args.labelnum)
test_save_path = os.path.join(snapshot_path, "test/")
batch_size = 4
labeled_bs = 2
base_lr = 0.01
lr_ = base_lr
labeled_idxs = list(range(args.labelnum))
unlabeled_idxs = list(range(args.labelnum, args.max_samples))
seed = 1337
num_classes = args.num_classes
eps = 1e-6 
temperature = 2.0


if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

cudnn.benchmark = False  # True #
cudnn.deterministic = True  # False #
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    model = net.cuda()

    if args.dataset == "LA":
        db_train = LAHeart(base_dir=args.root_path,
                           split='train',
                           train_flod=args.trainlist,  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(args.patch_size),
                               ToTensor()]))
    elif args.dataset == "LV":
        print(f'undefined dataset {args.dataset}')
    
                       
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,  momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    iterator = tqdm(range(max_epoch), ncols=70)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print(volume_batch.shape, label_batch.shape)        # torch.Size([4, 1, 112, 112, 80]) torch.Size([4, 112, 112, 80])
            # label_batch = label_batch.unsqueeze(1)

            outputs, x1, x2, x3, x4, x5, x4_attention, x3_attention, x2_attention, x1_attention = model(volume_batch)

            outputs_soft = torch.sigmoid(outputs)
            # print(outputs.shape, label_batch.shape)
            label_batch = F.one_hot(label_batch, num_classes = num_classes).permute(0,4,1,2,3)

            loss_seg = ce_loss(outputs[:labeled_bs, :,:,:,:], label_batch[:labeled_bs, :, :,:,:].float())
            loss_seg_dice = 0
            for i in range(num_classes):
                loss_seg_dice += losses.dice_loss(outputs_soft[:labeled_bs, i,:,:,:], label_batch[:labeled_bs, i,:,:,:] == 1)


         ### 特征一致性损失
            shape = [[112,112,80],[56, 56, 40],[28, 28, 20],[14, 14, 10],[7, 7, 5]]
            feature = [x1, x2, x3, x4, x5]
            consistency_loss = 0
            for j in [0,1,2,3,4]:
                label_guild_batch = F.interpolate(label_batch.float(), size= shape[j], mode='nearest')
                unlabel_guild_batch = F.interpolate(outputs_soft, size= shape[j], mode='nearest')
                label_logits_soft = None
                unlabel_logits_soft = None

                for i in range(num_classes):
                    label_logits_avg = torch.sum((feature[j][:labeled_bs, :, :, :, :] * label_guild_batch[:labeled_bs, i:i+1, :, :, :]),(2,3,4)) / (torch.sum(label_guild_batch[:labeled_bs, i:i+1, :, :, :]) + eps)
                    unlabel_logits_avg = torch.sum((feature[j][labeled_bs:, :, :, :, :] * unlabel_guild_batch[labeled_bs:, i:i+1, :, :, :]),(2,3,4)) / (torch.sum(unlabel_guild_batch[labeled_bs:, i:i+1, :, :, :]) + eps)
                
                    if label_logits_soft is None:
                        label_logits_soft = label_logits_avg.unsqueeze(0)
                        unlabel_logits_soft = unlabel_logits_avg.unsqueeze(0)
                    else:
                        label_logits_soft = torch.cat((label_logits_soft, label_logits_avg.unsqueeze(0)), 0) 
                        unlabel_logits_soft = torch.cat((unlabel_logits_soft, unlabel_logits_avg.unsqueeze(0)), 0) 
                consistency_loss += F.mse_loss(unlabel_logits_soft, label_logits_soft, reduction='mean')

                ### 注意力一致性损失
            shape = [[14, 14, 10],[28, 28, 20],[56, 56, 40],[112,112,80]]
            feature = [x4_attention, x3_attention, x2_attention, x1_attention]
            consistency_attention_loss = 0
            for j in [0,1,2,3]:
                label_guild_batch = F.interpolate(label_batch.float(), size= shape[j], mode='nearest')
                unlabel_guild_batch = F.interpolate(outputs_soft, size= shape[j], mode='nearest')
                label_logits_soft = None
                unlabel_logits_soft = None

                for i in range(num_classes):
                    label_logits_avg = torch.sum((feature[j][:labeled_bs, :, :, :, :] * label_guild_batch[:labeled_bs, i:i+1, :, :, :]),(2,3,4)) / (torch.sum(label_guild_batch[:labeled_bs, i:i+1, :, :, :]) + eps)
                    unlabel_logits_avg = torch.sum((feature[j][labeled_bs:, :, :, :, :] * unlabel_guild_batch[labeled_bs:, i:i+1, :, :, :]),(2,3,4)) / (torch.sum(unlabel_guild_batch[labeled_bs:, i:i+1, :, :, :]) + eps)
                
                    if label_logits_soft is None:
                        label_logits_soft = label_logits_avg.unsqueeze(0)
                        unlabel_logits_soft = unlabel_logits_avg.unsqueeze(0)
                    else:
                        label_logits_soft = torch.cat((label_logits_soft, label_logits_avg.unsqueeze(0)), 0) 
                        unlabel_logits_soft = torch.cat((unlabel_logits_soft, unlabel_logits_avg.unsqueeze(0)), 0) 
                consistency_attention_loss += F.mse_loss(unlabel_logits_soft, label_logits_soft, reduction='mean') 
            supervised_loss = loss_seg_dice 
            loss = supervised_loss + consistency_attention_loss + 0.5 * consistency_loss 


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dc = losses.dice_loss(torch.argmax(outputs_soft[:], dim=1), torch.argmax(label_batch[:]))

            

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            print('iteration %d : loss : %f,  loss_seg: %f, loss_dice: %f, loss_consistency: %f, loss_attention_consistency: %f' %
                    (iter_num, loss.item(),  loss_seg.item(), loss_seg_dice.item(), consistency_loss.item(), consistency_attention_loss.item()))
            
                
            # if iter_num % 50 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)
            #
            #     for i in range(num_classes):
            #         image = outputs_soft[0, i:i+1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #         grid_image = make_grid(image, 5, normalize=False)
            #         writer.add_image('train/Predicted_label_'+str(i), grid_image, iter_num)
            #
            #     for i in range(num_classes):
            #         image = label_batch[0, i:i+1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #         grid_image = make_grid(image, 5, normalize=False)
            #         writer.add_image('train/Groundtruth_label_'+str(i),grid_image, iter_num)
                
            # change lr
            lr_ = poly_lr(epoch_num, max_epoch, base_lr, 0.9)
            optimizer.param_groups[0]['lr']  = lr_

            iter_num = iter_num + 1
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            #     with torch.no_grad():
            #         model.eval()
            #         avg_metric = test_all_case(model, image_list, num_classes=num_classes,
            #                 patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
            #                 save_result=True, test_save_path=test_save_path,
            #                 metric_detail=1, nms=0)
            #         print("avg_metric:", iter_num, avg_metric)
            #
            #     model.train()
            # if iter_num >= max_iterations:
            #     with torch.no_grad():
            #         model.eval()
            #         avg_metric = test_all_case(model, image_list, num_classes=num_classes,
            #                    patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
            #                    save_result=True, test_save_path=test_save_path,
            #                    metric_detail=1, nms=0)
            #         break
            #     model.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    writer.close()
