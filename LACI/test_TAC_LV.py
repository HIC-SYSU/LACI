import argparse
import os
import shutil
from glob import glob

import torch

from networks.vnet_TAC import VNet
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LV", help='dataset')
parser.add_argument('--exp', type=str,  default="TAC", help='model_name')
parser.add_argument('--testlist', type=str,  default="test.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=32, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='3', help='num_classes')
parser.add_argument('--model', type=str, default='unet_3D', help='model_name')
parser.add_argument('--patch_size', type=list, default=[112,112,80],help='patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='stride_xy')
parser.add_argument('--stride_z', type=int, default=4, help='stride_z')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
if FLAGS.dataset == "LA":
    with open('/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/'+FLAGS.testlist, 'r') as f:  # todo change test flod
        image_list = f.readlines()
    image_list = [item.replace('\n', '') + "/mri_norm_new.h5" for item in image_list]
elif FLAGS.dataset == "LV":
    with open('/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/test.txt',
              'r') as f:  # todo change test flod
        image_list = f.readlines()
        print(image_list)
    image_list = [item.replace('\n', '') for item in image_list]


def Inference(FLAGS):
    iter_num = 6000
    snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(FLAGS.dataset, FLAGS.exp, FLAGS.labelnum)
    # snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    # test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    num_classes = FLAGS.num_classes
    test_save_path = os.path.join(snapshot_path, 'test/')
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(iter_num))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    # avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
    #                            patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z,
                               save_result=True, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':

    metric = Inference(FLAGS)
    print(metric)
