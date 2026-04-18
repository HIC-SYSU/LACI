import argparse
import os
import shutil
from glob import glob
import torch

from utils.test_3d_patch import test_all_case
from networks.vnet_ACMT import VNet


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LA", help='dataset')
parser.add_argument('--exp', type=str,  default="ACMT_flod0", help='model_name')
parser.add_argument('--testlist', type=str,  default="test0.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='2', help='num_classes')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')
parser.add_argument('--patch_size', type=list, default=[112,112,80],help='patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='stride_xy')
parser.add_argument('--stride_z', type=int, default=4, help='stride_z')

def Inference(FLAGS):
    snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(FLAGS.dataset, FLAGS.exp, FLAGS.labelnum)
    test_save_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}/predict/".format(FLAGS.dataset, FLAGS.exp, FLAGS.labelnum)
    num_classes = FLAGS.num_classes
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = VNet(n_channels=1, n_classes=num_classes,normalization='batchnorm', has_dropout=True).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_8000.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    if FLAGS.dataset == "LA":
        with open('/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/' + FLAGS.testlist,
                  'r') as f:  # todo change test flod
            image_list = f.readlines()
        image_list = [item.replace('\n', '') + "/mri_norm_new.h5" for item in image_list]
    elif FLAGS.dataset == "LV":
        print(f'undefined')


    net.eval()
    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    metric = Inference(FLAGS)
    # print('dice, jc, hd, asd:', metric)
