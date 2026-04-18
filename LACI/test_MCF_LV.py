import os
import argparse
import torch

# from utils.test_util_MCF import test_all_case
from utils.test_3d_patch import test_all_case, test_all_MCF, test_all_case_MCF_s, test_all_MCF_PR
from networks.vnet_MCF import VNet
from networks.ResNet34_MCF import Resnet34
# from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LV", help='dataset')
parser.add_argument('--exp', type=str,  default="MCF", help='model_name')
parser.add_argument('--testlist', type=str,  default="test.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='3', help='num_classes')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--patch_size', type=list, default=[112,112,80],help='patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='stride_xy')
parser.add_argument('--stride_z', type=int, default=4, help='stride_z')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(FLAGS.dataset, FLAGS.exp, FLAGS.labelnum)
test_save_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/PR_LV/{}/{}/{}/text/".format(FLAGS.dataset, FLAGS.exp, FLAGS.labelnum)
num_classes = FLAGS.num_classes
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

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
def create_model(name='vnet'):
    # Network definition
    if name == 'vnet':
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
    if name == 'resnet34':
        net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()

    return model

def test_calculate_metric(epoch_num):
    vnet   = create_model(name='vnet')
    resnet = create_model(name='resnet34')

    v_save_mode_path = os.path.join(snapshot_path, 'vnet_iter_' + str(epoch_num) + '.pth')
    vnet.load_state_dict(torch.load(v_save_mode_path))
    print("init weight from {}".format(v_save_mode_path))
    vnet.eval()

    r_save_mode_path = os.path.join(snapshot_path, 'resnet_iter_' + str(epoch_num) + '.pth')
    resnet.load_state_dict(torch.load(r_save_mode_path))
    print("init weight from {}".format(r_save_mode_path))
    resnet.eval()

    # avg_metric = test_all_MCF(vnet, resnet, image_list, num_classes=num_classes,
    #                            patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z,
    #                            save_result=True, test_save_path=test_save_path)

    avg_metric = test_all_MCF_PR(vnet, resnet, image_list, num_classes=num_classes,
                              patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z,
                              save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    iters =6000
    metric = test_calculate_metric(iters)
    print('iter:', iter)
    print(metric)
