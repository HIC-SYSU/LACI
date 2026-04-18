import os
import argparse
import torch
import pdb

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case, test_all_case_LV_PR

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LV", help='dataset')
parser.add_argument('--exp', type=str,  default="BCP", help='model_name')
parser.add_argument('--testlist', type=str,  default="test.txt", help='model_name')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='3', help='num_classes')
parser.add_argument('--model', type=str,  default='VNet_BCP', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing?')
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--patch_size', type=list, default=[112,112,80],help='patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='stride_xy')
parser.add_argument('--stride_z', type=int, default=4, help='stride_z')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}/self_train".format(FLAGS.dataset, FLAGS.exp, FLAGS.labelnum)
# snapshot_path = '/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/LV/BCP/32/self_train'
test_save_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/PR_LV/{}/{}/{}/text/".format(FLAGS.dataset, FLAGS.exp, FLAGS.labelnum)
num_classes = FLAGS.num_classes

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
if FLAGS.dataset == "LA":
    with open('/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/'+FLAGS.testlist, 'r') as f:  # todo change test flod
        image_list = f.readlines()
    image_list = [item.replace('\n', '') + "/mri_norm_new.h5" for item in image_list]
elif FLAGS.dataset == "LV":
    with open('/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/test.txt', 'r') as f:  # todo change test flod
        image_list = f.readlines()
        print(image_list)
    image_list = [item.replace('\n', '') for item in image_list]
def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    # save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    save_model_path = os.path.join(snapshot_path, 'iter_6000.pth')     ## 15000
    model.load_state_dict(torch.load(save_model_path))      ## ['net']
    print("init weight from {}".format(save_model_path))

    model.eval()

#### test_all_case
    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms, exp=FLAGS.exp)   ## test_all_case_LV_P

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

# python test_LA.py --model 0214_re01 --gpu 0
