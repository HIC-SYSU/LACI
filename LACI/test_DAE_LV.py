import os
import argparse
import torch
from networks.vnet_MCF import VNet
from utils.test_3d_patch import test_all_case, test_all_case_LV_PR

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str, default="LV", help='dataset')
parser.add_argument('--exp', type=str, default='AU_MT', help='model_name')
parser.add_argument('--testlist', type=str,  default="test.txt", help='model_name')
parser.add_argument('--nb_labels', type=int, default=8, help='trained samples')
parser.add_argument('--num_classes', type=int, default='3', help='num_classes')
parser.add_argument('--total_labels', type=int, default=369, help='maximum samples to train')
parser.add_argument('--patch_size', type=list, default=[112,112,80],help='patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='stride_xy')
parser.add_argument('--stride_z', type=int, default=4, help='stride_z')

parser.add_argument('--model', type=str,  default='Vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
parser.add_argument('--split', type=str,  default='test', help='train/val/test split')
parser.add_argument('--save', action='store_true',  default=False, help='save results')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
model_path = '../DAE_MT_models/'
snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(FLAGS.dataset, FLAGS.exp, FLAGS.nb_labels)
# test_save_path = os.path.join(snapshot_path, "predict_score/")
test_save_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/PR_LV/{}/{}/{}/text/".format(FLAGS.dataset, FLAGS.exp, FLAGS.nb_labels)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = FLAGS.num_classes

# root_path = FLAGS.root_path
# split = FLAGS.split

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
def test_calculate_metric():
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_6000.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # Normalization : mean_percentile, full_volume_mean
    # avg_metric = test_all_case(net, root_path, split, normalization='mean_percentile', num_classes=num_classes,
    #                            patch_size=patch_size, stride_xy=18, stride_z=4,
    #                            save_result=FLAGS.save, test_save_path=test_save_path)
    # avg_metric = test_all_case(net, image_list, num_classes=num_classes,
    #                            patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z,
    #                            save_result=True, test_save_path=test_save_path, exp=FLAGS.exp)
    avg_metric = test_all_case_LV_PR(net, image_list, num_classes=num_classes,
                                     patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z,
                                     save_result=True, test_save_path=test_save_path,
                                     exp=FLAGS.exp)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print("metric: ", metric)
