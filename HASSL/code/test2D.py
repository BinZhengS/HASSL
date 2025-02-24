import argparse
import os
import shutil
from glob import glob

import torch
import numpy as np

from val_2D import test_single_skin

from dataloaders.brats2019 import (Skin_dataset,BraTS2019,LAHeart,Pancreas,CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from torch.utils.data import DataLoader

from albumentations.core.composition import Compose
from albumentations.augmentations import transforms

from networks.net_factory import net_factory
from networks import unet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ISIC2017', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ISIC2017_CPS_plr_100_0.3', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--epoch_num', type=int,
                    default=6000, help='epoch_num')
parser.add_argument('--save_mode_path', type=int,
                    default=None, help='save_mode_path')


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 1
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet.UNet(in_chns=3,class_num=num_classes).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))

    net.load_state_dict(torch.load(save_mode_path))

    print("init weight from {}".format(save_mode_path))

    net.eval()

    db_test = Skin_dataset(base_dir=FLAGS.root_path, split="test", transform=Compose([
        transforms.Normalize(),
    ]))

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)

    metric_list = 0.0
    dsc_list=[]
    with open(test_save_path+'/test.list','a') as f:
        for i_batch, sampled_batch in enumerate(testloader):
            metric_i = test_single_skin(
                sampled_batch, test_save_path,net, classes=num_classes)

            metric_list += np.array(metric_i)
            f.writelines('id:{} Dice:{} jc:{} pre:{} g_mean:{} HD:{}\n'.format(sampled_batch['id'][0],metric_i[0][0],metric_i[0][1],metric_i[0][2],metric_i[0][3],metric_i[0][4]))
            dsc_list.append(metric_i[0][1])
        avg_metric = metric_list / len(db_test)
        f.writelines('Mean Metric:{}'.format(avg_metric))
    f.close()
    np.save(os.path.join(test_save_path, 'test_dsc_list.npy'), np.array(dsc_list))
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)







