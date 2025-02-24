import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from tqdm import tqdm

from albumentations.core.composition import Compose
from albumentations.augmentations import transforms
from albumentations import RandomRotate90, Resize

from dataloaders import utils
from dataloaders.brats2019 import (Skin_dataset,BraTS2019,LAHeart,Pancreas,CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_skin

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ISIC2017', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ISIC2017_HDSSL', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-2,
                    help='segmentation network learning rate')
parser.add_argument('--min_lr', type=float, default=1e-5,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_percent', type=float, default=0.3,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.75, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 1

    net1 = net_factory(net_type=args.model, in_chns=3, class_num=num_classes).cuda()
    net2 = net_factory(net_type=args.model, in_chns=3, class_num=num_classes).cuda()
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
    model1.train()
    model2.train()

    db_train = Skin_dataset(base_dir=args.root_path, split="train", num=None, transform=Compose([
        RandomRotate90(),
        transforms.Flip(),
        transforms.Normalize(),
    ]))
    db_val = Skin_dataset(base_dir=args.root_path, split="val", transform=Compose([
        transforms.Normalize(),
    ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    total_slices = len(db_train)

    labeled_idxs = list(range(0, round(total_slices * args.labeled_percent)))
    unlabeled_idxs = list(range(round(total_slices * args.labeled_percent), total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    best_performance1 = 0.0
    best_performance2 = 0.0
    iter_num = 0

    Dice_loss = losses.DiceLoss_()
    H_Dice_loss = losses.harmonious_DiceLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.sigmoid(outputs1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.sigmoid(outputs2)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss1 =Dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs]) + 0.5*F.binary_cross_entropy_with_logits(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].float())
            loss2 =Dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs]) + 0.5*F.binary_cross_entropy_with_logits(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].float())

            SP_loss = loss1 + loss2

            if iter_num<=200:
                stable_map1 = (outputs_soft1[args.labeled_bs:].detach() > 0.8).float() + (outputs_soft1[args.labeled_bs:].detach() < 0.2).float()
                stable_map2 = (outputs_soft2[args.labeled_bs:].detach() > 0.8).float() + (outputs_soft2[args.labeled_bs:].detach() < 0.2).float()
                stable_map = (stable_map1 == stable_map2)

                USP_loss = 2 * (nn.MSELoss(reduction='none')(outputs_soft1[args.labeled_bs:],
                                                             outputs_soft2[args.labeled_bs:]) * stable_map).sum() / stable_map.sum()

            else:
                pseudo_outputs1 = (outputs_soft1[args.labeled_bs:].detach() > 0.5).float()
                pseudo_outputs2 = (outputs_soft2[args.labeled_bs:].detach() > 0.5).float()

                pseudo_supervision1 = F.binary_cross_entropy_with_logits(outputs1[args.labeled_bs:], pseudo_outputs2)
                pseudo_supervision2 = F.binary_cross_entropy_with_logits(outputs2[args.labeled_bs:], pseudo_outputs1)

                HD_loss = H_Dice_loss(outputs_soft1[args.labeled_bs:],outputs_soft2[args.labeled_bs:], pseudo_outputs2, pseudo_outputs1)
                USP_loss = 0.5 * (pseudo_supervision1 + pseudo_supervision2) + 2 * HD_loss

            SP = SP_loss.item()
            USP = USP_loss.item()

            Mul = (SP * np.sqrt(1-consistency_weight) + 1e-5) / ((np.sqrt(consistency_weight) * np.sqrt((1-consistency_weight)*np.square(SP)+np.square(USP)))+1e-5)
            Fi = (USP * np.sqrt(consistency_weight) + 1e-5) / ((np.sqrt(1-consistency_weight) * np.sqrt((consistency_weight)*np.square(USP)+np.square(SP)))+1e-5)
            k = Mul / Fi
            Mul = k / (k+1)
            Fi = 1. / (k+1)

            loss = Mul * SP_loss + Fi * USP_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            # scheduler1.step()
            # scheduler2.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_
            iter_num = iter_num + 1

            writer.add_scalar('lr', lr_, iter_num)

            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/SP_loss',
                              SP_loss.item(), iter_num)
            writer.add_scalar('loss/USP_loss',
                              USP_loss.item(), iter_num)

            logging.info(
                'iteration %d : SP loss : %f USP loss : %f  con : %f Mul : %f Fi : %f' % (iter_num, SP_loss.item(),USP_loss.item(),consistency_weight,Mul,Fi))
         
            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_skin(
                        sample_batch=sampled_batch, test_save_path=None, net=model1, classes=num_classes)
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[-1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_skin(
                        sample_batch=sampled_batch, test_save_path=None, net=model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[-1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            # change lr
            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer1.param_groups:
            #         param_group['lr'] = lr_
            #     for param_group in optimizer2.param_groups:
            #         param_group['lr'] = lr_

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_percent, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)


