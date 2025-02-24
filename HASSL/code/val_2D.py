import os

import numpy as np
import torch
import cv2
from medpy import metric
from scipy.ndimage import zoom
from hausdorff import hausdorff_distance

def calculate_metric_skin(pred, gt,test_save_path):

    assert pred.shape == gt.shape
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    if test_save_path is not None:
        cv2.imwrite(test_save_path,(pred*255).astype('uint8'))

    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        se = metric.binary.recall(pred, gt)
        sp = metric.binary.specificity(pred, gt)
        pre = metric.binary.precision(pred, gt)
        g_mean = np.sqrt(se * sp)
        hd95 = hausdorff_distance(pred, gt)
        return dice, jc, pre, g_mean, hd95
    else:
        return 0, 0, 0, 0, 0



def test_single_skin(sample_batch,test_save_path, net, classes):

    image = sample_batch['image'].cuda()
    label = sample_batch['label'].cuda()
    net.eval()

    with torch.no_grad():

        pred = torch.sigmoid(net(image)[0])
        pred = torch.where(pred>0.5,1,0)
        pred = pred.squeeze()


    label = label.squeeze()

    metric_list = []

    if test_save_path is not None:
        test_save_path = os.path.join(test_save_path,sample_batch['id'][0]+'.png')


    metric_list.append(calculate_metric_skin(
        pred , label,test_save_path))

    return metric_list
