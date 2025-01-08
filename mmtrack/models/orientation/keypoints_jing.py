# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from genericpath import exists
import json
import os
import cv2
import pprint
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from config import cfg
from config import update_config

from lib.core.inference import get_final_preds

from lib.core.function import train
from lib.core.function import validate

from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary

import lib.dataset as dataset
from lib.dataset import KeypointsDataset
import lib.models as models

from AlphaPose.YOLOX.detector import PersonDetector
from AlphaPose.PoseEstimateLoader import SPPE_FastPose
_dir = os.path.split(os.path.realpath(__file__))[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--img_root',
                        help='image root directory',
                        type=str,
                        default='')
    parser.add_argument('--gt_path',
                        help='person bbox json file',
                        type=str,
                        default='')
    parser.add_argument('--vis_dir',
                        help='output visualization results',
                        type=str,
                        default='visualization')
    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args

# this is to  draw images
def draw_orientation(img_np,  pred_ori ,index, path, alis=''):
    if not os.path.exists(path):
        os.makedirs(path)
    for idx in range(len(pred_ori)):
        img_tmp = img_np[idx]

        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= [0.229, 0.224, 0.225]
        img_tmp += [0.485, 0.456, 0.406]
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)

        # then draw the image
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(1, 1, 1)

        theta_2 = pred_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_2)], [0, np.sin(theta_2)], color="blue", linewidth=3)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        fig.savefig(os.path.join(path, str(index+idx)+'_'+alis+'.jpg'))
        ori_img = cv2.imread(os.path.join(path, str(index+ idx)+'_'+alis+'.jpg'))

        width = img_tmp.shape[1]
        ori_img = cv2.resize(ori_img, (width, width), interpolation=cv2.INTER_CUBIC)
        img_all = np.concatenate([img_tmp, ori_img],axis=0)
        im = Image.fromarray(img_all)
        im.save(os.path.join(path, str(index+ idx)+'_'+alis+'_raw.jpg'))
        plt.close("all")
        del ori_img,img_all, im,img_tmp
        gc.collect()

def floatize(input):
    output = []
    for i in input:
        output.append(float(i))
    return output

def xywh_to_xyxy_torch(xywhs, img_width, img_height):
        """
        input:
            xywhs(N,4)
        """
        xyxys = []
        for xywh in xywhs:
            x1 = xywh[0]
            x2 = min(int(xywh[0]+xywh[2]),img_width-1)
            y1 = xywh[1]
            y2 = min(int(xywh[1]+xywh[3]),img_height-1)
            xyxys.append([x1, y1, x2, y2])

        return torch.Tensor(xyxys)

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False)
    # import pdb;pdb.set_trace()

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = KeypointsDataset(
        cfg, args.img_root, args.gt_path, False,        
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    box_detector = PersonDetector(model='yolox-s', ckpt=os.path.join(_dir, 'AlphaPose/YOLOX/weights/yolox_s.pth.tar'))
    joints_detector = SPPE_FastPose(backbone="resnet50", input_height=224, input_width=160, device="cuda:0")


    i = 0
    with torch.no_grad():
        import time
        for i, (keypoints, img, meta) in enumerate(valid_loader):
            # img: (1,3,H,W)
            xywhs, scores = box_detector.detect(img, conf=0.7)
            detected = xywh_to_xyxy_torch(xywhs, img.shape[-1], img.shape[-2])
            poses = joints_detector.predict(img.squeeze(), detected, torch.Tensor(scores))
            track_kpts = []
            for ps in poses:
                track_kpts.append(torch.cat((ps['keypoints'], ps['kp_score']), axis=1).tolist())
            # [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle]
            coco_kpts = torch.zeros((17, 2))
            coco_kpts[0, :] = torch.Tensor(track_kpts[0, :2] * track_kpts[0, 2])  # Nose
            for i in range(5, 17):
                coco_kpts[i, :] = torch.Tensor(track_kpts[i-4, :2] * track_kpts[i-4, 2])
            coco_kpts = coco_kpts.flatten()

            # keypoints = batch_size x 17 x 2 =  batch_size x 34
            start = time.time()
            # hoe_output = model(keypoints)
            hoe_output = model(coco_kpts)

            tmp = hoe_output.detach().cpu().numpy()
            pre_ori = tmp.argmax(axis = 1)*5
            end = time.time()
            # print("estimate_time{}".format(end-start))
            img = img.numpy()
            draw_orientation(img, pre_ori, i, args.vis_dir,"esitimated")
    