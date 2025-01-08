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

# import _init_paths
from config import cfg
from config import update_config

from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary

import lib.dataset as dataset
from lib.dataset import My_Dataset
import lib.models as models


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
                        default='')
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
    parser.add_argument('--write_ori', action='store_true')
    parser.add_argument('--write_keypoints', action='store_true')
    args = parser.parse_args()

    return args

# this is to  draw images
def draw_orientation(img_np, keypoints, pred_ori ,index, path, alis=''):
    if not os.path.exists(path):
        os.makedirs(path)
    for idx in range(len(pred_ori)):
        img_tmp = img_np[idx]

        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= [0.229, 0.224, 0.225]
        img_tmp += [0.485, 0.456, 0.406]
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        for joint_index, position in keypoints.items():
            # left red , right green
            # import pdb;pdb.set_trace()
            img_tmp = img_tmp.copy()
            # print("joint_index{} x{} y{}".format(joint_index, position[0]*4, position[1]*4))
            if int(joint_index) %2 ==0:
                #right
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            else:
                #left
                cv2.circle(img_tmp, (int(position[1]*4), int(position[0]*4)), radius=2, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)

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

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False)
    # import pdb;pdb.set_trace()e

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

    valid_dataset = My_Dataset(
        cfg, args.img_root,args.gt_path, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    i = 0
    with torch.no_grad():
        import time
        for i, input in enumerate(valid_loader):
            start = time.time()
            plane_output, hoe_output = model(input)

            keypoints = dict()
            for j in range(17):
                # 关键点输出值大于0.4 认为其存在
                if plane_output[0][j].cpu().numpy().max() > 0.4:
                    position = np.unravel_index(plane_output[0][j].cpu().numpy().argmax(), plane_output[0][j].cpu().numpy().shape)
                    keypoint = {"{}".format(j):position}
                    keypoints.update(keypoint)

            tmp = hoe_output.detach().cpu().numpy()
            pre_ori = tmp.argmax(axis = 1)*5
            end = time.time()
            # print("estimate_time{}".format(end-start))
            draw_orientation(input.numpy(), keypoints, pre_ori, i, args.vis_dir,"esitimated")