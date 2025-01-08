from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import cv2
from matplotlib.transforms import Bbox
import torch
import random
import numpy as np
from scipy.io import loadmat, savemat
from collections import OrderedDict
import torch.utils.data as data

from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform
from lib.utils.transforms import fliplr_joints
from lib.utils.transforms import hoe_heatmap_gen

logger = logging.getLogger(__name__)

class My_Dataset(data.Dataset):
    def __init__(self, cfg, image_path, gt_path, is_train, transform=None):
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.is_train = is_train

        # set parameters for key points
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_path = image_path
        self.gt_path = gt_path

        with open(self.gt_path, "r") as f:
            self.annotations = f.readlines()

        self.img_list = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                    os.listdir(self.image_path)),
            key=lambda x: int(x.split('.')[0]))

        # generate heatmap label
        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.hoe_sigma = cfg.DATASET.HOE_SIGMA

        self.transform = transform

    def _box2cs(self, box):
        x, y, w, h = box
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale

    def __len__(self,):
        return len(self.annotations)

    def _load_image(self, index):

        imgfile = self.img_list[int(self.annotations[index].split(" ")[0])]
        print(imgfile)
        bbox = list(map(int,self.annotations[index].split(" ")[1:5]))
        # bbox[1] += 200
        # bbox[3] -= 200

        center, scale = self._box2cs(bbox)
        return imgfile, center, scale

    def __getitem__(self, index):
        imgfile, c, s = self._load_image(index)
        data_numpy = cv2.imread(self.image_path+imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(imgfile))
            raise ValueError('Fail to read {}'.format(imgfile))

        # Not use score
        # score = 0
        trans = get_affine_transform(c, s, 0, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        if self.transform:
            input = self.transform(input)
        # input = input.float()

        return input


if __name__ == '__main__':
    import argparse
    from config import cfg
    from config import update_config
    import torchvision.transforms as transforms
    import torch

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cfg = "experiments/w32_256x192_adam_lr1e-3.yaml"
    args.opts, args.modelDir, args.logDir, args.dataDir = "", "", "", ""
    update_config(cfg, args)
    normalize = transforms.Normalize(
        mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]
    )
    train_dataset = My_Dataset(
        cfg, cfg.DATASET.TRAIN_ROOT, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    for i, b in enumerate(train_loader):
        if i == 5:
            break
        else:
            print(b[0].shape, b[1].shape, b[2].shape)
            print('fdsa')