from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import cv2
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

# from AlphaPose.YOLOX.detector import PersonDetector
# from lib.dataset.AlphaPose.PoseEstimateLoader import SPPE_FastPose
from AlphaPose.fn import draw_single
_dir = os.path.split(os.path.realpath(__file__))[0]

class KeypointsDataset(data.Dataset):
    def __init__(self, cfg, img_path, bbox_path, is_train, joint_detector, transform=None):
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.is_train = is_train

        # set parameters for key points
        self.num_joints = 17

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)
        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT

        # for data processing
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY

        # generate heatmap label
        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.hoe_sigma = cfg.DATASET.HOE_SIGMA

        self.transform = transform

        self.bbox_path = bbox_path
        with open(self.bbox_path, "r") as f:
            self.annotations = f.readlines()

        self.image_path = img_path
        self.img_list = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                    os.listdir(self.image_path)),
            key=lambda x: int(x.split('.')[0]))
        
        ### joint detector model ###
        self.joints_detector = joint_detector

    def _box2cs(self, box):
        x, y, w, h = box[:4]
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
        return len(self.img_list)

    def _load_image(self, index):
        imgfile = self.img_list[int(self.annotations[index].split(" ")[0])]
        print(imgfile)
        bbox = list(map(int,self.annotations[index].split(" ")[1:5]))

        center, scale = self._box2cs(bbox)

        # label of orienation degree
        center, scale = self._box2cs(bbox)
        return imgfile, center, scale, bbox

    def __getitem__(self, index):
        imgfile, c, s, bbox = self._load_image(index)
        data_numpy =cv2.imread(self.image_path+imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        # if self.is_train:
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(imgfile))
            raise ValueError('Fail to read {}'.format(imgfile))
        
        # need add Alphapose
        # joints = 17 x 2, confidence = 17x1
        # joints, confidence =  Alphapose(input)
        tl_x, tl_y, w, h = bbox
        bbox_joint = torch.Tensor([tl_x, tl_y, tl_x+w, tl_y+h]).unsqueeze(0)
        # print(bbox)
        big_image_height, big_image_width = data_numpy.shape[:2]
        poses = self.joints_detector.predict(data_numpy, bbox_joint, torch.Tensor(1))
        # print(poses)
        # input = input.float()
        track_kpts = torch.zeros((13,3))
        # [Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle]
        for ps in poses:
            # in the image coordinate
            # print(ps)
            kpts = ps['keypoints']
            scores = ps['kp_score']
            ### draw keypoints ###
            pts = torch.cat((kpts, scores), axis=1).tolist()
            pts = np.array(pts)
            pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
            data_numpy = draw_single(data_numpy, pts)

            kpts[:,0] = torch.clamp(kpts[:,0], min=tl_x, max=tl_x+w)
            kpts[:,1] = torch.clamp(kpts[:,1], min=tl_y, max=tl_y+h)
            # in the resized image patch coordinate
            # print(self.image_size)
            kpts[:,0] = (kpts[:,0]-tl_x)/w*self.image_size[0]
            kpts[:,1] = (kpts[:,1]-tl_y)/h*self.image_size[1]
            # print(kpts)
            # track_kpts.append(torch.cat((kpts, ps['kp_score']), axis=1).tolist())
            track_kpts[:, :2] = kpts
            track_kpts[:, 2] = scores.squeeze()
            # print(track_kpts)
        

        coco_kpts = torch.zeros((17, 2))
        # print(track_kpts[0, :2].shape)
        # print(track_kpts[0, 2].shape)
        coco_kpts[0, :] = track_kpts[0, :2] * track_kpts[0, 2]  # Nose
        for i in range(5, 17):
            coco_kpts[i, :] = track_kpts[i-4, :2] * track_kpts[i-4, 2]
        # norm and rescale to [256,]
        coco_kpts = coco_kpts.flatten()
        # print(coco_kpts)

        meta = {
            'image_path': imgfile,
            'center': c,
            'scale': s,
        }
        # weighted_joints = joints * target_weight # 17 x 2
        # weighted_joints = weighted_joints.flatten() # 34 x 1
        # print("before flatten:{}".format(weighted_joints.shape))

        # weighted_joints[[2,3,4,5,6,7,8,9]] = 0     # get rid of eye and ear keypoints

        ### For visualization ###
        trans = get_affine_transform(c, s, 0, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        if self.transform:
            input = self.transform(input)

        return coco_kpts, input, meta


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
    train_dataset = COCO_HOE_Dataset(
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