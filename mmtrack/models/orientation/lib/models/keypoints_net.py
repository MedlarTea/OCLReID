from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

class KeypointsNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(KeypointsNet, self).__init__()
        self.output_size=72
        self.input_size = 34
        self.linear_size=512
        self.p_dropout=0.2
        self.num_stage=3
        self.device='cuda'
        self.num_stage = self.num_stage
        self.stereo_size = 1024
        self.mono_size = int(self.input_size / 2)
        self.linear_stages = []
        # Initialize weights

        # Preprocessing
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(self.num_stage):
            self.linear_stages.append(MyLinearSimple(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # Post processing
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

        # ------------------------Other----------------------------------------------
        # Auxiliary
        self.w_aux = nn.Linear(self.linear_size, 1)

        # Final
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # Auxiliary task
        y = self.w2(y)
        # aux = self.w_aux(y)

        # Final layers
        y = self.w3(y)
        y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w_fin(y)

        # Cat with auxiliary task
        # y = torch.cat((y, aux), dim=1)
        return y

    # def init_weights(self, pretrained=''):
    #     logger.info('=> init weights from normal distribution')
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.normal_(m.weight, std=0.001)
    #             for name, _ in m.named_parameters():
    #                 if name in ['bias']:
    #                     nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


class MyLinearSimple(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

def get_pose_net(cfg, is_train, **kwargs):
    model = KeypointsNet(cfg, **kwargs)

    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #     model.init_weights(cfg.MODEL.PRETRAINED)
    return model