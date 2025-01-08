# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PartAttention(nn.Module):
    def __init__(self, act='softmax', use_scale=False):
        super(PartAttention, self).__init__()
        self.act = act

    def forward(self, features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape

        # 如何进行维度计算,精髓
        if self.act == 'softmax':
            # normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)  # new
            normalized_heatmap =heatmaps.reshape(batch_size, num_joints, -1)  # new_2
        elif self.act == 'sigmoid':
            normalized_heatmap = torch.sigmoid(heatmaps.reshape(batch_size, num_joints, -1))
        features = features.reshape(batch_size, -1, height*width)

        # matmul只计算后两维,所以是(17,W*H) @ (W*H, Channel),输出维度是(Batch size, 17, Channel)
        attended_features = torch.matmul(normalized_heatmap, features.transpose(2,1))
        # new_attended_features = normalized_heatmap * features.transpose(2,1)
        attended_features = attended_features.transpose(2,1)

        return attended_features