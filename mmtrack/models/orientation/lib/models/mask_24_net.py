from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from lib.models.part_attention import PartAttention

import numpy as np
import cv2
import torchvision.transforms as transforms
from lib.utils.transforms import get_affine_transform
from lib.core.inference import get_final_preds
normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
        ])

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3,padding=1, stride=stride)



class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1]), requires_grad=True
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):

        x = x.unsqueeze(3)
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        # logger.info('LocallyConnected2d {} {}'.format(x.unsqueeze(1).shape, self.weight.shape))
        # torch.Size([1, 1, 4, 128, 1, 1]) torch.Size([1, 4, 4, 4, 1, 1])
        # 
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        out = out.squeeze(3)
        return out


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, mask_channel=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = LocallyConnected2d(inplanes, planes, output_size=[int(mask_channel),1], kernel_size = 1, stride=1)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = LocallyConnected2d(planes, planes, output_size=[int(mask_channel),1], kernel_size = 1, stride=1)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class score_embedding(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels):
        super(score_embedding, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reg = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.reg(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PartAttentionHRNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(PartAttentionHRNet, self).__init__()
        self.num_joints = cfg.MODEL.NUM_JOINTS
        # self.num_input_features = (32, 96)
        self.mask_channel = 25
        self.pose_inchannel = 23

        self.use_featuremap = cfg.MODEL.USE_FEATUREMAP

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # import pdb;pdb.set_trace()
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            # in_channels=96,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        if self.use_featuremap:
            self.inplanes = 128
        else:
            self.inplanes = self.num_joints
        self.conv_local_feature_layer1 = self._make_conv_layer(1, (96,), (3,), 96)
        self.mask_final_layer = nn.Conv2d(
                in_channels=96,
                out_channels=self.mask_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        # self.mask_weight_layer = nn.Conv2d(
        #         in_channels=96,
        #         out_channels=5,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     )
        self.part_attention = PartAttention(
            act='softmax')
        self.conv_local_feature_layer2 = self._make_conv_layer(1, (128, ), (3,), 96)
        # self.inplanes = 17
        # self.hoe_layer1 = self._make_layer(BasicBlock, 64, 2)
        self.hoe_layer2 = self._make_layer1d(BasicBlock1D, 1, 2, stride=1)
        # self.hoe_layer3 = self._make_layer1d(BasicBlock1D, 256, 2, stride=2)
        # self.hoe_layer4 = self._make_layer1d(BasicBlock1D, 512, 2, stride=2)

        self.mask_weight_layer1 = nn.Conv2d(
                in_channels=self.mask_channel,
                out_channels=12,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        self.mask_bn1 = nn.BatchNorm2d(12, momentum=BN_MOMENTUM)

        self.mask_weight_layer2 = nn.Conv2d(
                in_channels=12,
                out_channels=24,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        self.mask_bn2 = nn.BatchNorm2d(24, momentum=BN_MOMENTUM)

        self.mask_weight_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mask_select_fc = nn.Linear(24, 2)
        self.mask_activate_layer = nn.Sigmoid()

        # self.pose_subnet = Pose_Subnet(blocks=[OSBlock, OSBlock], in_channels=self.pose_inchannel,
        #                         channels=[32, 32, 32], att_num=self.num_joints, matching_score_reg=True)
        self.hoe_avgpool = nn.AdaptiveAvgPool1d(1)
        # self.hoe_fc1 = nn.Linear(128*(self.mask_channel-1), 256)
        # self.hoe_fc2 = nn.Linear(256, 72)
        self.hoe_fc3 = nn.Linear(128, 72)

    def _get_part_attention_map(self, features):
        heatmaps = self.mask_final_layer(features)
        return heatmaps
    
    def _get_mask_weight(self, mask):
        mask = self.mask_weight_layer1(mask)
        mask = self.mask_bn1(mask)
        mask = self.relu(mask)
        mask = self.mask_weight_layer2(mask)
        mask = self.mask_bn2(mask)
        mask = self.relu(mask)
        mask_weight = self.mask_weight_avgpool(mask)
        mask_weight = mask_weight.view(mask_weight.size(0), -1)
        mask_weight = self.mask_select_fc(mask_weight)
        # mask_weight = self.mask_activate_layer(mask_weight)
        mask_weight = F.softmax(mask_weight, dim=1)
        # heatmaps = heatmaps[:,1:,:,:] # remove background

        return mask_weight

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    def _make_conv_layer(self, num_layers, num_filters, num_kernels, input_feature_num):
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=input_feature_num,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            input_feature_num = planes

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            # if self.use_self_attention:
            #     layers.append(SelfAttention(planes))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer1d(self, block, planes, blocks, stride=1):
        downsample = None
        # input mask_channel
        # planes output_channel
        self.inplanes = 24
        channel_num = int(128)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                LocallyConnected2d(
                    self.inplanes, planes * block.expansion,
                    output_size=[channel_num, 1], kernel_size = 1, stride=1
                ),
                nn.BatchNorm1d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        logger.info('_make_layer1d {} {} {}'.format(self.inplanes, planes, block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, mask_channel=channel_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, mask_channel=channel_num))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        feature_map_2 = y_list[0]

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        feature_map_3 = y_list[0]

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        feature_map_4 = y_list[0]
        x = self.final_layer(y_list[0])

        # score = self.pose_subnet(x)
        # if self.use_featuremap:
        x_cat = torch.cat([feature_map_2, feature_map_3, feature_map_4], 1)
        # else:
        #     x_cat = x
        # y = self.hoe_layer1(x)
        # x_cat (1, 96,64,48)
        # y = self.hoe_layer2(x_cat)
        # y = self.hoe_layer3(y)
        # y = self.hoe_layer4(y)

        mask_feature_2d = self.conv_local_feature_layer1(x_cat)
        mask_feature_2d = F.dropout(mask_feature_2d, p=0.5, training=self.training)

        heatmaps = self._get_part_attention_map(mask_feature_2d)

        attention_mask_weight = self._get_mask_weight(heatmaps[:,:,:,:])
        feature_2d = self.conv_local_feature_layer2(x_cat)
        
        # print(attention_mask_weight)
        # new_attention_mask = attention_mask.detach()
        # import pdb;pdb.set_trace()
        attention_feature = self.part_attention(feature_2d, heatmaps[:,:self.mask_channel-1,:,:])
        # weighted_attention_feature = attention_feature * attention_mask_weight.unsqueeze(1)
        embedding = self.hoe_layer2(attention_feature.permute(0,2,1))
        #.permute(0,2,1)
        y = embedding.reshape(embedding.size(0), -1)
        # y = self.hoe_fc1(y)
        # y = self.hoe_fc2(y)
        y = self.hoe_fc3(y)
        y = F.softmax(y, dim=1)

        return x, y, heatmaps, attention_mask_weight

    def get_hoe_params(self):
        for m in self.named_modules():
            if "hoe" in m[0] or "final_layer" in m[0]:
                print(m[0])
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p

    def predict(self, image, img_metas, bboxs, bboxs_scores, bbox_ids=None, rescale=False):
        """
        rescale: True for the bbox of original size
        """
        import time
        # t_start= time.time()
        h, w, _ = img_metas[0]['img_shape']
        h_ori, w_ori, _ = img_metas[0]['ori_shape']
        if rescale:
            bboxs[:,:4] *= torch.tensor(img_metas[0]['scale_factor']).to(bboxs.device)
        bboxs[:,0::2] = torch.clamp(bboxs[:,0::2], min=0, max=w)
        bboxs[:,1::2] = torch.clamp(bboxs[:,1::2], min=0, max=h)

        inps, boxes_np, c_ses, invers_transes = self._crop_dets(image, bboxs, self.img_height, self.img_width)
        
        # print("pre_precess", (time.time()-t_start)*1000)
        # t_start= time.time()
        ### All these information are in the local coordinate of (64, 48) ###
        pre_key_heatmaps, orientations, pre_parsings, weights = self.forward(inps.cuda())
        
        # print("predict", (time.time()-t_start)*1000)
        # t_start= time.time()

        # transoform kpts to the image space
        keypoints = torch.zeros((pre_key_heatmaps.shape[0], pre_key_heatmaps.shape[1], 3))
        for i in range(pre_key_heatmaps.shape[0]):
            original_keypoints, maxvals = get_final_preds(self.cfg, pre_key_heatmaps[i].unsqueeze(0).cpu().numpy(), np.expand_dims(c_ses[i, :2],axis=0), np.expand_dims(c_ses[i, 2:4],axis=0))
            keypoints[i, :, :2], keypoints[i, :, 2] = torch.Tensor(original_keypoints.squeeze()), torch.Tensor(maxvals.squeeze())
        # print("kpts_process", (time.time()-t_start)*1000)
        # t_start= time.time()
        # transform parsing mask to the bbox space
        parsings = np.zeros((pre_parsings.shape[0], 4, pre_parsings.shape[2], pre_parsings.shape[3]))  # B,25,64,48
        pre_parsings = pre_parsings.cpu().numpy()
        pre_parsings[pre_parsings>0.1] = 1
        pre_parsings[pre_parsings!=1] = 0
        parsings[:, 0, :, :] = pre_parsings[:, 12, :, :]  # head
        parsings[:, 1, :, :] = np.sum(pre_parsings[:, [1,3,7,8,9,10,13,14,16,17,18,19,21,23], :, :], axis=1)  # torso
        parsings[:, 1, :, :][parsings[:, 1, :, :]>0] = 1
        parsings[:, 2, :, :] = np.sum(pre_parsings[:, [2,4,15,22], :, :], axis=1)  # legs
        parsings[:, 2, :, :][parsings[:, 2, :, :]>0] = 1
        parsings[:, 3, :, :] = np.sum(pre_parsings[:, [6,11], :, :], axis=1)  # feet
        parsings[:, 3, :, :][parsings[:, 3, :, :]>0] = 1  # feet

        # head
        for i in range(pre_parsings.shape[0]):
            bigger_bbox_w, bigger_bbox_h = (c_ses[i, 2:4]*self.pixel_std)
            x1, y1, w, h = boxes_np[i]
            transform_parse = parsings[i].transpose(1,2,0)
            transform_parse = cv2.warpAffine(
                    transform_parse,
                    invers_transes[i],
                    (int(bigger_bbox_w), int(bigger_bbox_h)),
                    flags=cv2.INTER_LINEAR)[int(bigger_bbox_h/2-h/2):int(bigger_bbox_h/2+h/2),int(bigger_bbox_w/2-w/2):int(bigger_bbox_w/2+w/2), :]
            transform_parse = cv2.resize(transform_parse, (parsings.shape[3], parsings.shape[2]))  # to 64,48
            parsings[i] = transform_parse.transpose(2,0,1)
        # print("parsing_process", (time.time()-t_start)*1000)
        # t_start= time.time()
        

        # transform orientation to degree
        orientations = orientations.detach().cpu().numpy()
        orientations = orientations.argmax(axis = 1)*5
        weights = weights.detach().cpu().numpy()
        if rescale:
            bboxs[:,:4] /= torch.tensor(img_metas[0]['scale_factor']).to(bboxs.device)
            keypoints[:,:,:2] /= torch.tensor(img_metas[0]['scale_factor'][:2]).to(bboxs.device)
        bboxs[:,0::2] = torch.clamp(bboxs[:,0::2], min=0, max=w_ori)
        bboxs[:,1::2] = torch.clamp(bboxs[:,1::2], min=0, max=h_ori)
        keypoints[:,:,0] = torch.clamp(keypoints[:,:,0], min=0, max=w_ori)  # scale to 
        keypoints[:,:,1] = torch.clamp(keypoints[:,:,1], min=0, max=h_ori)
        result = []
        for i in range(bboxs.shape[0]):
            result.append({
                "img":inps[i].cpu().numpy(),
                "bbox_id":bbox_ids[i],
                "bbox": bboxs[i],
                "bbox_score": bboxs_scores[i],
                "keypoints": keypoints[i].cpu().numpy(), 
                "orientation": orientations[i],
                "parsing": parsings[i],
                "weight": weights[i]
            })

        # print("post_process", (time.time()-t_start)*1000)

        # result = pose_nms(bboxs, bboxs_scores, xy_img, scores, bbox_ids=bbox_ids)
        
        return result
def get_pose_net(cfg, is_train, **kwargs):
    model = PartAttentionHRNet(cfg, **kwargs)
    return model

def im_to_torch(img):
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    img = img.astype(np.uint8)
    return img

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

# just for debug
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    import argparse
    import experiments
    import config
    from config import cfg
    from config import update_config
    import torchvision.transforms as transforms
    import torch

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cfg = "experiments/w32_256x192_adam_lr1e-3.yaml"
    args.opts, args.modelDir, args.logDir, args.dataDir = "", "", "", ""
    update_config(cfg, args)
    model = PoseHighResolutionNet(cfg)
    model.eval()
    input = torch.rand(1, 3, 256, 192)
    output, hoe = model(input)
    print(output.size(), hoe.shape)