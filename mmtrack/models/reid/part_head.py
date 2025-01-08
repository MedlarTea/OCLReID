# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcls.models.builder import HEADS
from mmcls.models.heads.base_head import BaseHead
from mmcls.models.losses import Accuracy
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import build_loss

from .fc_module import FcModule
import torch
import torch.nn.functional as F


@HEADS.register_module()
class PartHead(BaseHead):
    """Linear head for re-identification.

    Args:
        num_fcs (int): Number of fcs.
        in_channels (int): Number of channels in the input.
        fc_channels (int): Number of channels in the fcs.
        out_channels (int): Number of channels in the output.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to None.
        num_classes (int, optional): Number of the identities. Default to None.
        loss (dict, optional): Cross entropy loss to train the
            re-identificaiton module.
        loss_pairwise (dict, optional): Triplet loss to train the
            re-identificaiton module.
        topk (int, optional): Calculate topk accuracy. Default to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to dict(type='Normal',layer='Linear', mean=0, std=0.01,
            bias=0).
    """

    def __init__(self,
                 num_fcs,
                 in_channels,
                 fc_channels,
                 out_channels,
                 norm_cfg=None,
                 act_cfg=None,
                 num_classes=None,
                 loss=None,
                 loss_pairwise=None,
                 topk=(1, ),
                 init_cfg=dict(
                     type='Normal', layer='Linear', mean=0, std=0.01, bias=0)):
        super(PartHead, self).__init__(init_cfg)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        if not loss:
            if isinstance(num_classes, int):
                warnings.warn('Since cross entropy is not set, '
                              'the num_classes will be ignored.')
            if not loss_pairwise:
                raise ValueError('Please choose at least one loss in '
                                 'triplet loss and cross entropy loss.')
        elif not isinstance(num_classes, int):
            raise TypeError('The num_classes must be a current number, '
                            'if there is cross entropy loss.')
        self.loss_cls = build_loss(loss) if loss else None
        self.loss_triplet = build_loss(
            loss_pairwise) if loss_pairwise else None

        self.num_fcs = num_fcs
        self.in_channels = in_channels  # 512 for resnet18
        self.fc_channels = fc_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.accuracy = Accuracy(topk=self.topk)
        self.fp16_enabled = False

        self.normalization = "identity"
        self.pooling = "gap"
        self.dim_reduce_output = 512
        self.parts_attention_pooling_head = init_part_attention_pooling_head(self.normalization, self.pooling, self.dim_reduce_output)
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool1d(1)

        self._init_layers()

    def _init_layers(self):
        """Initialize fc layers."""
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            in_channels = self.in_channels if i == 0 else self.fc_channels
            self.fcs.append(
                FcModule(in_channels, self.fc_channels, self.norm_cfg,
                         self.act_cfg))
        in_channels = self.in_channels if self.num_fcs == 0 else \
            self.fc_channels
        self.fc_out = nn.Linear(in_channels, self.out_channels)
        if self.loss_cls:
            self.bn = nn.BatchNorm1d(self.out_channels)
            self.classifier = nn.Linear(self.out_channels, self.num_classes)

    @auto_fp16()
    def forward_train(self, x:torch.Tensor, vis_masks:torch.Tensor, is_train=False):
        """Model forward.
        Input:
            x(B, 512, 8, 4): Feature Map
            vis_masks (B, 4, 4, 8): Bool Map with (head, torso, legs, feet)
        Output:
            norm_features: l2 norm [head, torso, legs, feet, global] with shape of (B,5,512)
        """
        
        vis_masks = torch.transpose(vis_masks, 2, 3)  # (B, 4, 8, 4)
        # for GAP
        # for i in range(vis_masks.shape[1]):
        #     vis_masks[:,i,:,:] = vis_masks[:,i,:,:] / max(1e-6, vis_masks[:,i,:,:].sum())
        # vis_masks = vis_masks.reshape(vis_masks.shape[0], vis_masks.shape[1], -1)   # (B,4,HXW)
        # part_features = torch.matmul(vis_masks, x.transpose(1,2))  # (B,4,512), mask and sum
        part_features = self.parts_attention_pooling_head(x, vis_masks)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B,512,HXW)
        global_feature = self.gap(x).reshape(x.shape[0], 1, -1) # (B,1,512)
        all_features = torch.concat([part_features,global_feature],dim=1)  # (B,5,512)  (head, torse, legs, feet, whole)
        
        all_features = F.normalize(all_features, p=2, dim=2)  # (B,5,512)

        if is_train:
            feats_bn = self.bn(global_feature)
            cls_score = self.classifier(feats_bn)
            return (all_features, cls_score)
        return (all_features, )

    @force_fp32(apply_to=('feats', 'cls_score'))
    def loss(self, gt_label, feats, vis_masks=None, cls_score=None):
        """Compute losses."""
        losses = dict()

        if self.loss_triplet:
            losses['part_triplet_loss'] = self.loss_triplet(feats, gt_label, vis_masks)

        if self.loss_cls:
            assert cls_score is not None
            losses['ce_loss'] = self.loss_cls(cls_score, gt_label)
            # compute accuracy
            acc = self.accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }

        return losses



### For Pooling (reference from bpreid)###
class GlobalMaskWeightedPoolingHead(nn.Module):
    def __init__(self, depth, normalization='identity'):
        super().__init__()
        if normalization == 'identity':
            self.normalization = nn.Identity()
        elif normalization == 'batch_norm_3d':
            self.normalization = torch.nn.BatchNorm3d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_2d':
            self.normalization = torch.nn.BatchNorm2d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_1d':
            self.normalization = torch.nn.BatchNorm1d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            raise ValueError('normalization type {} not supported'.format(normalization))

    def forward(self, features, part_masks):
        """Pooling
        Input:
            features(1,512,8,4): 
            part_masks(4,8,4):
        Output:
            xxx
        """
        features = torch.unsqueeze(features, 2)  # (1,512,1,8,4)
        # features = 
        part_masks = torch.unsqueeze(part_masks, 2)  # (4,8,4)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = self.global_pooling(parts_features)
        parts_features = parts_features.view(N, M, -1)
        return parts_features

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def init_part_attention_pooling_head(normalization, pooling, dim_reduce_output):
    if pooling == 'gap':
        parts_attention_pooling_head = GlobalAveragePoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gmp':
        parts_attention_pooling_head = GlobalMaxPoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gwap':
        parts_attention_pooling_head = GlobalWeightedAveragePoolingHead(dim_reduce_output, normalization)
    else:
        raise ValueError('pooling type {} not supported'.format(pooling))
    return parts_attention_pooling_head

class GlobalMaxPoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveMaxPool2d((1, 1))


class GlobalAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveAvgPool2d((1, 1))

class GlobalWeightedAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = torch.sum(parts_features, dim=(-2, -1))
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))
        part_masks_sum = torch.clamp(part_masks_sum, min=1e-6)
        parts_features_avg = torch.div(parts_features, part_masks_sum)
        parts_features = parts_features_avg.view(N, M, -1)
        return parts_features