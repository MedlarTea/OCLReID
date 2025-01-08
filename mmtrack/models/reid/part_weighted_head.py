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

from mmtrack.models.reid.utils import *
from mmtrack.models.reid.losses.GiLt_loss import GiLtLoss
from mmtrack.models.identifier.utils.utils import maybe_cuda

@HEADS.register_module()
class PartWeightedHead(BaseHead):
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
                 use_ori,
                 use_visibility,
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
        super(PartWeightedHead, self).__init__(init_cfg)
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
        
        self.use_ori = use_ori
        self.use_visibility = use_visibility
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
        self.pooling = "gwap"
        self.dim_reduce_output = 512
        self.parts_attention_pooling_head = init_part_attention_pooling_head(self.normalization, self.pooling, self.dim_reduce_output)
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.part_nums = 4 if not self.use_ori else 8  # (front, back) x (head, torso, legs, feet, whole)
        losses_weights = {
            GLOBAL: {'id': 1., 'tr': 0.},
            FOREGROUND: {'id': 0., 'tr': 0.},
            CONCAT_PARTS: {'id': 1., 'tr': 0.},  # hanjing
            PARTS: {'id': 0., 'tr': 1.}
        }

        self.GiLt = GiLtLoss(losses_weights=losses_weights, use_visibility_scores=use_visibility, triplet_margin=1.5, loss_name='part_averaged_triplet_loss', use_gpu=True)
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
        

        ### Init id classifier referred from bpreid  ###
        self.global_identity_classifier = nn.ModuleList(
                [
                    BNClassifier(self.in_channels, self.num_classes)
                    for _ in range(self.part_nums//4)
                ]
            )
        self.concat_parts_identity_classifier = nn.ModuleList(
                [
                    BNClassifier(4 * self.in_channels, self.num_classes)
                    for _ in range(self.part_nums//4)
                ]
            )
        self.parts_identity_classifier = nn.ModuleList(
                [
                    BNClassifier(self.in_channels, self.num_classes)
                    for _ in range(self.part_nums)
                ]
            )
        # self.global_identity_classifier = BNClassifier(self.in_channels, self.num_classes)
        # self.background_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        # self.foreground_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)

        ### Original settings ###
        # self.fc_out = nn.Linear(in_channels, self.out_channels)
        # if self.loss_cls:
        #     self.bn = nn.BatchNorm1d(self.out_channels)
        #     self.classifier = nn.Linear(self.out_channels, self.num_classes)

    @auto_fp16()
    def forward_train(self, x:torch.Tensor, vis_masks:torch.Tensor, vis_indicators=None, is_train=False):
        """Model forward.
        Input:
            x(B, 512, 8, 4): Feature Map
            vis_masks (B, 4, 4, 8): Bool Map with (head, torso, legs, feet)
            vis_indicators: (B, 10): Bool indicator combining info of orientation and parts
        Output:
            
            norm_features: l2 norm [head, torso, legs, feet, global] with shape of (B,5,512)
        """
        if vis_masks.dtype is not torch.float32:
            vis_masks = vis_masks.to(torch.float32)
        
        vis_masks = torch.transpose(vis_masks, 2, 3)  # (B, 4, 8, 4)
        part_features = self.parts_attention_pooling_head(x, vis_masks)  # (B,4,512)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B,512,HXW)
        global_feature = self.gap(x).reshape(x.shape[0], 1, -1) # (B,1,512)

        ### hanjing ###
        part_features_norm = F.normalize(part_features, p=2, dim=2)  # (B,4,512)
        global_feature_norm = F.normalize(global_feature, p=2, dim=2)  # (B,1,512)

        all_features = torch.concat([part_features_norm,global_feature_norm],dim=1)  # (B,5,512)  (head, torse, legs, feet, whole)

        if self.use_ori:
            part_features_norm = part_features_norm.repeat(1,2,1)

        if is_train:
            # Concatenated part features
            _, global_cls_score = self.global_identity_classification(self.in_channels, global_feature[0], torch.squeeze(global_feature))  # [N, D], [N, 2or1, num_classes]

            concat_parts_embeddings = part_features.flatten(1, 2)  # [N, K*D]
            _, concat_parts_cls_score = self.concat_parts_identity_classification(self.part_nums//2*self.in_channels, concat_parts_embeddings[0],concat_parts_embeddings)  # [N, K*D], [N, 2or1, num_classes]

            original_expand_parts_embeddings, parts_cls_score = self.parts_identity_classification(self.in_channels, part_features[0], part_features)  # [N, K, D], [N, K, num_classes]

            # Visualize indicator to visibility
            global_visibility = maybe_cuda(torch.zeros((vis_indicators.size(0), self.part_nums//4)))
            parts_visibility = maybe_cuda(torch.zeros((vis_indicators.size(0), self.part_nums)))
            for i in range(vis_indicators.size(1)):
                if i < 4:
                    parts_visibility[:, i] = vis_indicators[:, i]
                elif i > 5 and i < 9:
                    parts_visibility[:, i-1] = vis_indicators[:, i]
                elif i == 4 or i == 9:
                    global_visibility[:, i//5] = vis_indicators[:, i]

            embeddings = {
                GLOBAL: global_feature_norm,  # [N, D]
                CONCAT_PARTS: concat_parts_embeddings,  # [N, K*D]
                PARTS: part_features_norm,  # [N, K*2or1, D]
                # PARTS: original_expand_parts_embeddings,  # [N, K*2or1, D]
                # PARTS: part_features,  # [N, K, D]

                # BN_GLOBAL: bn_global_embeddings,  # [N, D]
                # BN_CONCAT_PARTS: bn_concat_parts_embeddings,  # [N, K*D]
                # BN_PARTS: bn_parts_embeddings,  #  [N, K, D]
            }
            visibility_scores = {
                GLOBAL: global_visibility.bool(),  # [N, 2or1]
                CONCAT_PARTS: global_visibility.bool(),  # [N, 2or1]
                PARTS: parts_visibility.bool(),  # [N, K]
            }
            id_cls_scores = {
                GLOBAL: global_cls_score,  # [N, 2or1, num_classes]
                CONCAT_PARTS: concat_parts_cls_score,  # [N, 2or1, num_classes]
                PARTS: parts_cls_score,  # [N, K, num_classes]
            }
            return (embeddings, visibility_scores, id_cls_scores)
        return (all_features, )

    # @force_fp32(apply_to=('feats', 'cls_score'))
    # def loss(self, gt_label, feats, vis_masks=None, cls_score=None):
    def loss(self, embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids):
        """Compute losses."""

        # loss_summary is a dict with keys of "GLOBAL, CONCAT_PARTS, PARTS", and includes keys of "c, a, t, tt, vt" per item
        loss, loss_summary = self.GiLt(embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids)

        return loss, loss_summary

    def parts_identity_classification(self, D, N, parts_embeddings):
        """
        parts_embeddings: 2/4
        """
        # apply K classifiers on each of the K part embedding, each part has therefore it's own classifier weights
        scores = []
        # embeddings = []
        original_embeddings = []
        for i, parts_identity_classifier in enumerate(self.parts_identity_classifier):
            bn_part_embeddings, part_cls_score = parts_identity_classifier(parts_embeddings[:, i%(self.part_nums//2)])
            scores.append(part_cls_score.unsqueeze(1))
            # embeddings.append(bn_part_embeddings.unsqueeze(1))
            original_embeddings.append(parts_embeddings[:, i%(self.part_nums//2)].unsqueeze(1))
        part_cls_score = torch.cat(scores, 1)
        # bn_part_embeddings = torch.cat(embeddings, 1)
        original_embeddings = torch.cat(original_embeddings, 1)
        return original_embeddings, part_cls_score

    def global_identity_classification(self, D, N, global_embeddings):
        # apply K classifiers on each of the K part embedding, each part has therefore it's own classifier weights
        scores = []
        embeddings = []
        for i, global_identity_classifier in enumerate(self.global_identity_classifier):
            bn_global_embeddings, global_cls_score = global_identity_classifier(global_embeddings)
            scores.append(global_cls_score.unsqueeze(1))
            embeddings.append(bn_global_embeddings.unsqueeze(1))
        global_cls_score = torch.cat(scores, 1)
        bn_global_embeddings = torch.cat(embeddings, 1)
        return bn_global_embeddings, global_cls_score
    
    def concat_parts_identity_classification(self, D, N, c_embeddings):
        # apply K classifiers on each of the K part embedding, each part has therefore it's own classifier weights
        scores = []
        embeddings = []
        for i, concat_parts_identity_classifier in enumerate(self.concat_parts_identity_classifier):
            bn_c_embeddings, c_cls_score = concat_parts_identity_classifier(c_embeddings)
            scores.append(c_cls_score.unsqueeze(1))
            embeddings.append(bn_c_embeddings.unsqueeze(1))
        c_cls_score = torch.cat(scores, 1)
        bn_c_embeddings = torch.cat(embeddings, 1)
        return bn_c_embeddings, c_cls_score



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


class BNClassifier(nn.Module):
    # Source: https://github.com/upgirlnana/Pytorch-Person-REID-Baseline-Bag-of-Tricks
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)  # BoF: this doesn't have a big impact on perf according to author on github
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self._init_params()

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

