# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models import ImageClassifier
from mmcv.runner import auto_fp16

from ..builder import REID

import torch.nn as nn


@REID.register_module()
class PartWeightedClassifier(ImageClassifier):
    """Base class for re-identification."""

    def forward_train(self, imgs, gt_labels, vis_map, vis_indicator=None, is_vis_att_map=False, is_train=False, **kwargs):
        """"Training forward function.
        img (B, 3, 224, 160):
        vis_map (B, 8, 4): Bool Map to get part features
        """
        if imgs.ndim == 5:
            # change the shape of image tensor from NxSxCxHxW to NSxCxHxW
            # where S is the number of samples by triplet sampling
            imgs = imgs.view(-1, *imgs.shape[2:])
            # change the shape of label tensor from NxS to NS
            gt_labels = gt_labels.view(-1)
        x = self.extract_feat(imgs, stage="backbone")  # B, 512, 8, 4 (B,C,H,W)

        if is_vis_att_map:
            att_map = self.visualize_att_map(x[0])
        head_outputs = self.head.forward_train(x[0], vis_map, vis_indicator, is_train=is_train)  # (B,5,512), [part_features,global_feature]

        losses = dict()
        loss, loss_summary = self.head.loss(*head_outputs, gt_labels)
        losses.update(loss_summary)
        if is_vis_att_map:
            return head_outputs, loss, losses, att_map
        return head_outputs, loss, losses

    @auto_fp16(apply_to=('img', ), out_fp32=True)
    def simple_test(self, img, vis_map, vis_indicators, **kwargs):
        """Test without augmentation."""
        if img.nelement() > 0:
            x = self.extract_feat(img, stage="backbone")  # B, 512, 8, 4
            head_outputs = self.head.forward_train(x[0], vis_map, vis_indicators, is_train=True)
            vis_score = head_outputs[1]
            cls_score = head_outputs[2]  # dict of logits
            return vis_score, cls_score
        else:
            return img.new_zeros(0, self.head.num_classes)
    
    def extract_features(self, img, vis_map, is_vis_att_map, **kwargs):
        """Test without augmentation."""
        if img.nelement() > 0:
            x = self.extract_feat(img, stage="backbone")  # B, 512, 8, 4
            if is_vis_att_map:
                att_map = self.visualize_att_map(x[0])
            head_outputs = self.head.forward_train(x[0], vis_map)
            feats = head_outputs[0]
            if is_vis_att_map:
                return feats, att_map
            return feats
        else:
            return img.new_zeros(0, self.head.out_channels)
    
    def visualize_att_map(self, x):
        """
        Input:
            x: (B,C,H,W)
        Output:
            vis_att_map: (B,H,W)
        """
        h, w = x.shape[2], x.shape[3]
        m = nn.AdaptiveAvgPool1d(1)
        x = x.reshape(x.shape[0], h*w, x.shape[1])  # (B,H,W,512)
        return m(x).reshape(x.shape[0], h, w)
