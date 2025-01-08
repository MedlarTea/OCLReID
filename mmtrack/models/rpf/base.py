# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import mmcv
import cv2
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmtrack.core import imshow_tracks, results2outs
from mmtrack.utils import get_root_logger

# fro debug
from mmtrack.utils import Visdom
import time

vis_parts = ["HEAD", "TORSO", "LEGS", "FEET", "FRONT", "HEAD", "TORSO", "LEGS", "FEET", "BACK"]

class BaseRobotPersonFollower(BaseModule, metaclass=ABCMeta):
    """Base class for robot person follower."""

    def __init__(self, init_cfg=None, hyper_config=None):
        super(BaseRobotPersonFollower, self).__init__(init_cfg)
        self.logger = get_root_logger()
        self.visdom = None
        self.hyper_config = hyper_config

    @property
    def with_detector(self):
        """bool: whether the framework has a detector."""
        return hasattr(self, 'detector') and self.detector is not None

    @property
    def with_reid(self):
        """bool: whether the framework has a reid model."""
        return hasattr(self, 'reid') and self.reid is not None

    @property
    def with_motion(self):
        """bool: whether the framework has a motion model."""
        return hasattr(self, 'motion') and self.motion is not None

    @property
    def with_track_head(self):
        """bool: whether the framework has a track_head."""
        return hasattr(self, 'track_head') and self.track_head is not None

    @property
    def with_tracker(self):
        """bool: whether the framework has a tracker."""
        return hasattr(self, 'tracker') and self.tracker is not None
    
    @abstractmethod
    def simple_test(self, img, img_metas, gt_bbox, **kwargs):
        """Test function with a single scale."""
        pass
    
    @abstractmethod
    def forward_train(self, imgs, img_metas, gt_bbox, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    def forward_test(self, imgs, img_metas, gt_bbox, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], gt_bbox, **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, gt_bbox=None, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, gt_bbox, **kwargs)
        else:
            return self.forward_test(img, img_metas, gt_bbox, **kwargs)

    def show_result(self,
                    img,
                    result,
                    score_thr=0.0,
                    thickness=1,
                    font_scale=0.5,
                    show=False,
                    out_file=None,
                    wait_time=0,
                    backend='cv2',
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                - The value of key 'track_bboxes' is list with length
                num_classes, and each element in list is ndarray with
                shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.
                - The value of key 'det_bboxes' is list with length
                num_classes, and each element in list is ndarray with
                shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_bboxes = result.get('track_bboxes', None)
        track_masks = result.get('track_masks', None)
        track_oris = result.get('track_oris', None)

        ### draw target person following info ### 
        target_bbox = result.get('target_bbox', None)
        target_id = result.get('target_id', None)
        target_conf = result.get('target_conf', None)
        tracks_target_conf_bbox = result.get('tracks_target_conf_bbox', None)
        att_maps = result.get('att_maps', None)
        visibility_maps = result.get('visibility_maps', None)
        vis_indicator = result.get('vis_indicator', None)
        gt_bbox = result.get('gt_bbox', None)
        state_name = result.get('state', "None")

        if isinstance(img, str):
            img = mmcv.imread(img)
        outs_track = results2outs(
            bbox_results=track_bboxes,
            mask_results=track_masks,
            mask_shape=img.shape[:2])
        img = imshow_tracks(
            img,
            outs_track.get('bboxes', None),
            outs_track.get('labels', None),
            outs_track.get('ids', None),
            outs_track.get('masks', None),
            classes=self.CLASSES,
            score_thr=score_thr,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=None,  # I change to None
            wait_time=wait_time,
            backend=backend)

        ### draw target person following info ### 
        conf_text_width, conf_text_height = 9, 13
        state_text_width, state_text_height = 18, 22
        state_font_scale = 0.8
        thickness=4
        font_scale=0.4
        # draw state
        # state_width = len(state_name) * state_text_width
        # img[40-state_text_height-5:40, :state_width, :] = (255,255,255)
        img[40-state_text_height-5:40, :, :] = (255,255,255)
        # cv2.putText(img, state_name, (0, 35), cv2.FONT_HERSHEY_COMPLEX, state_font_scale, color=(0, 0, 0))

        # draw concatenated image
        # if att_maps is not None and visibility_maps is not None:
        #     vis_img = np.zeros((img.shape[0], img.shape[1]+img.shape[0], 3), dtype=np.uint8)
        #     vis_img.fill(0)
        # else:
        #     vis_img = np.zeros(img.shape, dtype=np.uint8)
        #     vis_img.fill(0)
        vis_img = np.zeros(img.shape, dtype=np.uint8)
        vis_img.fill(0)
        
        # draw target conf
        if tracks_target_conf_bbox != None:
            for idx in tracks_target_conf_bbox:
                if idx == target_id and target_bbox is not None:
                    bbox_color=(0,255,0)
                    x1, y1, x2, y2 = target_bbox
                    # bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)
                    # target conf
                    if target_conf is not None:
                        text = "{:.3f}".format(target_conf)
                        width = len(text) * conf_text_width
                        img[y2-conf_text_height-4:y2, x1:x1 + width, :] = bbox_color
                        cv2.putText(img, text, (x1, y2-4), cv2.FONT_HERSHEY_COMPLEX, font_scale, color=(0, 0, 0))

                    # draw att_map and vis_map
                    # if att_maps is not None and visibility_maps is not None:
                    #     att_img = np.array(att_maps[idx])
                    #     att_img = (att_img - att_img.min()) / (att_img.max() - att_img.min()) * 255  # min-max norm
                    #     att_img = np.uint8(att_img)
                    #     att_img = cv2.applyColorMap(att_img, cv2.COLORMAP_JET)
                    #     att_img = cv2.resize(att_img, (img.shape[0]//2, img.shape[0]))
                    #     vis_img[:, img.shape[1]:img.shape[1]+img.shape[0]//2] = att_img

                    #     # print(np.array(visibility_maps[idx]).shape)
                    #     visibility_map = np.array(visibility_maps[idx])
                    #     visibility_map = visibility_map.transpose((0,2,1))  # (4,8,4)
                    #     visibility_img = np.zeros(visibility_map.shape[1:])  # (8,4)
                    #     for i in range(visibility_map.shape[0]):
                    #         visibility_img[i*2:(i+1)*2, :] = 1 if visibility_map[i].sum() != 0 else 0
                    #     visibility_img = np.uint8(visibility_img*255)
                    #     visibility_img = cv2.applyColorMap(visibility_img, cv2.COLORMAP_JET)
                    #     visibility_img = cv2.resize(visibility_img, (img.shape[0]//2, img.shape[0]))
                    #     vis_img[:, img.shape[1]+img.shape[0]//2:] = visibility_img
                    
                    if vis_indicator is not None:
                        pos_vis_indicator = vis_indicator[idx]
                        str_print = [state_name]
                        for i in range(len(pos_vis_indicator)):
                            if pos_vis_indicator[i] == 1:
                                str_print += [vis_parts[i]]
                        str_print = ",".join(str_print)
                        cv2.putText(img, str_print, (0, 35), cv2.FONT_HERSHEY_COMPLEX, state_font_scale, color=(0, 0, 0))


                elif idx != -1:
                    bbox_color=(0,255,0)
                    track_conf = tracks_target_conf_bbox[idx][0]
                    # None when the target is missed, predictor would not predict confidence
                    if track_conf is not None:
                        # print(tracks_target_conf_bbox[idx][1])
                        x1, y1, x2, y2 = tracks_target_conf_bbox[idx][1]
                        text = "{:.3f}".format(track_conf)
                        width = len(text) * conf_text_width
                        img[y2-conf_text_height-4:y2, x1:x1 + width, :] = bbox_color
                        cv2.putText(img, text, (x1, y2-4), cv2.FONT_HERSHEY_COMPLEX, font_scale, color=(0, 0, 0))
                    # if vis_indicator is not None:
                    #     neg_vis_indicator = vis_indicator[idx]
                    #     str_print = [state_name]
                    #     for i in range(len(neg_vis_indicator)):
                    #         if neg_vis_indicator[i] == 1:
                    #             str_print += [vis_parts[i]]
                    #     str_print = ",".join(str_print)
                    #     cv2.putText(img, str_print, (0, 35), cv2.FONT_HERSHEY_COMPLEX, state_font_scale, color=(0, 0, 0))

        # draw orientation
        bboxes = outs_track.get('bboxes', None)
        if track_oris is not None and bboxes is not None:
            for index, track_bbox in enumerate(bboxes):
                bbox_color=(0,255,0)
                # print(track_bbox)
                x1, y1, x2, y2 = track_bbox[:4].astype(np.int32)
                text = "{:d}".format(track_oris[index])
                width = len(text) * conf_text_width
                img[y2-conf_text_height-4:y2, x2-width:x2, :] = bbox_color
                cv2.putText(img, text, (x2-width+2, y2-4), cv2.FONT_HERSHEY_COMPLEX, font_scale, color=(0, 0, 0))
            
        # draw ground truth as triangle
        if gt_bbox is not None:
            bbox_color=(0,255,0)
            x1, y1, x2, y2 = gt_bbox
            # gt bbox
            self.draw_gt_triangle(img, gt_bbox)
        
        vis_img[:, :img.shape[1]] = img
        if out_file is not None:
            mmcv.imwrite(vis_img, out_file)
        return vis_img
    
    def freeze_module(self, module):
        """Freeze module during training."""
        if isinstance(module, str):
            modules = [module]
        else:
            if not (isinstance(module, list) or isinstance(module, tuple)):
                raise TypeError('module must be a str or a list.')
            else:
                modules = module
        for module in modules:
            m = getattr(self, module)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def draw_gt_triangle(self, image, gt_bbox):
        center_uv = [int((gt_bbox[0]+gt_bbox[2])/2), int(gt_bbox[3])]
        bottom_left_uv = [center_uv[0]-20, center_uv[1]+20]
        bottom_right_uv = [center_uv[0]+20, center_uv[1]+20]
        if center_uv[1] >= image.shape[0]:
            center_uv[1] = image.shape[0]-20
            bottom_left_uv[1] = image.shape[0]-1
            bottom_right_uv[1] = image.shape[0]-1
        return cv2.polylines(image, [np.array([center_uv, bottom_left_uv, bottom_right_uv])], True, (0, 255, 0), 3)
    
    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)

                # Show help
                help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                            'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                            'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                            'block list.'
                self.visdom.register(help_text, 'text', 1, 'Help')
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True
    
    

    
    
