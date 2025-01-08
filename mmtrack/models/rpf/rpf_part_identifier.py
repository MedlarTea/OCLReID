import os
import os.path as osp
import warnings
import torch
# from mmdet.models import build_detector
import mmcv

from mmtrack.core import outs2results, results2outs, imshow_tracks
from ..builder import MODELS, build_motion, build_reid, build_tracker, build_identifier
from .base import BaseRobotPersonFollower
import numpy as np

### Our modules ###
import cv2 
import time
from mmtrack.utils.meters import AverageMeter

import mmtrack.models.orientation as orient
from mmtrack.models.orientation.config import update_config_w_yaml, create_model, orientation_cfg

from .utils import process_kpts

from mmtrack.models.identifier.utils.utils import maybe_cuda

@MODELS.register_module()
class RPFPartIdentifier(BaseRobotPersonFollower):
    def __init__(self,
                 reid=None,
                 pretrains=None,
                 init_cfg=None,
                 identifier=None,
                 hyper_config=None):
        super().__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            if reid:
                reid_pretrain = pretrains.get('reid', None)
                if reid_pretrain:
                    reid.init_cfg = dict(
                        type='Pretrained', checkpoint=reid_pretrain)
                else:
                    reid.init_cfg = None


        self.vis_parts = ["HEAD", "TORSO", "LEGS", "FEET", "FRONT", "HEAD", "TORSO", "LEGS", "FEET", "BACK"]

        if reid is not None:
            self.reid = build_reid(reid)

        if identifier is not None:
            self.identifier = build_identifier(identifier)

        self.save = hyper_config.save_vis_result
        self.debug = hyper_config.debug
        ## build an orientation estimator ##
        ori_cfg_file = osp.join(osp.dirname(orient.__file__), 'experiments/coco/keypoints.yaml')
        ori_cfg = orientation_cfg
        update_config_w_yaml(ori_cfg, ori_cfg_file)
        self.image_patch_size = np.array(ori_cfg.MODEL.IMAGE_SIZE)
        self.orientation_estimator = create_model(ori_cfg)

        self.SELECT_TARGET_THRESHOLD = hyper_config.select_target_threshold
        # self.identifier = None
        self.target_id = None
        self.target_bbox = None

        self.detection_time = AverageMeter()
        self.tracking_time = AverageMeter()
        self.kpts_extraction_time = AverageMeter()
        self.ori_time = AverageMeter()

        # seed_everything(123)

    def get_from_raw_result_mot(self, raw_result: dict):
        assert isinstance(raw_result, dict)
        track_bboxes = raw_result.get('track_bboxes', None)
        track_masks = raw_result.get('track_masks', None)
        # track_kpts = raw_result.get('track_kpts', None)

        outs_track = results2outs(
            bbox_results=track_bboxes,
            mask_results=track_masks)
        bboxes = outs_track.get('bboxes', None)
        labels = outs_track.get('labels', None)
        ids = outs_track.get('ids', None)
        masks = outs_track.get('masks', None)
        result = {}

        for i, (bbox, label, id) in enumerate(zip(bboxes, labels, ids)):
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            score = float(bbox[-1])
            result[int(id)] = [int(x1), int(y1), int(x2), int(y2)]
        return result

    def aggregate_results(self, track_bboxes, track_ids, track_kpts, track_oris):
        result = {}

        for i, (bbox, id, kpts, ori) in enumerate(zip(track_bboxes, track_ids, track_kpts, track_oris)):
            x1, y1, x2, y2 = bbox[:4]
            score = float(bbox[-1])
            result[int(id)] = [int(x1), int(y1), int(x2), int(y2), score, kpts, ori]
        return result

    def get_iou(self, pred_box, gt_box):
        """
        pred_box : the coordinate for predict bounding box
        gt_box :   the coordinate for ground truth bounding box
        return :   the iou score
        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
        """
        # 1.get the coordinate of inters
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax-ixmin+1., 0.)
        ih = np.maximum(iymax-iymin+1., 0.)

        # 2. calculate the area of inters
        inters = iw*ih

        # 3. calculate the area of union
        uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
            (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
            inters)

        # 4. calculate the overlaps between pred_box and gt_box
        iou = inters / uni

        return iou

    def init(self, gt_bbox, raw_result):
        """Init the target person from candidate_bboxes
        gt_bbox (np.numpy): [tl_x, tl_y, br_x, br_y]
        raw_result: raw result from MOT
        """
        result = self.get_from_raw_result_mot(raw_result)
        max_iou = 0.5
        print("\ngt: {}".format(gt_bbox))
        for id in result.keys():
            c_bbox = result[id]
            iou = self.get_iou(c_bbox, gt_bbox)
            print("es: {}, iou: {:.3f}".format(c_bbox, iou))
            if iou > self.SELECT_TARGET_THRESHOLD and iou > max_iou:
                max_iou = iou
                self.target_id = id
                self.target_bbox = c_bbox
        return max_iou

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        # return self.detector.forward_train(*args, **kwargs)
        return 

    ### CORE FUNCTION ###
    def simple_test(self,
                    img,
                    img_metas,
                    gt_bbox=None,
                    track_ids=None,
                    track_bboxes=None,
                    track_kpts=None,
                    target_id=None,
                    rescale=False,
                    **kwargs):  
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned track_bboxes: 
        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        # suppose rescale=True
        frame_id = img_metas[0].get('frame_id', -1)

        ### orientation estimation ###
        track_oris = []
        if len(track_bboxes) != 0:
            # print(len(track_bboxes))
            # print(track_kpts)
            track_kpts = torch.Tensor(track_kpts).to(img)
            # print(track_ids.shape, track_bboxes.shape, track_kpts.shape)
            t4 = time.time()
            poses = []
            for i in range(len(track_bboxes)):
                poses.append({'keypoints': track_kpts[i,:,:2], 'kp_score': track_kpts[i,:,2], 'bbox': track_bboxes[i]})
            _, processed_kpts = process_kpts(poses, input_height=self.image_patch_size[1], input_width=self.image_patch_size[0])
            hoe_outputs = self.orientation_estimator(processed_kpts)
            hoe_outputs = hoe_outputs.detach().cpu().numpy()
            track_oris = hoe_outputs.argmax(axis = 1)*5  # degree
            # track_oris = [170 for i in range(track_bboxes.shape[0])]
            self.ori_time.update((time.time()-t4)*1000)
        

        ### init the target person ###
        if len(track_bboxes)!=0 and self.target_id is None and target_id is not None:
            self.target_id = target_id
            self.identifier.init_identifier(target_id=self.target_id, rpf_model=self)

        ### identify the target person ###
        ident_result = {}
        if len(track_bboxes)!=0 and self.target_id is not None:
            # result = self.get_from_raw_result_mot(raw_result)
            result = self.aggregate_results(track_bboxes, track_ids, track_kpts.tolist(), track_oris)
            # identify the target
            # identify with scaled image
            ident_result = self.identifier.identify(
                img=img,
                img_metas=img_metas,
                model=self,
                tracks=result,
                frame_id=frame_id,
                rescale=rescale,
                gt_bbox=None,
                **kwargs
            )
            if ident_result is not None:
                self.target_id = ident_result["target_id"]
                # In ReID state
                if self.target_id == -1:
                    ident_result["target_bbox"] = None
                # Good tracking
                else:
                    ident_result["target_bbox"] = result[self.target_id][:4]
            # if (frame_id+1) == 2300:
            #     buffer_images = self.get_buffer_samples(self.identifier)
            #     if buffer_images is not None:
            #         ident_result["buffer_imgs"] = buffer_images

        return ident_result
    
    
    def get_buffer_samples(self, identifier):
        # st_memory_buffer = identifier.classifier.memory_manager.memory["st_set"]
        if not (hasattr(identifier.classifier.memory_manager, "memory") and "lt_set" in identifier.classifier.memory_manager.memory.keys()):
            return None
        # st_memory_buffer = identifier.classifier.memory_manager.memory["st_set"]
        lt_memory_buffer = identifier.classifier.memory_manager.memory["lt_set"]
        lt_memory_buffer_tracker = lt_memory_buffer.buffer_tracker
        buffer_images = {}
        # [4,9]
        for i in range(lt_memory_buffer.vis_map_nums, lt_memory_buffer.part_nums, lt_memory_buffer.part_nums//2):
            buffer_images[i] = {}
            part_buffer_img = getattr(lt_memory_buffer, "buffer_img_{}".format(i))
            # part_buffer_label = getattr(lt_memory_buffer, "buffer_label_{}".format(i))
            pos_index = list(lt_memory_buffer_tracker.part_index_cache[i][1].keys())
            neg_index = list(lt_memory_buffer_tracker.part_index_cache[i][0].keys())
            pos_index = maybe_cuda(torch.Tensor(pos_index).long())
            neg_index = maybe_cuda(torch.Tensor(neg_index).long())
            buffer_images[i][1] = part_buffer_img[pos_index]
            buffer_images[i][0] = part_buffer_img[neg_index]
        return buffer_images

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
        #  raw_result = dict(
        #     det_bboxes=det_results['bbox_results'],
        #     track_bboxes=track_results['bbox_results'],
        #     track_kpts=track_kpts,
        #     track_oris=track_oris,
        #     track_parsings=track_parsings,
        #     gt_bbox=gt_bbox,
        #     )
        assert isinstance(result, dict)
        track_bboxes = result.get('track_bboxes', None)  # cuda
        track_masks = result.get('track_masks', None)  # 
        track_oris = result.get('track_oris', None)
        # track_kpts = result.get('track_kpts', None)
        # track_parsings = result.get('track_parsings', None)

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
                    
                    if vis_indicator is not None:
                        pos_vis_indicator = vis_indicator[idx]
                        part_track_conf = tracks_target_conf_bbox[idx][0]
                        # print(pos_vis_indicator)
                        # print(part_track_conf)
                        str_print = [state_name]
                        for i in range(len(pos_vis_indicator)):
                            if pos_vis_indicator[i] == 1:
                                if part_track_conf[i] is not None:
                                    str_print += [self.vis_parts[i]+":{:.2f}".format(part_track_conf[i])]
                                else:
                                    str_print += [self.vis_parts[i]]
                        str_print = ",".join(str_print)
                        cv2.putText(img, str_print, (0, 35), cv2.FONT_HERSHEY_COMPLEX, state_font_scale, color=(0, 0, 0))

                elif idx != -1:
                    bbox_color=(0,255,0)
                    track_conf = tracks_target_conf_bbox[idx][1]
                    # None when the target is missed, predictor would not predict confidence
                    if track_conf is not None:
                        # print(tracks_target_conf_bbox[idx][1])
                        x1, y1, x2, y2 = tracks_target_conf_bbox[idx][2]
                        text = "{:.3f}".format(track_conf)
                        width = len(text) * conf_text_width
                        img[y2-conf_text_height-4:y2, x1:x1 + width, :] = bbox_color
                        cv2.putText(img, text, (x1, y2-4), cv2.FONT_HERSHEY_COMPLEX, font_scale, color=(0, 0, 0))
                    if vis_indicator is not None and len(tracks_target_conf_bbox) == 1:
                        part_track_conf = tracks_target_conf_bbox[idx][0]
                        pos_vis_indicator = vis_indicator[idx]
                        str_print = [state_name]
                        for i in range(len(pos_vis_indicator)):
                            if pos_vis_indicator[i] == 1:
                                if part_track_conf[i] is not None:
                                    str_print += [self.vis_parts[i]+":{:.2f}".format(part_track_conf[i])]
                                else:
                                    str_print += [self.vis_parts[i]]
                        str_print = ",".join(str_print)
                        cv2.putText(img, str_print, (0, 35), cv2.FONT_HERSHEY_COMPLEX, state_font_scale, color=(0, 0, 0))

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