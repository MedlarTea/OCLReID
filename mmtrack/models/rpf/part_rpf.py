import os.path as osp
import warnings
import torch
from mmdet.models import build_detector
import mmcv

from mmtrack.core import outs2results, results2outs, imshow_tracks
from ..builder import MODELS, build_motion, build_reid, build_tracker, build_identifier
from .base import BaseRobotPersonFollower
import numpy as np

### Our modules ###
import cv2 
import time

from mmtrack.utils.preprocessing import numpy_to_torch
from mmtrack.utils.meters import AverageMeter

from mmtrack.models.pose.PoseEstimateLoader import SPPE_FastPose
from mmtrack.models.pose.fn import draw_single

import mmtrack.models.orientation as orient
from mmtrack.models.orientation.config import update_config_w_yaml, create_model, orientation_cfg

from .utils import process_kpts


@MODELS.register_module()
class PartRPF(BaseRobotPersonFollower):
    def __init__(self,
                 detector=None,
                 reid=None,
                 tracker=None,
                 motion=None,
                 pretrains=None,
                 init_cfg=None,
                 identifier=None,
                 hyper_config=None):
        super().__init__(init_cfg)
        if isinstance(pretrains, dict):
            warnings.warn('DeprecationWarning: pretrains is deprecated, '
                          'please use "init_cfg" instead')
            if detector:
                detector_pretrain = pretrains.get('detector', None)
                if detector_pretrain:
                    detector.init_cfg = dict(
                        type='Pretrained', checkpoint=detector_pretrain)
                else:
                    detector.init_cfg = None
            if reid:
                reid_pretrain = pretrains.get('reid', None)
                if reid_pretrain:
                    reid.init_cfg = dict(
                        type='Pretrained', checkpoint=reid_pretrain)
                else:
                    reid.init_cfg = None


        self.vis_parts = ["HEAD", "TORSO", "LEGS", "FEET", "FRONT", "HEAD", "TORSO", "LEGS", "FEET", "BACK"]
        # init visdom
        self.save = hyper_config.save_vis_result
        self.debug = hyper_config.debug
        if self.debug == True:
            self._init_visdom(hyper_config.visdom_info, self.debug)


        if detector is not None:
            self.detector = build_detector(detector)
            # print("detector: {}".format(self.detector))

        if reid is not None:
            self.reid = build_reid(reid)
            # print("reid: {}".format(self.reid))

        if motion is not None:
            self.motion = build_motion(motion)
            # print("motion: {}".format(self.motion))

        if tracker is not None:
            self.tracker = build_tracker(tracker)
        
        if identifier is not None:
            self.identifier = build_identifier(identifier)
        
        ### build a yolox ###
        # self.detector = PersonDetector(model="yolox-m")

        ## build an orientation estimator ##
        ori_cfg_file = osp.join(osp.dirname(orient.__file__), 'experiments/coco/keypoints.yaml')
        ori_cfg = orientation_cfg
        update_config_w_yaml(ori_cfg, ori_cfg_file)
        self.image_patch_size = np.array(ori_cfg.MODEL.IMAGE_SIZE)
        self.orientation_estimator = create_model(ori_cfg)

        ## build a 2d pose estimator ##
        # self.pose_estimator = SPPE_FastPose(backbone="resnet50", input_height=self.image_patch_size[1], input_width=self.image_patch_size[0], device="cuda:0")
        self.pose_estimator = SPPE_FastPose(backbone="resnet50", input_height=224, input_width=160, device="cuda:0")


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

    def get_from_raw_result_rpf(self, raw_result: dict):
        assert isinstance(raw_result, dict)
        track_bboxes = raw_result.get('track_bboxes', None)
        track_masks = raw_result.get('track_masks', None)
        track_kpts = raw_result.get('track_kpts', None)
        track_oris = raw_result.get('track_oris', None)

        outs_track = results2outs(
            bbox_results=track_bboxes,
            mask_results=track_masks)
        bboxes = outs_track.get('bboxes', None)
        labels = outs_track.get('labels', None)
        ids = outs_track.get('ids', None)
        masks = outs_track.get('masks', None)
        result = {}


        for i, (bbox, label, id, kpts, ori) in enumerate(zip(bboxes, labels, ids, track_kpts, track_oris)):
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
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
        return self.detector.forward_train(*args, **kwargs)

    ### CORE FUNCTION ###
    def simple_test(self,
                    img,
                    img_metas,
                    gt_bbox=None,
                    rescale=False,
                    public_bboxes=None,
                    **kwargs):  
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            gt_bbox (numpy): list of ground truth bboxes for each
                image with shape (4,) in [tl_x, tl_y, br_x, br_y] format.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        # suppose rescale=True
        frame_id = img_metas[0].get('frame_id', -1)
        ### tracker init###
        if frame_id == 0:
            self.tracker.reset()

        ### detect and output bboxes ###
        # The detected bounding boxes have been rescaled to original size if rescale is set to True
        t1 = time.time()
        # real bboxes
        det_results = self.detector.simple_test(
            img, img_metas, rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        bbox_results = det_results[0]
        num_classes = len(bbox_results)
        
        # Only detect people
        if num_classes != 1:
            bbox_results = bbox_results[:1]
            num_classes = 1

        ### begin to track ###
        outs_det = results2outs(bbox_results=bbox_results)
        det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        det_labels = torch.from_numpy(outs_det['labels']).to(img).long()
        self.detection_time.update((time.time()-t1)*1000)
        
        
        ### save results ###
        # track_bboxes: (n,5)--[tl_x, tl_y, br_x, br_y, score] 
        # real track_bboxes
        t2 = time.time()
        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)
        self.tracking_time.update((time.time()-t2)*1000)


        ### 2D pose estimation ###
        track_kpts = []
        if track_bboxes.shape[0] != 0:
            t3 = time.time()
            poses = self.pose_estimator.predict(img.squeeze().cpu(), img_metas, track_bboxes[:, :4].cpu(), track_bboxes[:, 4].cpu(), bbox_ids=track_ids, rescale=rescale)  # add track_ids to track all; or construct detection object to be tracked in MoT.
            track_bboxes = []
            track_ids = []
            for ps in poses:
                track_bboxes.append(torch.cat([ps['bbox'], ps['bbox_score'].unsqueeze(0)]).tolist())
                track_ids.append(ps['bbox_id'].tolist())
                track_kpts.append(torch.cat((ps['keypoints'], ps['kp_score']), axis=1).tolist())
            track_bboxes = torch.Tensor(track_bboxes).to(img)
            track_ids = torch.Tensor(track_ids).to(img).long()
            track_labels = torch.zeros(track_ids.size())
            self.kpts_extraction_time.update((time.time()-t3)*1000)
        
        # print("track_bboxes of joint", track_bboxes)
        # print(track_kpts)

        ### orientation estimation ###
        track_oris = []
        if track_bboxes.shape[0] != 0:
            t4 = time.time()
            _, processed_kpts = process_kpts(poses, input_height=self.image_patch_size[1], input_width=self.image_patch_size[0])
            hoe_outputs = self.orientation_estimator(processed_kpts)
            hoe_outputs = hoe_outputs.detach().cpu().numpy()
            track_oris = hoe_outputs.argmax(axis = 1)*5  # degree
            # track_oris = [170 for i in range(track_bboxes.shape[0])]
            self.ori_time.update((time.time()-t4)*1000)
        
        if self.debug:
            print("\n[ms] Detection {:.3f} ({:.3f})\t Tracking {:.3f} ({:.3f})\t Kpts {:.3f} ({:.3f})\t Ori {:.3f} ({:.3f})".format(
                        self.detection_time.val, self.detection_time.avg, 
                        self.tracking_time.val, self.tracking_time.avg,
                        self.kpts_extraction_time.val, self.kpts_extraction_time.avg,
                        self.ori_time.val, self.ori_time.avg)) 

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        # [tl_x, tl_y, br_x, br_y]
        raw_result = dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'],
            track_kpts=track_kpts,
            track_oris=track_oris,
            gt_bbox=gt_bbox)

        ### select the target person ###
        if len(track_bboxes)!=0 and self.target_id is None:
            self.init(gt_bbox, raw_result)
            if self.target_id is not None:
                self.identifier.init_identifier(target_id=self.target_id, rpf_model=self)

        ### For MOT-only evaluation ###
        # if len(track_bboxes)!=0 and self.target_id is not None:
        #     result = self.get_from_raw_result_rpf(raw_result)
        #     if self.target_id in result.keys():
        #         raw_result["target_bbox"] = result[self.target_id][:4]
        #     else:
        #         raw_result["target_bbox"] = None

        # print("Tracking: {}".format(time.time()-t2))
        # t3 = time.time()
        ### identify the target person ###
        if len(track_bboxes)!=0 and self.target_id is not None:
            # result = self.get_from_raw_result_mot(raw_result)
            result = self.get_from_raw_result_rpf(raw_result)
            # identify the target
            # identify with scaled image
            ident_result = self.identifier.identify(
                img=img,
                img_metas=img_metas,
                model=self,
                tracks=result,
                frame_id=frame_id,
                rescale=rescale,
                gt_bbox=gt_bbox,
                **kwargs
            )
            if ident_result is not None:
                self.target_id = ident_result["target_id"]
                raw_result = {**raw_result, **ident_result}
                # In ReID state
                if self.target_id == -1:
                    raw_result["target_bbox"] = None
                # Good tracking
                else:
                    raw_result["target_bbox"] = result[self.target_id][:4]

        # vis
        if self.debug:
            # show tracking image
            vis_img = self.show_result(img_metas[0]['filename'], raw_result)
            for track_kpt in track_kpts:
                pts = np.array(track_kpt)
                pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
                vis_img = draw_single(vis_img, pts)
        if self.save:
            raw_result["vis_img"] = vis_img
        if self.visdom is not None:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            self.visdom.register(numpy_to_torch(vis_img).squeeze(0), "image", 0, "Tracking Image")

        return raw_result
    
    

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