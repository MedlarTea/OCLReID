import warnings
import torch
from mmdet.models import build_detector

from mmtrack.core import outs2results, results2outs
from ..builder import MODELS, build_motion, build_reid, build_tracker, build_identifier
from .base import BaseRobotPersonFollower
import numpy as np

import cv2 
import time

from mmtrack.utils.preprocessing import numpy_to_torch
from mmtrack.utils.meters import AverageMeter

from mmtrack.models.pose.PoseEstimateLoader import SPPE_FastPose
from mmtrack.models.pose.fn import draw_single


@MODELS.register_module()
class GlobalRPF(BaseRobotPersonFollower):
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

        ## build a 2d pose estimator ##
        self.pose_estimator = SPPE_FastPose(backbone="resnet50", input_height=224, input_width=160, device="cuda:0")

        self.SELECT_TARGET_THRESHOLD = hyper_config.select_target_threshold
        # self.identifier = None
        self.target_id = None
        self.target_bbox = None

        self.detection_time = AverageMeter()
        self.tracking_time = AverageMeter()
        self.kpts_extraction_time = AverageMeter()

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

        outs_track = results2outs(
            bbox_results=track_bboxes,
            mask_results=track_masks)
        bboxes = outs_track.get('bboxes', None)
        labels = outs_track.get('labels', None)
        ids = outs_track.get('ids', None)
        masks = outs_track.get('masks', None)
        result = {}


        for i, (bbox, label, id, kpts) in enumerate(zip(bboxes, labels, ids, track_kpts)):
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            score = float(bbox[-1])
            result[int(id)] = [int(x1), int(y1), int(x2), int(y2), score, kpts]
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
        # print(num_classes)

        ### begin to track ###
        # print("\nbbox_results", bbox_results)
        # bbox_results (list[np.ndarray])
        outs_det = results2outs(bbox_results=bbox_results)
        det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        det_labels = torch.from_numpy(outs_det['labels']).to(img).long()
        # print("Num of bboxes: {}".format(det_bboxes.shape))

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
        # print("track_bboxes of track: ", track_bboxes)
        # print("track_ids: ", track_ids)

        # print("Num of track_bboxes: {}".format(track_bboxes.shape))


        ### 2D pose estimation ###
        t3 = time.time()
        track_kpts = []
        if track_bboxes.shape[0] != 0:
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
        if self.debug:
            print("\n[ms] Detection {:.3f} ({:.3f})\t Tracking {:.3f} ({:.3f})\t Kpts {:.3f} ({:.3f})".format(
                        self.detection_time.val, self.detection_time.avg, 
                        self.tracking_time.val, self.tracking_time.avg,
                        self.kpts_extraction_time.val, self.kpts_extraction_time.avg))
        # print("track_bboxes of joint", track_bboxes)
        # print(track_kpts)


        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        # print(track_results['bbox_results'])
        # [tl_x, tl_y, br_x, br_y]
        raw_result = dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'],
            track_kpts=track_kpts,
            gt_bbox=gt_bbox)
        
        

        ### select the target person ###
        if len(track_bboxes)!=0 and self.target_id is None:
            self.init(gt_bbox, raw_result)
            # print("dist: {}".format(min_dist))
            # print("gt bbox: {}".format(gt_bbox))
            # print("matched target bbox: {}".format(self.target_bbox))
            if self.target_id is not None:
                self.identifier.init_identifier(target_id=self.target_id, rpf_model=self)
                # self.identifier = NaiveIdentifier(target_id=self.target_id, visdom=self.visdom, debug=self.debug, save=self.save)


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

        ### visualization ###
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
    
    

