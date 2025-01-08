import numpy as np
import math

import mmcv
from mmtrack.apis import inference_mot, init_model, inference_sot, inference_rpf
from mmtrack.core import imshow_tracks, results2outs

import time
SELECT_TARGET_THRESHOLD = 100

class Tracker():
    def __init__(self, tracker_type, config, checkpoint, hyper_config, seed, identifier_config=None) -> None:
        self.tracker_type = tracker_type
        self.config = config
        self.identifier_config = identifier_config
        self.checkpoint = checkpoint
        self.image_shape = hyper_config.image_shape
        self.device = hyper_config.device
        self.SELECT_TARGET_THRESHOLD = hyper_config.select_target_threshold

        self.init_box = None  # For SOT only
        self.target_id = None  # For MOT only 
        self.obj_tracker = self.init_tracker(hyper_config=hyper_config, seed=seed) 

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

    def init_target_id(self, result, gt_bbox):
        if result is None:
            return None, None
        max_iou = 0.1
        print("\ngt: {}".format(gt_bbox))
        for id in result.keys():
            c_bbox = result[id]
            iou = self.get_iou(c_bbox, gt_bbox)
            print("es: {}, iou: {:.3f}".format(c_bbox, iou))
            if iou > self.SELECT_TARGET_THRESHOLD and iou > max_iou:
                max_iou = iou
                self.target_id = id
        return None, self.target_id

    def init_target_bbox(self, target_box):
        return target_box

    def get_distance(self, box1, box2):
        return math.sqrt(math.pow((box1[0]+box1[2])/2-(box2[0]+box2[2])/2, 2) + math.pow((box1[1]+box1[3])/2-(box2[1]+box2[3])/2, 2))
    
    def init_tracker(self, hyper_config, seed=123):
        self.init_box = None  # For SOT only
        self.target_id = None  # For MOT only
        return init_model(self.config, self.checkpoint, device=self.device, hyper_config=hyper_config, identifier_config=self.identifier_config, seed=seed)
    
    def get_from_raw_result_mot(self, raw_result: dict):
        assert isinstance(raw_result, dict)
        track_bboxes = raw_result.get('track_bboxes', None)
        track_masks = raw_result.get('track_masks', None)

        outs_track = results2outs(
            bbox_results=track_bboxes,
            mask_results=track_masks,
            mask_shape=self.image_shape[:2])
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
        target_bbox = raw_result.get('target_bbox', None)

        outs_track = results2outs(
            bbox_results=track_bboxes,
            mask_results=track_masks,
            mask_shape=self.image_shape[:2])
        bboxes = outs_track.get('bboxes', None)
        labels = outs_track.get('labels', None)
        ids = outs_track.get('ids', None)
        masks = outs_track.get('masks', None)
        result = {}
        for i, (bbox, label, id) in enumerate(zip(bboxes, labels, ids)):
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            score = float(bbox[-1])
            result[int(id)] = [int(x1), int(y1), int(x2), int(y2)]
        return result, target_bbox

    def get_from_raw_result_sot(self, raw_result: dict):
        assert isinstance(raw_result, dict)
        track_bboxes = raw_result.get('track_bboxes', None)
        return [int(track_bboxes[0]), int(track_bboxes[1]), int(track_bboxes[2]), int(track_bboxes[3])]

    def infer(self, image_fname, target_gt_bbox, frame_id):
        """Infer the result by trackers of mmTracking
        Input:
            image_fname: image filename
            frame_id: index of this image, used for tracking
        Return:
            distance (float): 1) euclidean distance between the centers of the target box and the estimated box; 2) None if the estimated box is None
            match_box (List): 1) the estimated bbox with [x1, y1, x2, y2]; 2) None if the estimated box is None
        """
        return_result = {}
        if self.tracker_type == "mot":
            raw_result = inference_mot(self.obj_tracker, image_fname, frame_id=frame_id)
            result = self.get_from_raw_result_mot(raw_result)
            return_result["all_tracks"] = result
            if self.target_id == None:
                return_result["distance"], self.target_id = self.init_target_id(result, target_gt_bbox)
                return_result["match_box"] = result[self.target_id] if self.target_id != None else None
            else:
                if self.target_id in result.keys():
                    return_result["distance"] = self.get_distance(result[self.target_id], target_gt_bbox)
                    return_result["match_box"] = result[self.target_id]
                else:
                    return_result["distance"] = None
                    return_result["match_box"] = None
            # return distance, match_box, result
        
        elif self.tracker_type == "sot":
            if self.init_box is None:
                self.init_box = self.init_target_bbox(target_gt_bbox)
            img = mmcv.imread(image_fname)
            raw_result = inference_sot(self.obj_tracker, img, self.init_box, frame_id=frame_id)
            return_result["match_box"] = self.get_from_raw_result_sot(raw_result)
            return_result["distance"] = self.get_distance(return_result["match_box"], target_gt_bbox)
            # return distance, match_box, None
        
        elif self.tracker_type == "rpf":
            raw_result = inference_rpf(self.obj_tracker, image_fname, frame_id=frame_id, gt_bbox=target_gt_bbox)
            result, target_bbox = self.get_from_raw_result_rpf(raw_result)
            if target_bbox is not None:
                return_result["distance"] = self.get_distance(target_bbox, target_gt_bbox) if target_gt_bbox is not None else -1
                return_result["match_box"] = target_bbox

        return return_result, raw_result
        

    

    

    
    
    