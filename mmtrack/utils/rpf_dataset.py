""" Dataset construction for lab_reid dataset


"""
import torch
import torch.nn.functional as F
import json
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

from torch.nn import Parameter
import shutil

class Fragment_Part_Dataset(Dataset):
    def __init__(self, dataset_list, cfg, img_scale, device):
        super(Fragment_Part_Dataset, self).__init__()
        self.dataset_list = dataset_list
        self.pipeline = Compose(cfg.data.test.pipeline)
        self.device = device
        self.img_scale = img_scale
    
    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)
    
    def _get_ori(self, ori):
        """Segment the orientation
        Output:
            0 means the person not faces to the camera (Front)
            1 means the person faces to the camera (Back)
        """
        # Front
        if ori > 90 and ori < 270:
            return 0
        # Back
        else:
            return 1

    def _get_single_item(self, index):
        frame_id, fname, bbox, kpt, ori, vis_indicator, vis_map, label = self.dataset_list[index]
        # fname, bbox, label = self.dataset_dict[index]
        data = dict(
            img_info=dict(filename=fname, frame_id=index), 
            img_prefix=None)
        data = self.pipeline(data)
        data['img_metas'] = data['img_metas'][0].data
        img = data["img"][0].unsqueeze(0).to(self.device)
        # print(data['img_metas'])
        # data = collate([data], samples_per_gpu=1)
        # data = scatter(data, [self.device])[0]
        # print(data["img"][0].shape)
        frame_id = torch.Tensor([frame_id]).long().to(self.device)
        crop_img = self._crop_imgs(img, data["img_metas"], bbox, rescale=True)
        label = torch.Tensor([label]).long().to(self.device)
        kpt = torch.Tensor(kpt).float().to(self.device)
        ori = self._get_ori(ori)
        ori = torch.Tensor([ori]).long().to(self.device)
        vis_indicator = torch.Tensor(vis_indicator).bool().to(self.device)
        vis_map = torch.Tensor(vis_map).float().to(self.device)
        return frame_id, crop_img, label, kpt, ori, vis_indicator, vis_map
    
    def _crop_imgs(self, img, img_metas, bbox, rescale=False):
        """Crop the images according to some bounding boxes. Typically for re-
        identification sub-module.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            bboxes (Tensor): of shape (N, 4) or (N, 5).
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the scale of the image. Defaults to False.

        Returns:
            Tensor: Image tensor of shape (N, C, H, W).
        """
        h, w, _ = img_metas['img_shape']
        img = img[:, :, :h, :w]
        
        ### TODO: remove in the future###
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.Tensor(bbox).to(img.device)

        if rescale:
            bbox[:4] *= torch.tensor(img_metas['scale_factor']).to(
                bbox.device)
        bbox[0::2] = torch.clamp(bbox[0::2], min=0, max=w)
        bbox[1::2] = torch.clamp(bbox[1::2], min=0, max=h)

        # crop_imgs = []
        x1, y1, x2, y2 = map(int, bbox)
        if x2 == x1:
            x2 = x1 + 1
        if y2 == y1:
            y2 = y1 + 1
        crop_img = img[:, :, y1:y2, x1:x2]
        if self.img_scale is not None:
            crop_img = F.interpolate(
                crop_img,
                size=self.img_scale,
                mode='bilinear',
                align_corners=False)
        return crop_img.squeeze()

class Fragment_Global_Dataset(Dataset):
    def __init__(self, dataset_list, cfg, img_scale, device):
        super(Fragment_Global_Dataset, self).__init__()
        self.dataset_list = dataset_list
        self.pipeline = Compose(cfg.data.test.pipeline)
        self.device = device
        self.img_scale = img_scale
    
    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        frame_id, fname, bbox, kpt, label = self.dataset_list[index]
        # fname, bbox, label = self.dataset_dict[index]
        data = dict(
            img_info=dict(filename=fname, frame_id=index), 
            img_prefix=None)
        data = self.pipeline(data)
        data['img_metas'] = data['img_metas'][0].data
        img = data["img"][0].unsqueeze(0).to(self.device)
        # print(data['img_metas'])
        # data = collate([data], samples_per_gpu=1)
        # data = scatter(data, [self.device])[0]
        # print(data["img"][0].shape)
        frame_id = torch.Tensor([frame_id]).long().to(self.device)
        crop_img = self._crop_imgs(img, data["img_metas"], bbox, rescale=True)
        label = torch.Tensor([label]).long().to(self.device)
        kpt = torch.Tensor(kpt).float().to(self.device)
        return frame_id, crop_img, label, kpt

    def _crop_imgs(self, img, img_metas, bbox, rescale=False):
        """Crop the images according to some bounding boxes. Typically for re-
        identification sub-module.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            bboxes (Tensor): of shape (N, 4) or (N, 5).
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the scale of the image. Defaults to False.

        Returns:
            Tensor: Image tensor of shape (N, C, H, W).
        """
        h, w, _ = img_metas['img_shape']
        img = img[:, :, :h, :w]
        
        ### TODO: remove in the future###
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.Tensor(bbox).to(img.device)

        if rescale:
            bbox[:4] *= torch.tensor(img_metas['scale_factor']).to(
                bbox.device)
        bbox[0::2] = torch.clamp(bbox[0::2], min=0, max=w)
        bbox[1::2] = torch.clamp(bbox[1::2], min=0, max=h)

        # crop_imgs = []
        x1, y1, x2, y2 = map(int, bbox)
        if x2 == x1:
            x2 = x1 + 1
        if y2 == y1:
            y2 = y1 + 1
        crop_img = img[:, :, y1:y2, x1:x2]
        if self.img_scale is not None:
            crop_img = F.interpolate(
                crop_img,
                size=self.img_scale,
                mode='bilinear',
                align_corners=False)
        return crop_img.squeeze()


class RPF_Part_Dataset(Dataset):
    def __init__(self, sequence, dataset_dir, cfg, height=256, width=128, device="cpu", fragment_nums=10):
        super(RPF_Part_Dataset, self).__init__()
        self.sequence = sequence
        self.dataset_dir = dataset_dir
        self.img_scale = (height, width)
        self.IOU_THRESHOLD = 0.4  # 0.8
        self.device = device
        self.cfg = cfg
        self.fragment_size = None
        self.fragment_nums = fragment_nums

        bboxes_file = osp.join(self.dataset_dir, self.sequence, "debug", "ALL_INFO.json")
        
        self.rpf_datasets = {}
        self._construct_dataset(bboxes_file)
    
    def _construct_dataset(self, bboxes_file):
        frag_dataset = []
        bboxes_file = open(bboxes_file, 'r')
        bboxes_dict = json.load(bboxes_file)
        self.fragment_size = len(bboxes_dict.keys()) // self.fragment_nums
        for frame_id, index in enumerate(sorted(bboxes_dict.keys())):
            img_fname = bboxes_dict[index]["img_fname"]
            target_bbox = bboxes_dict[index]["target_bbox"]
            # all_bboxes = bboxes_dict[index]["all_bboxes"]
            bbox_score = bboxes_dict[index]["bbox_score"]
            kpts = bboxes_dict[index]["kpts"]
            oris = bboxes_dict[index]["ori"]
            vis_indicators = bboxes_dict[index]["vis_indicator"]
            vis_maps = bboxes_dict[index]["vis_map"]
            for j in range(len(bbox_score)):
                if bbox_score[j][-1] < 0.8:
                    continue
                bbox = bbox_score[j][:4]
                kpt = kpts[j]
                ori = oris[j]
                vis_indicator = vis_indicators[j]
                vis_map = vis_maps[j]
                if target_bbox[0]==0 or get_iou(target_bbox, bbox_score[j][:4]) < self.IOU_THRESHOLD:
                    info = [frame_id, img_fname, bbox, kpt, ori, vis_indicator, vis_map, 0]
                else:
                    info = [frame_id, img_fname, bbox, kpt, ori, vis_indicator, vis_map, 1]
                frag_dataset.append(info)
            # if (frame_id+1) % self.fragment_size == 0 or (frame_id+1) == len(bboxes_dict):
            #     self.rpf_datasets[frame_id+1] = Fragment_Part_Dataset(frag_dataset, self.cfg, self.img_scale, self.device)
            #     frag_dataset = []
            if (frame_id+1) % self.fragment_size == 0:
                self.rpf_datasets[frame_id+1] = Fragment_Part_Dataset(frag_dataset, self.cfg, self.img_scale, self.device)
                frag_dataset = []
            if len(self.rpf_datasets) == self.fragment_nums:
                break

class RPF_Global_Dataset(Dataset):
    def __init__(self, sequence, dataset_dir, cfg, height=256, width=128, device="cpu", fragment_nums=10):
        super(RPF_Global_Dataset, self).__init__()
        self.sequence = sequence
        self.dataset_dir = dataset_dir
        self.img_scale = (height, width)
        self.IOU_THRESHOLD = 0.8
        self.device = device
        self.cfg = cfg
        self.fragment_size = None
        self.fragment_nums = fragment_nums

        bboxes_file = osp.join(self.dataset_dir, self.sequence, "debug", "ALL_INFO.json")
        
        self.rpf_datasets = {}
        self._construct_dataset(bboxes_file)
    
    def _construct_dataset(self, bboxes_file):
        frag_dataset = []
        bboxes_file = open(bboxes_file, 'r')
        bboxes_dict = json.load(bboxes_file)
        self.fragment_size = len(bboxes_dict.keys()) // self.fragment_nums
        for frame_id, index in enumerate(sorted(bboxes_dict.keys())):
            img_fname = bboxes_dict[index]["img_fname"]
            target_bbox = bboxes_dict[index]["target_bbox"]
            bbox_score = bboxes_dict[index]["bbox_score"]
            kpts = bboxes_dict[index]["kpts"]
            for j in range(len(bbox_score)):
                if bbox_score[j][-1] < 0.8:
                    continue
                bbox = bbox_score[j][:4]
                kpt = kpts[j]
                if target_bbox[0]==0 or get_iou(target_bbox, bbox_score[j][:4]) < self.IOU_THRESHOLD:
                    info = [frame_id, img_fname, bbox, kpt, 0]
                else:
                    info = [frame_id, img_fname, bbox, kpt, 1]
                frag_dataset.append(info)
            # if (frame_id+1) % self.fragment_size == 0 or (frame_id+1) == len(bboxes_dict):
            #     self.rpf_datasets[frame_id+1] = Fragment_Global_Dataset(frag_dataset, self.cfg, self.img_scale, self.device)
            #     frag_dataset = []
            if (frame_id+1) % self.fragment_size == 0:
                self.rpf_datasets[frame_id+1] = Fragment_Part_Dataset(frag_dataset, self.cfg, self.img_scale, self.device)
                frag_dataset = []
            if len(self.rpf_datasets) == self.fragment_nums:
                break

class RPF_Parsing_Dataset(Dataset):
    def __init__(self, sequence, dataset_dir, cfg, height=256, width=128, device="cpu", fragment_nums=10, parsing_nums=2):
        super(RPF_Parsing_Dataset, self).__init__()
        self.sequence = sequence
        self.dataset_dir = dataset_dir
        self.img_scale = (height, width)
        self.IOU_THRESHOLD = 0.8
        self.device = device
        self.cfg = cfg
        self.fragment_size = None
        self.fragment_nums = fragment_nums

        if parsing_nums == 2:
            bboxes_file = osp.join(self.dataset_dir, self.sequence, "debug", "parsing_upper_lower_info.json")
        elif parsing_nums == 4:
            bboxes_file = osp.join(self.dataset_dir, self.sequence, "debug", "parsing4_upper_lower_info.json")
            
        self.rpf_datasets = {}
        self._construct_dataset(bboxes_file)
    
    def _construct_dataset(self, bboxes_file):
        frag_dataset = []
        bboxes_file = open(bboxes_file, 'r')
        bboxes_dict = json.load(bboxes_file)
        self.fragment_size = len(bboxes_dict.keys()) // self.fragment_nums
        for frame_id, index in enumerate(sorted(bboxes_dict.keys())):
            img_fname = bboxes_dict[index]["img_fname"]
            target_bbox = bboxes_dict[index]["target_bbox"]
            bbox_score = bboxes_dict[index]["bbox_score"]
            kpts = bboxes_dict[index]["kpts"]
            oris = bboxes_dict[index]["ori"]
            vis_indicators = bboxes_dict[index]["vis_indicator"]
            vis_maps = bboxes_dict[index]["vis_map"]
            for j in range(len(bbox_score)):
                if bbox_score[j][-1] < 0.8:
                    continue
                bbox = bbox_score[j][:4]
                kpt = kpts[j]
                ori = oris[j]
                vis_indicator = vis_indicators[j]
                vis_map = vis_maps[j]
                if target_bbox[0]==0 or get_iou(target_bbox, bbox_score[j][:4]) < self.IOU_THRESHOLD:
                    info = [frame_id, img_fname, bbox, kpt, ori, vis_indicator, vis_map, 0]
                else:
                    info = [frame_id, img_fname, bbox, kpt, ori, vis_indicator, vis_map, 1]
                frag_dataset.append(info)
            # if (frame_id+1) % self.fragment_size == 0 or (frame_id+1) == len(bboxes_dict):
            #     self.rpf_datasets[frame_id+1] = Fragment_Part_Dataset(frag_dataset, self.cfg, self.img_scale, self.device)
            #     frag_dataset = []
            if (frame_id+1) % self.fragment_size == 0:
                self.rpf_datasets[frame_id+1] = Fragment_Part_Dataset(frag_dataset, self.cfg, self.img_scale, self.device)
                frag_dataset = []
            if len(self.rpf_datasets) == self.fragment_nums:
                break

def get_iou(pred_box, gt_box):
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
    






