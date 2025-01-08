import os
import cv2
import torch

from mmtrack.models.pose.SPPE.src.main_fast_inference import InferenNet_fast, InferenNet_fastRes50
from mmtrack.models.pose.SPPE.src.utils.img import crop_dets
from mmtrack.models.pose.pPose_nms import pose_nms
from mmtrack.models.pose.SPPE.src.utils.eval import getPrediction


class SPPE_FastPose(object):
    def __init__(self,
                 backbone,
                 input_height=320,
                 input_width=256,
                 device='cuda'):
        assert backbone in ['resnet50', 'resnet101'], '{} backbone is not support yet!'.format(backbone)

        self.inp_h = input_height
        self.inp_w = input_width
        self.device = device

        if backbone == 'resnet101':
            self.model = InferenNet_fast().to(device)
        else:
            self.model = InferenNet_fastRes50().to(device)
        self.model.eval()

    def predict(self, image, img_metas, bboxs, bboxs_scores, bbox_ids=None, rescale=False):
        """
        rescale: True for the bbox of original size
        """
        h, w, _ = img_metas[0]['img_shape']
        h_ori, w_ori, _ = img_metas[0]['ori_shape']
        if rescale:
            bboxs[:,:4] *= torch.tensor(img_metas[0]['scale_factor']).to(bboxs.device)
        bboxs[:,0::2] = torch.clamp(bboxs[:,0::2], min=0, max=w)
        bboxs[:,1::2] = torch.clamp(bboxs[:,1::2], min=0, max=h)

        inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
        pose_hm = self.model(inps.to(self.device)).cpu().data

        # Cut eyes and ears.
        pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)

        # xy_img: [batch_size, 13, 2]
        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
        if rescale:
            bboxs[:,:4] /= torch.tensor(img_metas[0]['scale_factor']).to(bboxs.device)
            xy_img[:,:,:] /= torch.tensor(img_metas[0]['scale_factor'][:2]).to(bboxs.device)
        bboxs[:,0::2] = torch.clamp(bboxs[:,0::2], min=0, max=w_ori)
        bboxs[:,1::2] = torch.clamp(bboxs[:,1::2], min=0, max=h_ori)
        result = pose_nms(bboxs, bboxs_scores, xy_img, scores, bbox_ids=bbox_ids)
        
        return result