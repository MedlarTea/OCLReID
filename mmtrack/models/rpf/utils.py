import torch
import numpy as np
from mmtrack.models.identifier.utils.utils import maybe_cuda



def process_kpts(poses, input_height, input_width):
    """Suppose only one bbox and pose

    """
    track_kpts = torch.zeros((len(poses), 13, 3))
    processed_kpts = torch.zeros((len(poses), 34))
    for i in range(len(poses)):
        kpts = poses[i]['keypoints']
        scores = poses[i]['kp_score']
        bbox = poses[i]['bbox']
        # tl_x, tl_y, br_x, br_y, bbox_conf = bbox
        tl_x, tl_y, br_x, br_y = bbox
        ### draw keypoints ###
        # pts = torch.cat((kpts, scores), axis=1).tolist()
        # pts = np.array(pts)
        # pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
        # data_numpy = draw_single(data_numpy, pts)

        kpts[:,0] = torch.clamp(kpts[:,0], min=tl_x, max=br_x)
        kpts[:,1] = torch.clamp(kpts[:,1], min=tl_y, max=br_y)
        # in the resized image patch coordinate
        # print(self.image_size)
        kpts[:,0] = (kpts[:,0]-tl_x)/(br_x-tl_x)*input_width
        kpts[:,1] = (kpts[:,1]-tl_y)/(br_y-tl_y)*input_height
        # print(kpts)
        # track_kpts.append(torch.cat((kpts, ps['kp_score']), axis=1).tolist())
        track_kpts[i, :, :2] = kpts
        track_kpts[i, :, 2] = scores.squeeze()
    
        coco_kpts = torch.zeros((17, 2))
        # print(track_kpts[0, :2].shape)
        # print(track_kpts[0, 2].shape)
        coco_kpts[0, :] = track_kpts[i, 0, :2] * track_kpts[i, 0, 2]  # Nose
        for j in range(5, 17):
            coco_kpts[j, :] = track_kpts[i, j-4, :2] * track_kpts[i, j-4, 2]
        # norm and rescale to [256,]
        coco_kpts = coco_kpts.flatten()
        processed_kpts[i, :] = coco_kpts

    return track_kpts, maybe_cuda(processed_kpts)

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


