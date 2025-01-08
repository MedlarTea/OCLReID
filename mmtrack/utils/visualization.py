import cv2
import numpy as np

class Drawer():
    def __init__(self) -> None:
        pass
    
    def draw_es_box(self, image, estimated_bbox):
        return cv2.rectangle(image, (int(estimated_bbox[0]), int(estimated_bbox[1])), (int(estimated_bbox[2]), int(estimated_bbox[3])), (0,0,255), 3)
    
    def draw_all_box(self, image, bbox_dict):
        for id in bbox_dict.keys():
            box = bbox_dict[id]
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 3)
        return image
    
    def write_text(self, image, occlusion_txt, distance):
        image = cv2.putText(image, occlusion_txt, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        image = cv2.putText(image, "{:.2f}".format(distance), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        return image
    
    def draw_gt_triangle(self, image, gt_bbox):
        center_uv = [int((gt_bbox[0]+gt_bbox[2])/2), int(gt_bbox[3])]
        bottom_left_uv = [center_uv[0]-20, center_uv[1]+20]
        bottom_right_uv = [center_uv[0]+20, center_uv[1]+20]
        if center_uv[1] >= image.shape[0]:
            center_uv[1] = image.shape[0]-20
            bottom_left_uv[1] = image.shape[0]-1
            bottom_right_uv[1] = image.shape[0]-1
        return cv2.polylines(image, [np.array([center_uv, bottom_left_uv, bottom_right_uv])], True, (0, 255, 0), 3)
    
    