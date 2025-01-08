# model settings
_base_ = './yolox_s_8x8.py'
img_scale = (640, 640)  # height, width

# model settings
model = dict(
    detector=dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256)))
