image_shape = (512,384,3)
device = 'cuda:0'
mmtracking_dir = None
input = None
output = None
output_json = None
gt_bbox_file = None

rpf_config = "configs/rpf/ocl_rpf/naive_rpf_yolox_l_r18.py"
identifier_config = "configs/rpf/identifier/oclreid_identifier_r18.py"

show_result = False

# for result
visdom_info=dict(
        server="127.0.0.1",
        port=8097,
        use_visdom=False)

save_es_result=False  # whether to save "logs" and "files used for evaluation"
save_vis_result=False  # whether to save resulting video and images
description = "oclreid_resnet18"



debug=False  # whether to save the itermediate results (used inside the core codes)
select_target_threshold = 0.4  # iou
seed = 123
