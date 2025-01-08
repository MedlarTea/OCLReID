image_shape = (512,384,3)
device = 'cuda:0'
mmtracking_dir = None
input = None
output = None
output_json = None
gt_bbox_file = None

### change your desired configs ###
rpf_config = "configs/rpf/identifier/part_weighted_identifier_r18.py"
identifier_config = "configs/rpf/ocl_configs/part_lt_reservoir_simplified.py"

### change your desired configs ###

show_result = False

# for result
visdom_info=dict(
        server="127.0.0.1",
        port=8097,
        use_visdom=False)

save_es_result=False  # whether to save "logs" and "files used for evaluation"
save_vis_result=False  # whether to save resulting video and images
description = "robot_exp"



debug=True  # whether to save the itermediate results (used inside the core codes)
select_target_threshold = 0.7  # iou
seed = 123
