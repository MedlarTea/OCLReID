input_size = (256, 128)
identifier_params = dict(
    # type='GlobalNaiveIdentifier',
    type='GlobalNaiveIdentifier',
    params=dict(
        mem_size = 32,
        input_size = 128,
        id_switch_detection_thresh=0.35,  # original: 0.35 or 0.25 or 0.45
        reid_pos_confidence_thresh=0.6,  # original: 0.6
        reid_neg_confidence_thresh=0.3,
        reid_positive_count=5,
        # reid_positive_count=7,
        initial_training_num_samples=5,
        min_target_confidence=-1
    )
)
