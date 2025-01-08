input_size = (256, 192)
identifier_params = dict(
    type='OCLREIDIdentifier',
    params = dict(
        agent="GlobalResNetClassifier",
        seed=123,
        deep_feature_dim=128,
        height = 256,
        width = 128,
        norm_mean = [0.485, 0.456, 0.406],
        norm_std = [0.229, 0.224, 0.225],
        rr_alpha=1.0,  # original 0.1
        lt_rate=8,
        mem_size=16,
        id_switch_detection_thresh=0.50,
        reid_pos_confidence_thresh=0.80,
        reid_neg_confidence_thresh=0.3,
        reid_positive_count=7,
        initial_training_num_samples=5,
        min_target_confidence=-1,
        min_bbox_confidence=0.85,
        cuda=True,
    )
)