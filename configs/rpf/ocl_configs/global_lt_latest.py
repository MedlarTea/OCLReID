input_size = (256, 128)
identifier_params = dict(
    type='GlobalIdentifier',
    params = dict(
        agent="GlobalOCLClassifier",
        st_update_method="global_st_balance",
        st_retrieve_method="global_st_balance",
        lt_update_method="global_lt_balance",
        lt_retrieve_method="global_lt_balance",
        class_balance=True,
        not_freeze="conv4",
        st_feature="deep",  # deep, joint, all
        
        mem_size=32,
        lt_rate=8,
        rr_alpha=1.0,  # original 0.1
        seed=123,
        init_conf_thr=0.15,  # original 0.1
        sliding_window_size=16,
        deep_feature_dim=128,
        joints_feature_dim=28,
        input_size=input_size,
        
        epochs=1,
        batch_size=32,
        optimizer="SGD",
        backbone="resnet18",
        learning_rate=0.1,
        weight_decay=0,
        cuda=True,
        buffer_tracker=True,
        id_switch_detection_thresh=0.35,  # original: 0.35 or 0.25 or 0.45
        reid_pos_confidence_thresh=0.7,  # original: 0.6
        reid_neg_confidence_thresh=0.3,
        reid_positive_count=5,
        initial_training_num_samples=5,
        min_target_confidence=-1
    )
)
