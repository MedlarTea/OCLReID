
class HyperParams():
    def __init__(self, params) -> None:
        self.id_switch_detection_thresh = 0.35
        # self.id_switch_detection_thresh = 0.0
        self.reid_pos_confidence_thresh = 0.6
        self.reid_neg_confidence_thresh = 0.30
        self.reid_positive_count = 5
        self.initial_training_num_samples = 5
        self.min_target_confidence = -1
    