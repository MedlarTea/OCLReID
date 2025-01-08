import torch
# from .buffer import Buffer
"""
Update with the latest samples based on the FILO rule
"""

class Global_st_balance_update(object):
    def __init__(self, params):
        super().__init__()
        self.deep_feature_dim = params.deep_feature_dim
        self.joints_feature_dim = params.joints_feature_dim

        self.current_neg = 0
        self.neg_size = params.mem_size // 2
        self.current_pos = params.mem_size // 2
        self.pos_size = params.mem_size // 2
        self.mem_size = params.mem_size

    def update(self, buffer, tracklet, y, **kwargs):
        """
        0-15: neg, 16-31: pos
        """
        x = tracklet.image_patch
        deep_feature = tracklet.deep_feature
        # joints_feature = tracklet.joints_feature
        # add whatever still fits in the buffer
        y_int = y.item()
        if y_int == 0:
            insert_index = self.current_neg % self.neg_size
            self.current_neg += 1
        elif y_int == 1:
            insert_index = (self.current_pos % self.pos_size) + self.pos_size
            self.current_pos += 1
        if buffer.params.buffer_tracker:
            buffer.buffer_tracker.update_cache(buffer.buffer_label, y, [insert_index])
        
        buffer.buffer_img[insert_index] = x[:1]
        buffer.buffer_label[insert_index] = y[:1]
        buffer.buffer_feature[insert_index, :self.deep_feature_dim] = deep_feature
        # buffer.buffer_feature[insert_index, self.deep_feature_dim:] = joints_feature
        buffer.n_seen_so_far += 1

        return insert_index