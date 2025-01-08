import torch
# from .buffer import Buffer
"""
Update with the latest samples based on the FILO rule
"""

class Global_lt_reservoir_update(object):
    def __init__(self, params):
        super().__init__()
        self.deep_feature_dim = params.deep_feature_dim
        self.joints_feature_dim = params.joints_feature_dim

        self.current_neg = 0
        self.neg_size = params.mem_size*params.lt_rate // 2

        self.current_pos = params.mem_size*params.lt_rate // 2
        self.pos_size = params.mem_size*params.lt_rate // 2
        self.mem_size = params.mem_size*params.lt_rate

    def update(self, buffer, tracklet, y, **kwargs):
        """update with one sample input
        0-15: neg, 16-31: pos
        """
        x = tracklet.image_patch
        deep_feature = tracklet.deep_feature
        # joints_feature = tracklet.joints_feature
        # add whatever still fits in the buffer
        y_int = y.item()
        if y_int == 0:
            ### Full memory, so need to replace ###
            if buffer.n_neg_seen_so_far >= self.mem_size // 2:
                return self.reservoir_update(buffer, x, y, deep_feature , is_pos=False)
            insert_index = buffer.n_neg_seen_so_far % self.neg_size
            buffer.n_neg_seen_so_far += 1
        elif y_int == 1:
            ### Full memory, so need to replace ###
            if buffer.n_pos_seen_so_far >= self.mem_size // 2:
                return self.reservoir_update(buffer, x, y, deep_feature, is_pos=True)
            insert_index = (buffer.n_pos_seen_so_far % self.pos_size) + self.pos_size
            buffer.n_pos_seen_so_far += 1
        if buffer.params.buffer_tracker:
            buffer.buffer_tracker.update_cache(buffer.buffer_label, y, [insert_index])
        
        buffer.buffer_img[insert_index] = x[:1]
        buffer.buffer_label[insert_index] = y[:1]
        buffer.buffer_feature[insert_index, :] = deep_feature

        return insert_index

    def reservoir_update(self, buffer, x, y, deep_feature, is_pos):
        range_min = 0 if is_pos == False else self.mem_size // 2
        n_seen_so_far = buffer.n_neg_seen_so_far if is_pos == False else self.mem_size // 2 + buffer.n_pos_seen_so_far
        indices = torch.FloatTensor(1).to(x.device).uniform_(range_min, n_seen_so_far).long()
        valid_indices = (indices < range_min + self.mem_size // 2).long()
        idx_new_data = valid_indices.nonzero().squeeze(-1)  # data idxs that can be inserted
        idx_buffer = indices[idx_new_data]
        # if is_pos == False:
        #     buffer.n_neg_seen_so_far += 1
        # else:
        #     buffer.n_pos_seen_so_far += 1
        # print("------------size of{}------------".format(n_seen_so_far))
        if idx_buffer.numel() == 0:
            return []
        assert idx_buffer.max() < range_min + self.mem_size // 2
        assert idx_buffer.max() < range_min + self.mem_size // 2

        assert idx_new_data.max() < 1
        assert idx_new_data.max() < 1

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        replace_y = y[list(idx_map.values())]
        if buffer.params.buffer_tracker:
            buffer.buffer_tracker.update_cache(buffer.buffer_label, replace_y, list(idx_map.keys()))
        # perform overwrite op
        buffer.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.buffer_label[list(idx_map.keys())] = replace_y
        buffer.buffer_feature[list(idx_map.keys()), :] = deep_feature
        return list(idx_map.keys())