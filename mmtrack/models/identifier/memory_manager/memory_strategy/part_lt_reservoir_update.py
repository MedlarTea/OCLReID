import torch
from .part_buffer import PartBuffer
"""
Update with the latest samples based on the FILO rule
"""

class Part_lt_reservoir_update(object):
    def __init__(self, params):
        super().__init__()
        self.deep_feature_dim = params.deep_feature_dim
        self.joints_feature_dim = params.joints_feature_dim

        self.params = params
        self.vis_map_size = params.vis_map_size
        self.vis_map_nums = self.vis_map_size[0]
        self.vis_map_res = self.vis_map_size[1:]
        self.part_nums = self.vis_map_nums+1 if not params.use_ori else 2*(self.vis_map_nums+1)

        self.mem_size = params.mem_size * params.lt_rate
        self.current_neg = [0 for i in range(self.part_nums)]
        self.neg_size = self.mem_size // 2
        self.current_pos = [self.mem_size // 2 for i in range(self.part_nums)]
        self.pos_size = self.mem_size // 2

    def update(self, buffer:PartBuffer, tracklet, y, **kwargs):
        """Update the st buffer including image, label and part-global features
        0-15: neg, 16-31: pos
        """
        x = tracklet.image_patch  # (3,H,W)
        deep_feature = tracklet.deep_feature  # (5,512)
        vis_indicator = tracklet.visibility_indicator  # (10), front-parts, back-parts
        visibility_map = tracklet.visibility_map  # (4,4,8), front-parts, back-parts
        if self.params.use_ori:
            ori = tracklet.binary_ori  # 0 for not seeing face (front), 1 for seeing face (Back)
            start_idx = 0 if ori == 0 else self.part_nums//2
            seg_idx = self.part_nums//2
        else:
            start_idx = 0
            seg_idx = self.part_nums

        self.part_insert_index = [-1 for i in range(self.part_nums)]
        img_insert_index = -1
        # add whatever still fits in the buffer
        y_int = y.item()
        for _, part_idx in enumerate(range(start_idx, start_idx + seg_idx)):
            if vis_indicator[part_idx] == 1:
                if y_int == 0:
                    insert_index = buffer.n_seen_so_far[part_idx][y_int] % self.neg_size
                elif y_int == 1:
                    insert_index = (buffer.n_seen_so_far[part_idx][y_int] % self.pos_size) + self.pos_size
                self.part_insert_index[part_idx] = insert_index
            if part_idx == start_idx + seg_idx - 1:
                if buffer.n_seen_so_far[part_idx][y_int] >= self.mem_size:
                    img_insert_index = self.reservoir_update(buffer, x, y, part_idx)
                    self.part_insert_index[part_idx] = img_insert_index
                else:
                    img_insert_index = insert_index
            if vis_indicator[part_idx] == 1:
                buffer.n_seen_so_far[part_idx][y_int] += 1

        if buffer.params.buffer_tracker:
            buffer.buffer_tracker.update_cache(buffer, y, self.part_insert_index)

        # Add to part buffer feature
        for i, part_idx in enumerate(range(start_idx, start_idx + seg_idx)):
            if vis_indicator[part_idx] == 1:
                part_buffer_feature = getattr(buffer, "buffer_feature_{}".format(part_idx))
                part_buffer_label = getattr(buffer, "buffer_label_{}".format(part_idx))
                part_buffer_feature[self.part_insert_index[part_idx], :self.deep_feature_dim] = deep_feature[i,:]
                part_buffer_label[self.part_insert_index[part_idx]] = y[:1]
            # global feature must be observed
            if part_idx == start_idx + seg_idx - 1:
                buffer_img = getattr(buffer, "buffer_img_{}".format(part_idx))
                buffer_vis_map = getattr(buffer, "buffer_vis_map_{}".format(part_idx))
                buffer_vis_indicator = getattr(buffer, "buffer_vis_indicator_{}".format(part_idx))
                buffer_img[img_insert_index] = x[:1]
                buffer_vis_map[img_insert_index] = visibility_map
                buffer_vis_indicator[img_insert_index] = vis_indicator

        return insert_index


    def reservoir_update(self, buffer, x, y, part_idx):
        """update images including front and back
        
        """
        y_int = y.item()
        range_min = 0 if y_int == 0 else self.mem_size // 2
        n_seen_so_far = buffer.n_seen_so_far[part_idx][0] - 1 if y_int == 0 else self.mem_size // 2 + buffer.n_seen_so_far[part_idx][1] - 1
        indices = torch.FloatTensor(1).to(x.device).uniform_(range_min, n_seen_so_far).long()
        valid_indices = (indices < range_min + self.mem_size // 2).long()
        idx_new_data = valid_indices.nonzero().squeeze(-1)  # data idxs that can be inserted
        idx_buffer = indices[idx_new_data]
        # if y_int == 0:
        #     buffer.n_seen_so_far[part_idx][0] += 1
        # else:
        #     buffer.n_seen_so_far[part_idx][1] += 1
        # print("------------size of{}------------".format(n_seen_so_far))
        if idx_buffer.numel() == 0:
            return -1
        assert idx_buffer.max() < range_min + self.mem_size // 2
        assert idx_buffer.max() < range_min + self.mem_size // 2

        assert idx_new_data.max() < 1
        assert idx_new_data.max() < 1

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        # replace_y = y[list(idx_map.values())]
        # if buffer.params.buffer_tracker:
        #     buffer.buffer_tracker.update_cache(buffer.buffer_label, replace_y, list(idx_map.keys()))
        # perform overwrite op
        # buffer.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        # buffer.buffer_label[list(idx_map.keys())] = replace_y
        # buffer.buffer_feature[list(idx_map.keys()), :] = deep_feature
        return list(idx_map.keys())[0]