from .buffer_utils import random_retrieve
import torch
import numpy as np
class Global_st_balance_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.full_indexes = set([i for i in range(params.mem_size)])

    def retrieve(self, buffer, excl_indices=None, return_indices=False):
        indices = self.full_indexes.difference(buffer.buffer_tracker.remaining_indexes)
        indices = torch.Tensor(list(indices)).long()

        x = buffer.buffer_img[indices]
        y = buffer.buffer_label[indices]

        if return_indices:
            return x, y, indices
        else:
            return x, y
    
    def retrieve_features(self, buffer, excl_indices=None, return_indices=False):
        indices = self.full_indexes.difference(buffer.buffer_tracker.remaining_indexes)
        indices = torch.Tensor(list(indices)).long()
    
        x = buffer.buffer_feature[indices, :self.params.deep_feature_dim]
        y = buffer.buffer_label[indices]

        if return_indices:
            return x, y, indices
        else:
            return x, y