import torch
import numpy as np
from .yhj_utils import part_random_retrieve, part_class_balance_random_retrieve, part_random_retrieve_features, part_class_balance_random_features_retrieve
"""
retrieve samples
"""

class Part_lt_balance_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.mem_size = params.mem_size * params.lt_rate
        self.vis_map_size = params.vis_map_size
        self.vis_map_nums = self.vis_map_size[0]
        self.vis_map_res = self.vis_map_size[1:]
        self.part_nums = self.vis_map_nums+1 if not params.use_ori else 2*(self.vis_map_nums+1)

        self.full_indexes = [set([i for i in range(self.mem_size)]) for _ in range(self.part_nums)]

        self.num_retrieve = params.batch_size
        self.class_balance = params.class_balance

    def retrieve(self, buffer, excl_indices=None, return_indices=False):
        if not self.class_balance:
            return part_random_retrieve(buffer, self, self.num_retrieve)
        else:
            return part_class_balance_random_retrieve(buffer, self, self.num_retrieve)
    
    def retrieve_features(self, buffer, excl_indices=None, return_indices=False):
        if not self.class_balance:
            return part_random_retrieve_features(buffer, self, self.num_retrieve)
        else:
            return part_class_balance_random_features_retrieve(buffer, self, self.num_retrieve)

        
        

        
