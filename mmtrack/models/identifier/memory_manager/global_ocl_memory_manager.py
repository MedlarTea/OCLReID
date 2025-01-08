from .base import BaseMemory
import torch
import numpy as np
from utils import maybe_cuda
# import name_match
from .memory_strategy.buffer import Buffer
from .memory_strategy.yhj_utils import is_informative_st
# from queue import Queue



# TODO: better computer memory management by using vector-like container
class GlobalOCLMemoryManager(torch.nn.Module):
    def __init__(self, model, params=None):
        super().__init__()
        self.params = params
        self.model = model
        self.lt_size = params.mem_size * params.lt_rate
        self.st_size = params.mem_size

        input_size = (3, params.input_size[0], params.input_size[1])

        # self.sliding_window_size = params.sliding_window_size

        self.batch_indices = dict(
            lt_pos = [],
            lt_neg = [],
            st_pos = [],
            st_neg = []
        )

        self.memory = dict(st_set = Buffer(self, buffer_size=self.st_size, input_size=input_size, update=params.st_update_method, retrieve=params.st_retrieve_method, params=params))
        if self.params.lt_update_method is not None:
            self.memory["lt_set"] = Buffer(self, buffer_size=self.lt_size, input_size=input_size, update=params.lt_update_method, retrieve=params.lt_retrieve_method, params=params)

        self.clu_counter = 0
        self.cost = 0
    
    def update(self, tracklets:dict, target_id:int):
        """Update the memory including short-term and long-term memory
        Short-term memory only contains the latest samples
        Short-term memory has an updating strategy
        Input:
            tracklets
            target_id
        Output:
            None
        """
        if target_id == -1:
            return False
        for track_id in tracklets.keys():
            # print("target_id: ", target_id)
            if track_id == target_id:
                self.memory["st_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([1]).long()))
                if self.params.lt_update_method is not None:
                    self.memory["lt_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([1]).long()))
            else:
                self.memory["st_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([0]).long()))
                if self.params.lt_update_method is not None:
                    self.memory["lt_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([0]).long()))
        return True
    
    def retrieve_st(self):
        """Randomly generate indexes to be sampled
        Input:
            batch_size (int)
            random_seed (int)
        Output:
            valid random indexes
        """
        st_x, st_y = self.memory["st_set"].retrieve()
        # self.batch_indices["st_pos"] = self.get_batch_indices(batch_size, st_pos_indices)

    #     st_x = torch.cat([st_pos_x, st_neg_x])
    #     st_y = torch.cat([st_pos_y, st_neg_y])
        return st_x, st_y
    
    def retrieve_lt(self, st_x=None, st_y=None):
        """Randomly generate indexes to be sampled
        Input:
            batch_size (int)
            random_seed (int)
        Output:
            valid random indexes
        """
        lt_x, lt_y = self.memory["lt_set"].retrieve(x=st_x, y=st_y)

        return lt_x, lt_y
    
    def retrieve_st_features(self):
        st_x, st_y = self.memory["st_set"].retrieve_features()
        return st_x, st_y
    
    def retrieve_lt_features(self):
        lt_x, lt_y = self.memory["lt_set"].retrieve_features()
        return lt_x, lt_y
    

        