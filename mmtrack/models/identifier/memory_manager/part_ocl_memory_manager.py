from .base import BaseMemory
import torch
import numpy as np
from utils import maybe_cuda
# import name_match
from .memory_strategy.part_buffer import PartBuffer
from .memory_strategy.yhj_utils import is_informative_st
# from queue import Queue
import time
from ..utils.utils import MyAverageMeter

# TODO: better computer memory management by using vector-like container
class PartOCLMemoryManager(torch.nn.Module):
    def __init__(self, clf_model, reid_model, params=None):
        super().__init__()
        self.params = params
        self.model = reid_model
        self.clf_model = clf_model
        self.lt_size = params.mem_size * params.lt_rate
        self.st_size = params.mem_size

        input_size = (3, params.input_size[0], params.input_size[1])

        ### For keyframe selection ###
        # self.sliding_window_size = params.sliding_window_size

        self.batch_indices = dict(
            lt_pos = [],
            lt_neg = [],
            st_pos = [],
            st_neg = []
        )

        ### For keyframe selection ###
        self.factor = {
            1: dict(
            conf_thr=params.init_conf_thr,
            appearance=0.2,
            pose=3
        ),
            0: dict(
            conf_thr=params.init_conf_thr,
            appearance=0.2,
            pose=3
        )}

        self.memory = dict(st_set = PartBuffer(self, buffer_size=self.st_size, input_size=input_size, update=params.st_update_method, retrieve=params.st_retrieve_method, params=params))
        if self.params.lt_update_method is not None:
            self.memory["lt_set"] = PartBuffer(self, buffer_size=self.lt_size, input_size=input_size, update=params.lt_update_method, retrieve=params.lt_retrieve_method, params=params)

        self.clu_counter = 0
        self.cost = 0

        self.delta_loss_thr = self.params.delta_loss_thr
        self.increment_loss = MyAverageMeter()
        self.update_time = MyAverageMeter()

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
        t_update = time.time()
        if target_id == -1:
            return False
        for track_id in tracklets.keys():
            # print("target_id: ", target_id)
            if track_id == target_id:
                self.memory["st_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([1]).long()))
                ### With key frame selection ###
                if self.params.lt_update_method is not None and self._kf_check():
                ### With key frame selection ###
                # if self.params.lt_update_method is not None:
                    self.memory["lt_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([1]).long()))
            else:
                self.memory["st_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([0]).long()))
                if self.params.lt_update_method is not None:
                    self.memory["lt_set"].update(tracklets[track_id], maybe_cuda(torch.Tensor([0]).long()))
        self.update_time.update((time.time() - t_update)*1000)
        # print("[Memory Update] Current {:.3f}\tAverage {:.3f}".format(self.update_time.val, self.update_time.avg))
        return self.increment_loss
    
    def _kf_check(self):
        """Keyframe selection based on the change of the loss
        """
        if not hasattr(self.clf_model, "freeze_model") :
            return False
        self.clf_model.freeze_model.eval()
        last_st_loss = self.clf_model.last_st_loss  # item
        # print("================================")
        # print("Incremental Loss curr/avg/count: {:.3f}/{:.3f}/{}".format(self.increment_loss.val, self.increment_loss.avg, self.increment_loss.count))
        # print("================================")
        with torch.no_grad():
            # t=time.time()
            ### More retrieve, so change random sequence ###
            st_x, st_y, vis_map, vis_indicator  = self.retrieve_st()
            # print("TIME: {:.3f}".format((time.time()-t)*1000))
            # t=time.time()
            try:
                _, loss, loss_dict = self.clf_model.freeze_model.forward_train(st_x, st_y, vis_map, vis_indicator, False, True)
            except:
                return False
            # print("TIME: {:.3f}".format((time.time()-t)*1000))

            current_st_loss = loss.item()
        # self.increment_loss.update(current_st_loss - last_st_loss)
        if current_st_loss - last_st_loss > self.delta_loss_thr:
            self.increment_loss.update(current_st_loss - last_st_loss)
            return True
        return False


    
    def retrieve_st(self):
        """Randomly generate indexes to be sampled
        Input:
            batch_size (int)
            random_seed (int)
        Output:
            st_x: images with shape of (B, 3, H, W)
            st_y: labels with shape of (B)
            vis_indicator: visibility indicator with shape of (B, 10) -- front back 4 parts and 1 global
        """
        st_x, st_y, vis_indicator, vis_map = self.memory["st_set"].retrieve()
        return st_x, st_y, vis_indicator, vis_map
    
    def retrieve_lt(self, st_x=None, st_y=None):
        """Randomly generate indexes to be sampled
        Input:
            batch_size (int)
            random_seed (int)
        Output:
            lt_x: for training with shape of (B, 3, H, W)
            lt_y: labels with shape of (B)
            vis_indicator: for visibility indication with shape of (B, 10)---front back 4 parts and 1 global
            vis_map: visibility map with shape of (B,4,4,8)
        """
        lt_x, lt_y, vis_indicator, vis_map = self.memory["lt_set"].retrieve()

        return lt_x, lt_y, vis_indicator, vis_map
    
    def retrieve_st_features(self):
        """
        Output:
            st_x: part-global features with shape of (part_nums, x, feature_dim)
            st_y: labels with shape of (part_nums, x)
        """
        st_x, st_y = self.memory["st_set"].retrieve_features()
        return st_x, st_y
    
    def retrieve_lt_features(self):
        """
        Output:
            lt_x: part-global features with shape of (part_nums, x, feature_dim)
            lt_y: labels with shape of (part_nums, x)
        """
        lt_x, lt_y = self.memory["lt_set"].retrieve_features()
        return lt_x, lt_y
    

        