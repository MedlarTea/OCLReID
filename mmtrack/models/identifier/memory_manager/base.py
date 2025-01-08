import os
import numpy as np
from abc import ABCMeta, abstractmethod
import torch


class BaseMemory(metaclass=ABCMeta):
    def __init__(self):
        self.trainSet = []
        self.labels = []
    
    @abstractmethod
    def update(self, tracklets, target_id):
        pass

    
    @abstractmethod
    def retrieve(self):
        pass

    
    def maybe_cuda(self, what, use_cuda=True, **kw):
        """
        Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
            Args:
                what (object): any object to move to eventually gpu
                use_cuda (bool): if we want to use gpu or cpu.
            Returns
                object: the same object but eventually moved to gpu.
        """

        if use_cuda is not False and torch.cuda.is_available():
            what = what.cuda()
        return what

    





