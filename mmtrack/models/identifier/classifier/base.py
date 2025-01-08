import os
import numpy as np
from abc import ABCMeta, abstractmethod


"""
Classifier, is used to identify the target person
1) Supervised learning
2) Online machine learning
"""


class BaseClassifier():
    def __init__(self):
        super(BaseClassifier, self).__init__()
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict_single(self, feature):
        pass

    @abstractmethod
    def predict_batch(self, features):
        pass

    # @abstractmethod
    # def init_classifier(self):
    #     pass
