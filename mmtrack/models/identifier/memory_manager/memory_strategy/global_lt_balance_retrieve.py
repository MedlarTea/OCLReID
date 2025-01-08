from .yhj_utils import random_retrieve, class_balance_random_retrieve, class_balance_random_retrieve_features, random_retrieve_features



"""Retrieve based on the buffer_tracker!

"""
class Global_lt_balance_retrieve(object):
    def __init__(self, params):
        super().__init__()
        self.num_retrieve = params.batch_size
        self.class_balance = params.class_balance

    def retrieve(self, buffer, **kwargs):
        if self.class_balance:
            return class_balance_random_retrieve(buffer, self.num_retrieve)
        else:
            return random_retrieve(buffer, self.num_retrieve)
    
    def retrieve_features(self, buffer, **kwargs):
        if self.class_balance:
            return class_balance_random_retrieve_features(buffer, self.num_retrieve)
        else:
            return random_retrieve_features(buffer, self.num_retrieve)