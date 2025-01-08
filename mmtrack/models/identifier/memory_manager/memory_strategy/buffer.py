# from utils.setup_elements import input_size_match
from .. import name_match #import update_methods, retrieve_methods
from utils import maybe_cuda
import torch
from .buffer_utils import BufferClassTracker
# from utils.setup_elements import n_classes

class Buffer(torch.nn.Module):
    def __init__(self, memory_manager, buffer_size, input_size, update, retrieve, params=None):
        super().__init__()
        self.params = params
        self.memory_manager = memory_manager
        self.model = memory_manager.model
        self.current_index = 0
        self.n_seen_so_far = 0
        self.n_pos_seen_so_far = 0
        self.n_neg_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"

        if self.params.st_feature == "deep":
            feature_dim = params.deep_feature_dim
        elif self.params.st_feature == "joint":
            feature_dim = params.joints_feature_dim
        elif self.params.st_feature == "all":
            feature_dim = params.deep_feature_dim + params.joints_feature_dim

        use_feature = hasattr(params, 'deep_feature_dim') and hasattr(params, 'joints_feature_dim')


        # define buffer
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        if use_feature:
            buffer_feature = maybe_cuda(torch.FloatTensor(buffer_size, feature_dim).fill_(0))  # deep feature, bbox feature and joints feature
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        # buffer_label.index_fill_(0, maybe_cuda(torch.LongTensor(list(range(0, buffer_size)))), 1)  # half pos--1, half neg--0

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        if use_feature:
            self.register_buffer('buffer_feature', buffer_feature)
        self.register_buffer('buffer_label', buffer_label)

        # define update and retrieve method
        self.update_method = name_match.update_methods[update](params)
        self.retrieve_method = name_match.retrieve_methods[retrieve](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(2, buffer_size, self.device)  # we only have positive and negative samples

    def update(self, tracklet, y, **kwargs):
        """
        update with one sample
        """
        # print(type(self.update_method))
        return self.update_method.update(buffer=self, tracklet=tracklet, y=y, **kwargs)


    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)
    
    def retrieve_features(self, **kwargs):
        return self.retrieve_method.retrieve_features(buffer=self, **kwargs)
    