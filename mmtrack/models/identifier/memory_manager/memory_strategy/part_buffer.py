# from utils.setup_elements import input_size_match
from .. import name_match #import update_methods, retrieve_methods
from utils import maybe_cuda
import torch
from .yhj_utils import BufferPartClassTracker
# from utils.setup_elements import n_classes

class PartBuffer(torch.nn.Module):
    def __init__(self, memory_manager, buffer_size, input_size, update, retrieve, params=None):
        super().__init__()
        self.params = params
        vis_map_size = params.vis_map_size
        self.memory_manager = memory_manager
        self.model = memory_manager.model
        self.current_index = 0
        
        self.device = "cuda" if self.params.cuda else "cpu"

        # use_feature = hasattr(params, 'deep_feature_dim') and hasattr(params, 'joints_feature_dim')
        use_feature = hasattr(params, 'deep_feature_dim')

        if use_feature:
            # feature_dim = params.deep_feature_dim + params.joints_feature_dim
            feature_dim = params.deep_feature_dim

        self.vis_map_size = params.vis_map_size
        self.vis_map_nums = self.vis_map_size[0]
        self.vis_map_res = self.vis_map_size[1:]
        self.part_nums = self.vis_map_nums+1 if not params.use_ori else 2*(self.vis_map_nums+1)
        self.n_seen_so_far = [{0:0, 1:0} for _ in range(self.part_nums)]


        # define buffer
        buffer_imgs = {}
        buffer_vis_maps = {}
        buffer_vis_indicators = {}
        # [4, 9]
        for i in range(self.vis_map_nums, self.part_nums, self.part_nums//2):
            buffer_imgs[i] = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
            buffer_vis_maps[i] = maybe_cuda(torch.LongTensor(buffer_size, *vis_map_size).fill_(0).bool())
            buffer_vis_indicators[i] = maybe_cuda(torch.LongTensor(buffer_size, self.part_nums).fill_(0).bool())
        # front -> back; head -> feet
        if use_feature:
            # buffer_vis_map = maybe_cuda(torch.LongTensor(buffer_size, *vis_map_size).fill_(0).bool())
            # buffer_vis_indicator = maybe_cuda(torch.LongTensor(buffer_size, self.part_nums).fill_(0).bool())
            buffer_features = []
            buffer_labels = []
            for i in range(self.part_nums):
                buffer_features.append(maybe_cuda(torch.FloatTensor(buffer_size, feature_dim).fill_(0)))
                buffer_labels.append(maybe_cuda(torch.LongTensor(buffer_size).fill_(0)))
        
        # buffer_label.index_fill_(0, maybe_cuda(torch.LongTensor(list(range(0, buffer_size)))), 1)  # half pos--1, half neg--0

        # registering as buffer allows us to save the object using `torch.save`
        # [4, 9]
        for i in range(self.vis_map_nums, self.part_nums, self.part_nums//2):
            self.register_buffer('buffer_img_{}'.format(i), buffer_imgs[i])
            self.register_buffer('buffer_vis_map_{}'.format(i), buffer_vis_maps[i])
            self.register_buffer('buffer_vis_indicator_{}'.format(i), buffer_vis_indicators[i])
        
        if use_feature:
            for i in range(self.part_nums):
                self.register_buffer('buffer_feature_{}'.format(i), buffer_features[i])
                self.register_buffer('buffer_label_{}'.format(i), buffer_labels[i])

        # define update and retrieve method
        self.update_method = name_match.update_methods[update](params)
        self.retrieve_method = name_match.retrieve_methods[retrieve](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferPartClassTracker(2, buffer_size, self.part_nums, self.vis_map_nums, self.device)  # we only have positive and negative samples

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