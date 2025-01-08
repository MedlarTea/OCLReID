import torch
import time
from mmtrack.utils.meters import AverageMeter
from .base_identifier import BaseIdentifier
from .track_center.tracklet import Tracklet
from .states.initial_state import InitialState
from .classifier import name_match
from ..builder import MODELS
from mmtrack.models.identifier.utils.utils import maybe_cuda, mini_batch_deep_features, euclidean_distance, nonzero_indices, ohe_label


@MODELS.register_module()
class GlobalIdentifier(BaseIdentifier):
    """Identifier for identifying the target person
    CLassifier: Resnet + mlp
    Input: image patch                            
    """
    def __init__(self, params):
        self.target_id = -1
        self.target_conf = -1
        self.params = params

        # classifier
        self.classifier = None
        # life cycle
        self.state = None
        # for debug
        self.visdom = None
        self.debug = False
        self.save = False

        self.estimated_confs = torch.zeros(0)
        self.is_exist_gt = torch.zeros(0)
        self.people_nums = AverageMeter()
        self.feature_extraction_times = AverageMeter()
        self.identification_times = AverageMeter()
        self.frame_id = 0
        self.newest_st_loss = -1    
        self.newest_lt_loss = -1

        # self.feature_extraction_times = torch.zeros(0)  # ms
        # self.identification_times = torch.zeros(0)  # ms

    def init_identifier(self, target_id, rpf_model):
        self.visdom = rpf_model.visdom
        self.debug = rpf_model.debug
        self.save = rpf_model.save
        self.target_id = target_id
        self.classifier = name_match.classifiers[self.params.agent](params=self.params, reid_model=rpf_model.reid)
        self.state = InitialState(self.params)
        

    def identify(self, 
                 img, 
                 img_metas, 
                 model, 
                 tracks:dict, 
                 frame_id, 
                 rescale=False,
                 gt_bbox=None,
                 **kwargs):
        """extract image patches features based on the tracks information
        frame_id (int): frame id
        image (array): with shape (3, width, height)
        tracks (dict): {id(int):bbox[tl_x, tl_y, br_x, br_y]}
        """
        ### initialize tracks ###
        self.frame_id = frame_id
        self.tracklets = {}
        for id in tracks.keys():
            ### skip unhealthy bbox especially under occlusion ###
            # if len(tracks[id]) > 4 and tracks[id][4] < 0.5:
            #     continue
            self.tracklets[id] = Tracklet(img=img, 
                                          img_metas=img_metas,
                                          observation=tracks[id], 
                                          rescale=rescale) # Tensor
        if len(self.tracklets.keys()) == 0:
            return None
        
        ### extract deep features ###
        self.extract_features(model.reid, self.tracklets)
        # print(self.tracklets[list(tracks.keys())[0]].deep_feature.shape)
        # print(self.tracklets[list(tracks.keys())[0]].joints_feature.shape)

        self.people_nums.update(len(self.tracklets.keys()))
        ### init classifier ###
        # ts_extract = time.time()

        ### update states (life cycle)###
        # 1. update memory
        # 2. update classifier
        # 3. reid life cycle
        ident_result = {}
        ts_iden = time.time()
        next_state = self.state.update(identifier=self, tracklets=self.tracklets)
        if next_state is not self.state:
            self.state = next_state
        self.target_id = self.state.target()
        

        ### get result ###
        ident_result["state"] = next_state.state_name()
        ident_result["target_id"] = self.target_id
        ident_result["threshold"] = self.params.reid_pos_confidence_thresh
        ident_result["tracks_target_conf_bbox"] = {}
        if self.target_id != -1:
            ident_result["target_conf"] = self.tracklets[self.target_id].target_confidence
        else:
            ident_result["target_conf"] = -1
        for idx in self.tracklets.keys():
            ident_result["tracks_target_conf_bbox"][idx] = [-1, self.tracklets[idx].target_confidence, self.tracklets[idx].bbox]

        return ident_result
    

    def extract_features(self, model, tracklets: dict):
        idx = list(tracklets.keys())[0]
        img_size = tracklets[idx].image_patch.size()
        img_patches = torch.empty((len(tracklets.keys()), *img_size[1:]), dtype=torch.float32)
        for i, idx in enumerate(sorted(tracklets.keys())):
            img_patches[i, :] = tracklets[idx].image_patch


        # compute deep features with mini-batches
        num = len(tracklets.keys())
        total_x = maybe_cuda(img_patches)
        deep_features_ = mini_batch_deep_features(model, total_x, num)
        for i, idx in enumerate(sorted(tracklets.keys())):
            tracklets[idx].deep_feature = deep_features_[i]  # (128)
    
    def update_memory(self, tracklets: dict, target_id: int):
        """
        return bool
        """
        return self.classifier.memory_manager.update(tracklets, target_id)

    def update_classifier(self):
        """
        return loss
        """
        return self.classifier.train()

    def predict(self, tracklets: dict, state):
        """
        update the confidences of the tranlets
        """

        ### predict target confidence ###
        return self.classifier.predict(tracklets, state)

        ### assign scores to tracklets ###
        # for i in range(len(target_confidences)):
        #     tracklets[idxs[i]].target_confidence = target_confidences[i]
    
    
        