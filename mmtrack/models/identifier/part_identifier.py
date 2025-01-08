import torch
import time
from mmtrack.utils.meters import AverageMeter
from .base_identifier import BaseIdentifier
from .track_center.part_tracklet import PartTracklet
# from .classifier import RR
from .states.initial_state import InitialState
from .states.tracking_state import TrackingState
from .states.initial_training_state import InitialTrainingState
from .classifier import name_match
from ..builder import MODELS
from mmtrack.models.identifier.utils.utils import maybe_cuda, mini_batch_deep_part_features
from mmtrack.models.reid.losses.GiLt_loss import GiLtLoss
from mmtrack.models.reid.utils import *

@MODELS.register_module()
class PartIdentifier(BaseIdentifier):
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
        self.incremental_st_loss = -1

        # self.feature_extraction_times = torch.zeros(0)  # ms
        # self.identification_times = torch.zeros(0)  # ms

    def init_identifier(self, target_id, rpf_model):
        self.visdom = rpf_model.visdom
        self.debug = rpf_model.debug
        self.save = rpf_model.save
        self.target_id = target_id

        ### set hyperparameter for reid model ###
        losses_weights = {
            GLOBAL: {'id': 1., 'tr': 0.},
            FOREGROUND: {'id': 0., 'tr': 0.},
            CONCAT_PARTS: {'id': 1., 'tr': 0.},  # hanjing
            PARTS: {'id': 0., 'tr': 1.}
        }
        rpf_model.reid.head.GiLt = GiLtLoss(losses_weights=losses_weights, use_visibility_scores=True, triplet_margin=self.params.triplet_margin, ce_smooth=self.params.ce_smooth,loss_name=self.params.triplet_loss, use_gpu=True)

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
            self.tracklets[id] = PartTracklet(img=img, 
                                          img_metas=img_metas,
                                          params=self.params,
                                          observation=tracks[id], 
                                          rescale=rescale) # Tensor
            self.tracklets[id].target_confidence = -1
        if len(self.tracklets.keys()) == 0:
            return None
        
        ### extract deep features ###
        t_extract = time.time()
        self.extract_features(model.reid, self.tracklets)
        self.feature_extraction_times.update((time.time() - t_extract)*1000)
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
        if self.debug:
        # if hasattr(next_state, "target_id") and self.debug:
            # print(next_state.state_name()) 
            ident_result["state"] = next_state.state_name()
            ident_result["st_loss"] = self.newest_st_loss if isinstance(next_state, (TrackingState, InitialTrainingState)) else -1
            ident_result["lt_loss"] = self.newest_lt_loss if isinstance(next_state, (TrackingState, InitialTrainingState)) else -1
            ident_result["incremental_st_loss"] = self.incremental_st_loss if self.incremental_st_loss!= -1 else -1
            ident_result["identification_time"] = self.identification_times
            # print("target id: {}".format(next_state.target_id))
        if next_state is not self.state:
            self.state = next_state
        self.identification_times.update((time.time() - ts_iden)*1000)
        

        ### get result ###
        self.target_id = self.state.target()
        ident_result["target_id"] = self.target_id
        ident_result["tracks_target_conf_bbox"] = {}
        ident_result["bbox_score"] = {}
        ident_result["vis_indicator"] = {}
        ident_result["vis_map"] = {}
        ident_result["ori"] = {}
        ident_result["kpts"] = {}
        ident_result["att_maps"] = {}
        ident_result["visibility_maps"] = {}
        ident_result["buffer_samples"] = {}
        if self.target_id != -1:
            ident_result["target_conf"] = self.tracklets[self.target_id].target_confidence
        else:
            ident_result["target_conf"] = -1
        for idx in self.tracklets.keys():
            ident_result["tracks_target_conf_bbox"][idx] = [self.tracklets[idx].part_target_confidence, self.tracklets[idx].target_confidence, self.tracklets[idx].bbox]
            x1, y1, x2, y2 = self.tracklets[idx].bbox
            ident_result["bbox_score"][idx] = [x1, y1, x2, y2, self.tracklets[idx].bbox_score]
            ident_result["vis_indicator"][idx] = self.tracklets[idx].visibility_indicator.cpu().numpy().tolist()
            ident_result["vis_map"][idx] = self.tracklets[idx].visibility_map.cpu().numpy().tolist()
            ident_result["ori"][idx] = self.tracklets[idx].ori.tolist()
            ident_result["kpts"][idx] = self.tracklets[idx].kpts.tolist()

            ident_result["att_maps"][idx] = self.tracklets[idx].att_map.cpu().numpy().tolist() if self.tracklets[idx].att_map is not None else None
            ident_result["visibility_maps"][idx] = self.tracklets[idx].visibility_map.cpu().numpy().tolist() if hasattr(self.tracklets[idx], "visibility_map") else None

        # ident_result["buffer_samples"]

        
        ### store more info for debug ###
        if self.debug:
            # estimated target confidence
            estimated_conf = ident_result["target_conf"] if ident_result["target_conf"] is not None else -1
            # print(self.estimated_confs, estimated_conf)
            self.estimated_confs = torch.cat((self.estimated_confs, torch.Tensor([estimated_conf])))
            # time consuming for every modules
            time_consuming = "People nums: {} ({:.1f})\t [ms] Feature Extraction {:.3f} ({:.3f})\t Identification {:.3f} ({:.3f})\t".format(
            self.people_nums.val,self.people_nums.avg,
            self.feature_extraction_times.val, self.feature_extraction_times.avg,
            self.identification_times.val,
            self.identification_times.avg
            )
            # print(time_consuming)
            ident_result["time_consuming"] = time_consuming
            is_exist_gt = 1 if gt_bbox is not None and gt_bbox[0] != 0 else -1
            self.is_exist_gt = torch.cat((self.is_exist_gt, torch.Tensor([is_exist_gt])))

            if self.save:
                ident_result["estimated_confs"] = float(self.estimated_confs.cpu().numpy()[-1])
                ident_result["is_exist_gt"] = float(self.is_exist_gt.cpu().numpy()[-1])
                if self.params.buffer_tracker:
                    ident_result["st_pos_size"] = self.classifier.memory_manager.memory["st_set"].buffer_tracker.class_num_cache[1]
                    ident_result["st_neg_size"] = self.classifier.memory_manager.memory["st_set"].buffer_tracker.class_num_cache[0]
                    if self.params.lt_update_method is not None:
                        ident_result["lt_pos_size"] = self.classifier.memory_manager.memory["lt_set"].buffer_tracker.class_num_cache[1]
                        ident_result["lt_neg_size"] = self.classifier.memory_manager.memory["lt_set"].buffer_tracker.class_num_cache[0]
                
            if self.visdom is not None:
                # self.visdom.register(self.estimated_confs, 'lineplot', 0, 'Estimated Confidence')
                # self.visdom.register(self.is_exist_gt, 'lineplot', 0, 'Estimated Confidence')
                self.visdom.register([self.estimated_confs, self.is_exist_gt, "es_conf", "gt"], 'twolinesplot', 0, 'Score')
                self.visdom.register(time_consuming, "text", 0, "Time Consuming")

        return ident_result
    

    def extract_features(self, model, tracklets: dict):
        idx = list(tracklets.keys())[0]
        img_size = tracklets[idx].image_patch.size()
        vis_map_size = tracklets[idx].visibility_map.size()
        img_patches = torch.empty((len(tracklets.keys()), *img_size[1:]), dtype=torch.float32)
        vis_maps = torch.empty((len(tracklets.keys()), *vis_map_size), dtype=torch.float32)
        for i, idx in enumerate(sorted(tracklets.keys())):
            img_patches[i, :] = tracklets[idx].image_patch
            vis_maps[i, :] = tracklets[idx].visibility_map


        # compute deep features with mini-batches
        num = len(tracklets.keys())
        total_x = maybe_cuda(img_patches)

        # try to normalize
        # total_x = total_x / 255.0
        # # 2. 使用 ImageNet 的均值和标准差进行标准化：
        # mean = torch.tensor([0.485, 0.456, 0.406], device=total_x.device).view(1, 3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225], device=total_x.device).view(1, 3, 1, 1)
        # # 归一化操作
        # total_x = (total_x - mean) / std


        total_vis_map = maybe_cuda(vis_maps)
        deep_features_, att_maps_ = mini_batch_deep_part_features(model, total_x, num, total_vis_map, True)
        for i, idx in enumerate(sorted(tracklets.keys())):
            tracklets[idx].deep_feature = deep_features_[i]  # (5,512)
            tracklets[idx].att_map = att_maps_[i]  # (8,4)
    
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