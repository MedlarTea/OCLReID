from .base import BaseClassifier
from ..memory_manager.part_ocl_memory_manager import PartOCLMemoryManager

import numpy as np
import torch
from torch.nn import functional as F

from ..utils.utils import maybe_cuda, MyAverageMeter
from ..utils.loss import SupConLoss

from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from mmtrack.models.reid.utils import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS
import copy
import time

class PartOCLWeightedClassifier(BaseClassifier):
    def __init__(self, params, reid_model):
        # classifier
        self.clf = reid_model
        # self.clf.cuda(device)
        self.vis_map_size = params.vis_map_size
        self.vis_map_nums = self.vis_map_size[0]
        self.vis_map_res = self.vis_map_size[1:]
        self.part_nums = self.vis_map_nums+1 if not params.use_ori else 2*(self.vis_map_nums+1)
        self.end_part_indexs = list(range(self.vis_map_nums, self.part_nums, self.part_nums//2))
        self.st_clfs = {i:Ridge(alpha=params.rr_alpha, random_state=params.seed) for i in range(self.part_nums)}
        self.st_clf_weights = {i:1.0 for i in range(self.part_nums)}
        self.part_triplet_losses = [MyAverageMeter() for i in range(self.part_nums-1)] if not params.use_ori else [MyAverageMeter() for i in range(self.part_nums-2)] 
        self.temp=0.1

        self.memory_manager = PartOCLMemoryManager(self, self.clf, params)  # reference?
        self.params = params
        self.batch_size = params.batch_size
        self.lr = params.learning_rate
        self.wd = params.weight_decay
        self.epochs = params.epochs
        self.optim = params.optimizer
        self.backbone = params.backbone  # resnet18 or resnet50
        self.optimizer = None
        self.record_losses = MyAverageMeter()
        self.init_classifier()

        self.task_seen = 0
        self.old_labels = []

        self.train_st_time = MyAverageMeter()
        self.train_lt_s_time = MyAverageMeter()
        self.train_lt_l_time = MyAverageMeter()
        self.retrieve_lt_s_time = MyAverageMeter()
        self.retrieve_lt_l_time = MyAverageMeter()
        self.infer_st_time = MyAverageMeter()

        self.exchange_lt_model = True

    def init_classifier(self):
        """
        freeze some model weights and init optimizer
        """
        max_not_freeze_num_range = 169 if self.backbone == "resnet50" else 70
        not_freeze_num_range = []
        ### Hanjing ###
        if self.params.not_freeze == "conv3":
            not_freeze_num_range = list(range(159,169)) if self.backbone == "resnet50" else list(range(30,70))
        ### Hanjing ###
        elif self.params.not_freeze == "conv4":
            not_freeze_num_range = list(range(159,169)) if self.backbone == "resnet50" else list(range(45,70))
        elif self.params.not_freeze == "fcs":
            not_freeze_num_range = list(range(159,169)) if self.backbone == "resnet50" else list(range(60,70))
        elif self.params.not_freeze == "fc_out":
            not_freeze_num_range = list(range(163,169)) if self.backbone == "resnet50" else list(range(64,70))
        elif self.params.not_freeze == "classifier":
            not_freeze_num_range = list(range(167,169)) if self.backbone == "resnet50" else list(range(68,70))
        elif self.params.not_freeze == "all":
            not_freeze_num_range = list(range(0,169)) if self.backbone == "resnet50" else list(range(0,70))
        for i, (name, param) in enumerate(self.clf.named_parameters()):
            if i not in not_freeze_num_range and i < max_not_freeze_num_range:
                param.requires_grad = False
            # print(i, name, param.requires_grad)
        if self.params.lt_update_method is not None:
            self.optimizer = self.setup_opt(self.optim, self.clf, self.lr, self.wd)

    def _lock_copy_model(self, st_loss):
        self.freeze_model = copy.deepcopy(self.clf)
        self.last_st_loss = st_loss

    @torch.enable_grad()
    def train(self):
        self.clf.train()
        for _epoch in range(self.epochs):
            # st-classifier training
            t=time.time()
            stf_x, stf_y = self.memory_manager.retrieve_st_features()
            self.train_st(stf_x, stf_y)  # train the ridge regression

            self.train_st_time.update((time.time() - t)*1000)

            # lt-classifier training
            if self.params.lt_update_method is not None:
                # print("Short Term:")
                # self.memory_manager.memory["st_set"].buffer_tracker.print_class_nums()
                st_loss = -1
                if self.memory_manager.memory["st_set"].buffer_tracker.class_num_cache.sum() > 0:
                    st_t=time.time()
                    st_x, st_y, vis_map, vis_indicator  = self.memory_manager.retrieve_st()
                    self.retrieve_lt_s_time.update((time.time() - st_t)*1000)
                    if self._check_for_lt_learning(vis_indicator, st_y):
                        st_t=time.time()
                        all_loss, st_loss = self.train_lt(st_x, st_y, vis_map, vis_indicator)
                        self._lock_copy_model(all_loss)  # lock and copy the model for keyframe selection  1111111
                        self.train_lt_s_time.update((time.time() - st_t)*1000)
                
                # print("Long Term:")
                # self.memory_manager.memory["lt_set"].buffer_tracker.print_class_nums()
                lt_loss = -1
                if self.memory_manager.memory["lt_set"].buffer_tracker.class_num_cache.sum() > 0:
                    lt_t=time.time()
                    lt_x, lt_y, vis_map, vis_indicator = self.memory_manager.retrieve_lt()
                    self.retrieve_lt_l_time.update((time.time() - lt_t)*1000)
                    if self._check_for_lt_learning(vis_indicator, lt_y):
                        lt_t=time.time()
                        _, lt_loss = self.train_lt(lt_x, lt_y , vis_map, vis_indicator)
            else:
                st_loss = 0
                lt_loss = 0
        # print("[Memory Retrieve] LT_S {:.3f} ({:.3f})\tLT_L {:.3f} ({:.3f})".format(self.retrieve_lt_s_time.val, self.retrieve_lt_s_time.avg, self.retrieve_lt_l_time.val, self.retrieve_lt_l_time.avg))
        # print("[Classifier Update] ST {:.3f} ({:.3f})\tLT_S {:.3f} ({:.3f})\tLT_L {:.3f} ({:.3f})".format(self.train_st_time.val, self.train_st_time.avg, self.train_lt_s_time.val, self.train_lt_s_time.avg, self.train_lt_l_time.val, self.train_lt_l_time.avg))
        return st_loss, lt_loss
    
    def _check_for_lt_learning(self, vis_indicators, labels):
        _front_pos_sum = torch.sum(vis_indicators[labels==1][:, :self.end_part_indexs[0]], dim=0)>1
        _back_pos_sum = torch.sum(vis_indicators[labels==1][:, self.end_part_indexs[0]+1:self.end_part_indexs[1]], dim=0)>1 if self.params.use_ori else torch.zeros(1)
        _front_neg_sum = torch.sum(vis_indicators[labels==0][:, :self.end_part_indexs[0]], dim=0)>1
        _back_neg_sum = torch.sum(vis_indicators[labels==0][:, self.end_part_indexs[0]+1:self.end_part_indexs[1]], dim=0)>1 if self.params.use_ori else torch.zeros(1)
        _front_sum = (_front_pos_sum * _front_neg_sum).sum()  # there are intersection visible parts of front body
        _back_sum = (_back_pos_sum * _back_neg_sum).sum()
        # if len(labels) > torch.sum(labels).item() and torch.sum(labels).item() > 1 and (_front_sum>0 or _back_sum>0):
        #     return True
        if _front_sum>0 or _back_sum>0:
            return True
        return False
    
    def train_st(self, x, y):
        """Train the Ridge Regression Model
        Input:
            x(list): part features with shape of (part_nums, sample_nums, feature_size)
            y(list): labels with shape of (part_nums, sample_nums)
        Output:
            Update every part-classifier
        """
        # print("train_x:", len(x))
        # print("train_y:", len(y))
        for st_clf_key, train_x, train_y in zip(sorted(self.st_clfs.keys()), x, y):
            # print("[{}] posNum: {} negNum: {}".format(st_clf_key, train_x[train_y==1].shape[0], train_x[train_y==0].shape[0]))
            if train_x[train_y==1].shape[0] < self.params.initial_training_num_samples:
                continue
            # print("train", st_clf_key, train_x.shape, train_y.shape)
            train_x = train_x.cpu().numpy()
            train_y = train_y.cpu().numpy()
            self.st_clfs[st_clf_key].fit(train_x, train_y)
    
    def train_st_both(self, st_x, st_y, lt_x, lt_y):
        """Train the Ridge Regression Model
        Input:
            x(list): part features with shape of (part_nums, sample_nums, feature_size)
            y(list): labels with shape of (part_nums, sample_nums)
        Output:
            Update every part-classifier
        """
        # print("train_x:", len(x))
        # print("train_y:", len(y))
        for st_clf_key, train_st_x, train_st_y, train_lt_x, train_lt_y in zip(sorted(self.st_clfs.keys()), st_x, st_y, lt_x, lt_y):
            # print("[{}] posNum: {} negNum: {}".format(st_clf_key, train_x[train_y==1].shape[0], train_x[train_y==0].shape[0]))
            if train_st_x[train_st_y==1].shape[0] < self.params.initial_training_num_samples:
                continue
            # print("train", st_clf_key, train_x.shape, train_y.shape)
            train_x = torch.cat([train_lt_x, train_st_x])
            train_y = torch.cat([train_lt_y, train_st_y])
            train_x = train_x.cpu().numpy()
            train_y = train_y.cpu().numpy()
            self.st_clfs[st_clf_key].fit(train_x, train_y)


    @torch.enable_grad()
    def train_lt(self, x, y, vis_map=None, vis_indicator=None, is_vis_att_map=False, is_train=True):
        """Train the backbone with CE_LOSS and PART_TRIPLET_LOSS
        Input:
            x(Tensor): part features with shape of (sample_nums, 3, img_H, img_W)
            y(Tensor): labels with shape of (sample_nums)
        Output:
            Update the backbone
        """
        self.clf.train()
        # for BN
        if x.size(0) < 2:
            return
        _, loss, loss_dict = self.clf.forward_train(x, y, vis_map, vis_indicator, is_vis_att_map, is_train)
        self.optimizer.zero_grad()
        record_loss = loss.item()
        part_triplet_losses = loss_dict[PARTS]['ts'].tolist()
        for i, part_triplet_loss in enumerate(part_triplet_losses):
            if part_triplet_loss != -1:
                self.part_triplet_losses[i].update(part_triplet_loss)
        # print("Part Losses(cur/avg):" + "".join([" {}:{:.3f}({:.3f})".format(i,part_loss.val,part_loss.avg) for i, part_loss in enumerate(self.part_triplet_losses)]))
        
        # print("trained_acc: ", trained_acc)
        loss.backward()
        self.optimizer.step()
        self.record_losses.update(record_loss)
        return record_loss, loss_dict
    
    def predict(self, tracklets: dict, state="tracking"):
        """Predict target confidence of the cancidate, the confidence is an average score calculated from visible part features

        """
        ### predict with st classifier ###
        scores = []
        t_st_infer=time.time()
        for idx in sorted(tracklets.keys()):
            # get part-global feature
            feature = tracklets[idx].deep_feature  # (5,512)
            vis_indicator = tracklets[idx].visibility_indicator  # (10), front's head, torso, legs, feet and whole body; back's ...
            # print("vis_indicator: ", vis_indicator)
            if self.params.use_ori:
                ori = tracklets[idx].binary_ori  # 0 for not seeing face (front), 1 for seeing face (Back)
                start_idx = 0 if ori == 0 else self.part_nums//2
                seg_idx = self.part_nums//2
            else:
                start_idx = 0
                seg_idx = self.part_nums

            part_scores, target_score = self._tracking_predict(start_idx, seg_idx, vis_indicator, feature)

            for part_idx in part_scores.keys():
                tracklets[idx].part_target_confidence[part_idx] = part_scores[part_idx]
            tracklets[idx].target_confidence = target_score
            scores.append(target_score)
        self.infer_st_time.update((time.time() - t_st_infer)*1000)
        # print("[ST-Classifier Infer] {:.3f} ({:.3f})".format(self.infer_st_time.val, self.infer_st_time.avg))
        # print("scores: ", scores)
        return scores

    def _tracking_predict(self, start_idx, seg_idx, vis_indicator, feature):
        part_scores = {}
        for i, part_idx in enumerate(range(start_idx, start_idx + seg_idx)):
            ### Only use the global feature ###
            # if part_idx != (start_idx + seg_idx-1):
            #     continue
        
            ### aggregate all part-global features ###
            if vis_indicator[part_idx] == 0:
                continue
            elif not hasattr(self.st_clfs[part_idx], 'coef_') or not hasattr(self.st_clfs[part_idx], 'intercept_'):
                continue
            else:
                part_scores[part_idx] = self.st_clfs[part_idx].predict(feature[[i]].cpu().numpy()).item()

        if len(part_scores)!=0 and sum(part_scores.values())!=0:
            avg_score = (sum(part_scores.values())/len(part_scores))  # simple average
        else:
            avg_score = 0.5  # do not know pos or neg, maximum entropy
        return part_scores, avg_score
    
    def _reid_predict(self, start_idx, seg_idx, vis_indicator, feature):
        """Re-identify the target by matching strategy between current candidate and gallery features in the long-term buffer
        """
        part_scores = {}
        for i, part_idx in enumerate(range(start_idx, start_idx + seg_idx)):
            ### aggregate all part-global features ###
            if vis_indicator[part_idx] == 0:
                continue
            # elif not hasattr(self.st_clfs[part_idx], 'coef_') or not hasattr(self.st_clfs[part_idx], 'intercept_'):
            #     continue
            lt_buffer =self.memory_manager.memory["lt_set"]
            lt_buffer_tracker = lt_buffer.buffer_tracker
            if lt_buffer_tracker.return_part_nums(part_idx)[1] > 0:
                lt_buffer_feature = getattr(lt_buffer, "buffer_feature_{}".format(part_idx))
                lt_pos_indexes = torch.Tensor(list(lt_buffer_tracker.class_index_cache[part_idx][1])).long()
                lt_pos_features = lt_buffer_feature[lt_pos_indexes]
                part_scores[part_idx] = np.mean(np.dot(lt_pos_features.cpu().numpy(), feature[[i]].cpu().numpy().transpose()))
        # print("part_scores: ", part_scores)
        if len(part_scores)!=0 and sum(part_scores.values())!=0:
            avg_score = (sum(part_scores.values())/len(part_scores))  # simple average
        else:
            avg_score = 0.5  # do not know pos or neg, maximum entropy
        return part_scores, avg_score

    def _adjust_clf_weights(self, x, y, vis_map=None, vis_indicator=None,):
        """Adjust classifier weights based on the "adaptability"
        """
        
        for i in range(4):
            if self.part_triplet_losses[i].count < 100:
                continue
            part_loss_mean, part_loss_std = self.part_triplet_losses[i].get_u_std(100)
            self.st_clf_weights[i] = np.exp(-part_loss_mean**2 *(2*part_loss_std))
            # self.st_clf_weights[i] = np.exp(-self.part_triplet_losses[i].avg/self.temp)
        for i in range(5,9):
            if self.part_triplet_losses[i-1].count < 100:
                continue
            part_loss_mean, part_loss_std = self.part_triplet_losses[i-1].get_u_std(100)
            self.st_clf_weights[i] = np.exp(-part_loss_mean**2 *(2*part_loss_std))


    def setup_opt(self, optimizer, model, lr, wd):
        if optimizer == 'SGD':
            optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr,
                                    weight_decay=wd)
        elif optimizer == 'Adam':
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr,
                                    weight_decay=wd)
        elif optimizer == 'Adagrad':
            optim = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=lr,
                                        weight_decay=wd)
        else:
            raise Exception('wrong optimizer name')
        return optim
    
    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SupContrastReplay', 'SCP']:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            return ce(logits, labels)
    

    

