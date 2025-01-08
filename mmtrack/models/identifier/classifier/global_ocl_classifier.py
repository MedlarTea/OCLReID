# from lib2to3.pytree import Base
from .base import BaseClassifier
from ..memory_manager.global_ocl_memory_manager import GlobalOCLMemoryManager


import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

from ..utils.utils import maybe_cuda, AverageMeter
from ..utils.loss import SupConLoss
from ..utils.kd_manager import KdManager
import copy

from sklearn.linear_model import Ridge

class GlobalOCLClassifier(BaseClassifier):
    def __init__(self, params, reid_model):
        # classifier
        self.clf = reid_model
        # self.clf.cuda(device)
        self.st_clf = Ridge(alpha=params.rr_alpha, random_state=params.seed)
        self.memory_manager = GlobalOCLMemoryManager(self.clf, params)  # reference?
        self.verbose = False
        self.params = params
        self.batch_size = params.batch_size
        self.lr = params.learning_rate
        self.wd = params.weight_decay
        self.epochs = params.epochs
        self.optim = params.optimizer
        self.backbone = params.backbone  # resnet18 or resnet50
        self.optimizer = None
        self.record_losses = AverageMeter()
        self.init_classifier()
        # self.keys = ["st_pos", "st_neg", "lt_pos", "lt_neg"]

        # come from "online continual learning repo"
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.old_labels = []

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

    @torch.enable_grad()
    def train(self):
        self.clf.train()
        for _epoch in range(self.epochs):
            # retrieve the sample indices
            stf_x, stf_y = self.memory_manager.retrieve_st_features()
            self.train_st(stf_x, stf_y)  # train the ridge regression

            # print("st pos/neg:{}/{}".format(self.memory_manager.memory["st_set"].buffer_tracker.class_num_cache[1], self.memory_manager.memory["st_set"].buffer_tracker.class_num_cache[0]))
            st_x, st_y = self.memory_manager.retrieve_st()
            st_loss = self.train_lt(st_x, st_y)
            if self.verbose:
                print("\nst_loss:", st_loss)
            
            lt_x, lt_y = self.memory_manager.retrieve_lt()
            lt_loss = self.train_lt(lt_x, lt_y)
            if self.verbose:
                print("lt_loss:", lt_loss)
        return st_loss, lt_loss
    
    def train_st(self, x, y):
        train_set = x.cpu().numpy()
        labels = y.cpu().numpy()
        self.st_clf.fit(train_set, labels)

    @torch.enable_grad()
    def train_lt(self, x, y):
        self.clf.train()
        # for BN
        if x.size(0) < 2:
            return

        _, loss_dict = self.clf.forward_train(x, y)
        loss = loss_dict["ce_loss"]
        trained_acc = loss_dict["accuracy"]
        self.optimizer.zero_grad()
        record_loss = loss.item()
        # print("loss: ", loss)
        # print("trained_acc: ", trained_acc)
        loss.backward()
        self.optimizer.step()
        self.record_losses.update(record_loss)
        return record_loss
    
    def predict(self, tracklets: dict, state="tracking"):

        ### predict with st classifier ###
        features = []
        idxs = []
        for idx in tracklets.keys():
            if self.params.st_feature == "deep":
                features.append(tracklets[idx].deep_feature)
            elif self.params.st_feature == "joint":
                features.append(tracklets[idx].joints_feature)
            elif self.params.st_feature == "all":
                features.append(torch.cat([tracklets[idx].deep_feature,tracklets[idx].joints_feature]))
            else:
                return ValueError("Wrong feature")
            # deep_features.append(tracklets[idx].deep_feature)
            idxs.append(idx)
        features = torch.stack(features)
        scores = self.st_clf.predict(features.cpu().numpy())

        for i in range(scores.shape[0]):
            tracklets[idxs[i]].target_confidence = scores[i].tolist()
        return scores.tolist()
    
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
    

    

