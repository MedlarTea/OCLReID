# from lib2to3.pytree import Base
from .base import BaseClassifier
import numpy as np
from sklearn.linear_model import Ridge

class GlobalResNetClassifier(BaseClassifier):
    """Multiple part classifiers with KPR
    """
    def __init__(self, params, reid_model):
        # classifier
        self.params = params
        self.reid_model = reid_model
        self.st_clf = Ridge(alpha=self.params['rr_alpha'], random_state=self.params['seed'])

        self.max_size = self.params["lt_rate"] * self.params["mem_size"]

        self.pos_features = np.zeros((self.max_size, self.params['deep_feature_dim']))
        self.neg_features = np.zeros((self.max_size, self.params['deep_feature_dim']))

        self.indicators = {
            "pos": 0,
            "neg": 0,
            "pos_seen": 0,
            "neg_seen": 0
        }
    
    def update(self, tracklets: dict, target_id: int):
        """Update the memory with the new tracklets"""
        for idx in sorted(tracklets.keys()):
            feature = tracklets[idx].deep_feature.cpu().numpy()
            if idx == target_id:
                self.pos_features[self.indicators["pos"], :] = feature
                self.indicators["pos"] += 1
                self.indicators["pos_seen"] += 1
                self.indicators["pos"] %= self.max_size
            else:
                self.neg_features[self.indicators["neg"], :] = feature
                self.indicators["neg"] += 1
                self.indicators["neg_seen"] += 1
                self.indicators["neg"] %= self.max_size
        return True

    def train(self):
        if self.indicators["pos_seen"] == 0 or self.indicators["neg_seen"] == 0:
            return 0, 0
        pos_x = self.pos_features[:min(self.indicators["pos_seen"], self.max_size), :]
        neg_x = self.neg_features[:min(self.indicators["neg_seen"], self.max_size), :]
        x = np.concatenate([pos_x, neg_x], axis=0)
        y = np.concatenate([np.ones(pos_x.shape[0]), np.zeros(neg_x.shape[0])], axis=0)
        self.st_clf.fit(x, y)
        return 0, 0
    
    def predict(self, tracklets: dict, state="tracking"):
        """Predict target confidence of the cancidate, the confidence is an average score calculated from visible part features

        """
        for idx in sorted(tracklets.keys()):
            deep_feature = tracklets[idx].deep_feature.cpu().numpy()
            score = None
            if not hasattr(self.st_clf, 'coef_') or not hasattr(self.st_clf, 'intercept_'):
                continue
            else:
                score = self.st_clf.predict(deep_feature.reshape(1, -1)).item()
            if score is not None:
                tracklets[idx].target_confidence = score
            else:
                tracklets[idx].target_confidence = 0.5  # do not know pos or neg, maximum entropy
    

    

