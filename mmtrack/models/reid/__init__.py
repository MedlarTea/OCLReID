# Copyright (c) OpenMMLab. All rights reserved.
from .base_reid import BaseReID
from .fc_module import FcModule
from .gap import GlobalAveragePooling
from .linear_reid_head import LinearReIDHead
from .multi_scales_reid import MultiScalesReID
from .linear_head import LinearHead
from .base_classifier import BaseClassifier

from .part_classifier import PartClassifier
from .part_head import PartHead

from .part_weighted_classifier import PartWeightedClassifier
from .part_weighted_head import PartWeightedHead



__all__ = ['BaseReID', 'GlobalAveragePooling', 'LinearReIDHead', 'FcModule', "MultiScalesReID", "LinearHead", "BaseClassifier", "PartClassifier", "PartHead", "PartWeightedClassifier", "PartWeightedHead"]
