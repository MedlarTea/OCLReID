# Copyright (c) OpenMMLab. All rights reserved.
from .l2_loss import L2Loss
from .multipos_cross_entropy_loss import MultiPosCrossEntropyLoss
from .triplet_loss import TripletLoss
from .part_averaged_triplet_loss import PartAveragedTripletLoss

__all__ = ['L2Loss', 'TripletLoss', 'MultiPosCrossEntropyLoss', "PartAveragedTripletLoss"]
