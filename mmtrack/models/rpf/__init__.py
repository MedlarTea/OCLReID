# Copyright (c) OpenMMLab. All rights reserved.
from imageio import imopen
from .base import BaseRobotPersonFollower
from .naive_rpf import NaiveRPF
from .part_rpf import PartRPF
from .global_rpf import GlobalRPF
from .rpf_part_identifier import RPFPartIdentifier

__all__ = [
    'BaseRobotPersonFollower', 'NaiveRPF', "GlobalRPF", "PartRPF", "RPFPartIdentifier"
]
