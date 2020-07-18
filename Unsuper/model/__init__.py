import torch
from torch.utils.data import DataLoader
from ..utils.common_utils import get_dist_info
from .Unsuper import UnSuperPoint
from .ShortcutPoint import ShortcutPoint

__all__ = {
    'UnsuperPoint': UnSuperPoint,
    'ShortcutPoint': ShortcutPoint
}

def build_network(model_cfg, IMAGE_SHAPE, is_training):
    return __all__[model_cfg['name']](model_cfg, IMAGE_SHAPE, is_training)