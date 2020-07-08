import torch
import torch.nn
import cv2
import numpy as np
from pathlib import Path
import random

from .base_dataset import BaseDataset
from Unsuper.settings import DATA_PATH, EXPER_PATH
from ..utils.utils import resize_img, enhance


class COCODataset(BaseDataset):
    default_config = {}

    def _init_dataset(self, **config):
        base_path = Path(DATA_PATH, 'COCO/train2014/')
        image_paths = list(base_path.iterdir())
        image_paths = [str(p) for p in image_paths]
        random.shuffle(image_paths)
        data_len = len(image_paths)
        train_files = image_paths[:round(data_len * config['validation_size'])]
        return round(data_len*config['validation_size']), train_files
    
    def __getitem__(self, index):
        img_file = self.train_files[index]
        img = cv2.imread(img_file)
        src_img = resize_img(img, self.config['IMAGE_SHAPE'])  # reshape the image
        dst_img, mat = enhance(img, self.config['IMAGE_SHAPE'], self.config['perspective'])
        return src_img, dst_img, mat
