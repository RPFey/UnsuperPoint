import torch
import torch.nn
import glob
import cv2
import numpy as np
from pathlib import Path
import random

from .base_dataset import BaseDataset
from Unsuper.settings import DATA_PATH
from ..utils.utils import resize_img, enhance

class HPatchDataset(BaseDataset):
    default_config = {}

    def _init_dataset(self):
        self.name = 'hpatch'
        base_path = Path(DATA_PATH, 'HPatch')
        folders = list(base_path.iterdir())
        img_paths = []
        for folder in folders:
            imgs = glob.glob(str(folder)+'/*.ppm')
            img_paths += imgs
        data_len = len(img_paths)
        return data_len, img_paths

    def __getitem__(self, index):
        if self.is_training:
            raise NotImplementedError
        else:
            img_file = self.train_files[index]
            img = cv2.imread(img_file)
            new_h, new_w = self.config['IMAGE_SHAPE']
            src_img = cv2.resize(img, (new_w, new_h))
            return src_img, img_file

    def test_collate_batch(*batches):
        img = []
        img_idx = []
        for batch in batches[1]:
            img.append(batch[0])
            img_idx.append(batch[1])
        img_src = torch.tensor(img, dtype=torch.float32)

        return img, img_src.permute(0, 3, 1, 2), img_idx
        