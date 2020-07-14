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

    def _init_dataset(self, config):
        base_path = Path(DATA_PATH, 'COCO/train2014/')
        image_paths = list(base_path.iterdir())
        image_paths = [str(p) for p in image_paths]
        random.shuffle(image_paths)
        data_len = len(image_paths)
        train_files = image_paths[ : round( data_len * (1 - config['validation_size']) )]
        return round(train_files), train_files
    
    def __getitem__(self, index):
        img_file = self.train_files[index]
        img = cv2.imread(img_file)
        src_img = resize_img(img, self.config['IMAGE_SHAPE'])  # reshape the image
        dst_img, mat = enhance(img, self.config)
        return src_img, dst_img, mat

    def collate_batch(*batches):
        src_img = []
        dst_img = []
        mat = []
        for batch in batches[1]:
            src_img.append(batch[0])
            dst_img.append(batch[1])
            mat.append(batch[2])
        src_img = torch.tensor(src_img, dtype=torch.float32) # B * H * W * C
        dst_img = torch.tensor(dst_img, dtype=torch.float32) # B * H * W * C
        mat = torch.tensor(mat, dtype=torch.float32).squeeze() # B * 3 * 3
        mat = mat

        return src_img.permute(0, 3, 1, 2), dst_img.permute(0, 3, 1, 2), mat
