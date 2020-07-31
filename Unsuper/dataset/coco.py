import torch
import torch.nn
import cv2
import numpy as np
from pathlib import Path
import random

from .base_dataset import BaseDataset
from ..utils.utils import resize_img, enhance


class COCODataset(BaseDataset):
    default_config = {}

    def init_dataset(self):
        self.name = 'coco'
        if self.is_training:
            base_path = Path(self.config['train_path'], 'COCO/train2014/')
            image_paths = list(base_path.iterdir())
            image_paths = [str(p) for p in image_paths]
            np.random.shuffle(image_paths)
            data_len = len(image_paths)
            if self.config['truncate']:
                image_paths = image_paths[:round(data_len*self.config['truncate'])]
            return len(image_paths), image_paths
        else:
            base_path = Path(self.config['train_path'], 'COCO/val2014/')
            image_paths = list(base_path.iterdir())
            test_files = [str(p) for p in image_paths][:self.config['export_size']]
            return self.config['export_size'], test_files
    
    def __getitem__(self, index):
        img_file = self.train_files[index]
        img = cv2.imread(img_file)
        if self.is_training:
            src_img = resize_img(img, self.config['IMAGE_SHAPE'])  # reshape the image
            dst_img, mat = enhance(src_img, self.config)
            return src_img, dst_img, mat, img_file
        else:
            src_img = cv2.resize(img, (self.config['IMAGE_SHAPE'][1], self.config['IMAGE_SHAPE'][0]))
            return src_img, img_file
        

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
        mat = torch.tensor(mat, dtype=torch.float32, requires_grad=False).squeeze() # B * 3 * 3

        return src_img.permute(0, 3, 1, 2), dst_img.permute(0, 3, 1, 2), mat
    
    def test_collate_batch(*batches):
        src_img = []
        img_idx = []
        for batch in batches[1]:
            src_img.append(batch[0])
            img_idx.append(batch[1])
        src_img_tensor = torch.tensor(src_img, dtype=torch.float32) # B * H * W * C

        return src_img, src_img_tensor.permute(0, 3, 1, 2), img_idx

