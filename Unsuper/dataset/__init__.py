import torch
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset
from .coco import COCODataset
from torch.utils.data import DistributedSampler
from .hpatch import HPatchDataset
from ..utils.common_utils import get_dist_info

__all__ = {
    'COCODataset': COCODataset,
    'HPatchDataset': HPatchDataset
}

def build_dataloader(dataset_cfg, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True):

    if training:
        dataset = __all__[dataset_cfg['train_name']](
            dataset_cfg,
            training
        )
    else:
        dataset = __all__[dataset_cfg['export_name']](
            dataset_cfg,
            training
        )
        
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    if training:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
            drop_last=False, sampler=sampler, timeout=0
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training, collate_fn=dataset.test_collate_batch,
            drop_last=False, sampler=sampler, timeout=0
        )

    return dataset, dataloader, sampler