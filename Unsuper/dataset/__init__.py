import torch
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset
from .coco import COCODataset

__all__ = {
    'COCODataset': COCODataset,
}

def build_dataloader(dataset_cfg, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True):

    dataset = __all__[dataset_cfg['name']](
        dataset_cfg,
        training
    )
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler