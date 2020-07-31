from argparse import ArgumentParser
import torch
from torch import nn
import pytorch_lightning as pl
from Unsuper.config import cfg_from_yaml_file, cfg
from Unsuper.dataset import build_dataloader
from torch.utils.data import DataLoader
from Unsuper.model import build_network

if __name__=='__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--cfg_file', default=None, type=str)
    parser.add_argument('--epoch', default=20, type=int)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    train_dataset, train_dataloader = build_dataloader(cfg['DATA'], args, training=True)
    val_dataset, val_datalaoder = build_dataloader(cfg['DATA'], args, training=True)

    model = build_network(cfg['MODEL'])
    trainer = pl.Trainer()


