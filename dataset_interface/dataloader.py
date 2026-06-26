import random

import numpy as np
import pytorch_lightning as pl
import torch

from dataset_interface.dataset_interface import get_parking_data
from utils.config import Configuration

from torch.utils.data import DataLoader


class ParkingDataloaderModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: str):
        # Choose dataset implementation based on config
        # Default: use ParkingDataModuleReal (get_parking_data handles 'real_scene')
        use_carla = False
        if getattr(self.cfg, 'data_mode', None) == 'carla' or ('e2e_parking' in str(getattr(self.cfg, 'data_dir', ''))):
            use_carla = True

        if use_carla:
            # CarlaDataset has signature CarlaDataset(root_dir, is_train, config)
            from dataset.carla_dataset import CarlaDataset

            train_dataset = CarlaDataset(self.cfg.data_dir, is_train=1, config=self.cfg)
            val_dataset = CarlaDataset(self.cfg.data_dir, is_train=0, config=self.cfg)
        else:
            ParkingDataModule = get_parking_data(data_mode=getattr(self.cfg, 'data_mode', 'real_scene'))
            train_dataset = ParkingDataModule(config=self.cfg, is_train=1)
            val_dataset = ParkingDataModule(config=self.cfg, is_train=0)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.cfg.batch_size,
                                       shuffle=True,
                                       num_workers=self.cfg.num_workers,
                                       pin_memory=True,
                                       worker_init_fn=self.seed_worker,
                                       drop_last=True)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=self.cfg.batch_size,
                                     shuffle=False,
                                     num_workers=self.cfg.num_workers,
                                     pin_memory=True,
                                     worker_init_fn=self.seed_worker,
                                     drop_last=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
