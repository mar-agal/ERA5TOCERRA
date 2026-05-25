import os
import random
import numpy as np
import torch
from typing import Optional, List
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.dataset import NetCDFSpatialDataset, FilteredDatasetWrapper

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class GreeceDownscalingDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        upsample: bool = True,
        used_channels: Optional[List[int]] = None,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.era5_train_path: str = ""
        self.cerra_train_path: str = ""
        self.era5_val_path: str = ""
        self.cerra_val_path: str = ""
        self.era5_test_path: str = ""
        self.cerra_test_path: str = ""
        
        self.era5_stats_json: str = ""
        self.cerra_stats_json: str = ""
        self.orog_stats_json: str = ""

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        self.channel_indices = used_channels if used_channels is not None else list(range(7))
        
        self.persistent_workers = True if num_workers > 0 else False
        self.prefetch_factor = 2 if num_workers > 0 else None

    def setup(self, stage: Optional[str] = None) -> None:
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)
        
        if stage == "fit" or stage is None:
            if not self.train_dataset:
                raw_train = NetCDFSpatialDataset(
                    era5_nc_path=self.era5_train_path, 
                    cerra_nc_path=self.cerra_train_path,
                    era5_stats_json=self.era5_stats_json, 
                    cerra_stats_json=self.cerra_stats_json,
                    orog_stats_json=self.orog_stats_json, 
                    upsample=self.hparams.upsample
                )
                self.train_dataset = FilteredDatasetWrapper(raw_train, self.channel_indices)
                
            if not self.val_dataset:
                raw_val = NetCDFSpatialDataset(
                    era5_nc_path=self.era5_val_path, 
                    cerra_nc_path=self.cerra_val_path,
                    era5_stats_json=self.era5_stats_json, 
                    cerra_stats_json=self.cerra_stats_json,
                    orog_stats_json=self.orog_stats_json, 
                    upsample=self.hparams.upsample
                )
                self.val_dataset = FilteredDatasetWrapper(raw_val, self.channel_indices)

        if stage == "test" or stage is None:
            if not self.test_dataset:
                raw_test = NetCDFSpatialDataset(
                    era5_nc_path=self.era5_test_path, 
                    cerra_nc_path=self.cerra_test_path,
                    era5_stats_json=self.era5_stats_json, 
                    cerra_stats_json=self.cerra_stats_json,
                    orog_stats_json=self.orog_stats_json, 
                    upsample=self.hparams.upsample
                )
                self.test_dataset = FilteredDatasetWrapper(raw_test, self.channel_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=seed_worker,
            shuffle=True, 
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=seed_worker,
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=seed_worker,
            shuffle=False
        )
