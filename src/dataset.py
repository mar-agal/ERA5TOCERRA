import os
import json
import torch
import numpy as np
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, List

class NetCDFSpatialDataset(Dataset):
    """
    Preloads static invariant features to eliminate redundant I/O disk operations,
    validates temporal alignment, and enforces strict NaN checking on raw arrays.
    """
    def __init__(
        self, 
        era5_nc_path: str, 
        cerra_nc_path: str, 
        era5_stats_json: str,
        cerra_stats_json: str,
        orog_stats_json: str,
        upsample: bool = True
    ) -> None:
        super().__init__()
        
        self.static_nc_path = "/kaggle/input/datasets/mariaagalioti/static-data/CERRA_Greece_Static_B1_CHW.nc"
        
        for path in [era5_nc_path, cerra_nc_path, self.static_nc_path, era5_stats_json, cerra_stats_json, orog_stats_json]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing critical pipeline file: {path}")
        
        self.era5_ds = xr.open_dataset(era5_nc_path)
        self.cerra_ds = xr.open_dataset(cerra_nc_path)
        self.static_ds = xr.open_dataset(self.static_nc_path)
        self.upsample = upsample
        
        if not np.array_equal(self.era5_ds.time.values, self.cerra_ds.time.values):
            raise ValueError("CRITICAL MISALIGNMENT: ERA5 and CERRA timestamps do not match!")
        
        with open(era5_stats_json, 'r', encoding='utf-8') as f:
            self.era5_stats = json.load(f)["variables"]          
            
        with open(cerra_stats_json, 'r', encoding='utf-8') as f:
            self.cerra_stats = json.load(f)["statistics"]         
            
        with open(orog_stats_json, 'r', encoding='utf-8') as f:
            self.orog_stats = json.load(f)["statistics"]          
            
        self.era5_keys = ['t2m', 'u10', 'v10', 't850', 'd2m']
        
        static_var_key = list(self.static_ds.data_vars)[0]
        static_data = self.static_ds[static_var_key]
        if 'time' in static_data.dims:
            static_data = static_data.isel(time=0)
            
        orog_raw_np = static_data.isel(channel=0).to_numpy().squeeze()
        lsm_raw_np = static_data.isel(channel=1).to_numpy().squeeze()
        
        self.orog_raw = torch.tensor(orog_raw_np, dtype=torch.float32).unsqueeze(0)
        self.lsm_tensor = torch.tensor(lsm_raw_np, dtype=torch.float32).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.era5_ds.time)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        era5_slice = self.era5_ds['era5_data'].isel(time=idx)
        cerra_slice = self.cerra_ds['cerra_data'].isel(time=idx)
        
        weather_tensor = torch.stack([
            torch.tensor(era5_slice.isel(channel=ch_idx).to_numpy(), dtype=torch.float32) 
            for ch_idx in range(5)
        ], dim=0)
        
        target_h = cerra_slice.sizes['latitude'] if 'latitude' in cerra_slice.sizes else 256
        target_w = cerra_slice.sizes['longitude'] if 'longitude' in cerra_slice.sizes else 256

        if self.upsample:
            weather_processed = F.interpolate(
                weather_tensor.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).squeeze(0)
        else:
            weather_processed = weather_tensor

        if torch.isnan(weather_processed).any():
            raise ValueError(f"NaNs detected in ERA5 predictors at index: {idx}")

        for ch_idx, feature_key in enumerate(self.era5_keys):
            mean_val = self.era5_stats[feature_key]['mean']
            std_val = self.era5_stats[feature_key]['std']
            weather_processed[ch_idx] = (weather_processed[ch_idx] - mean_val) / (std_val + 1e-8)

        orog_mean = self.orog_stats['mean']                                
        orog_std = self.orog_stats['std']                                 
        orog_processed = (self.orog_raw - orog_mean) / (orog_std + 1e-8)

        final_input_tensor = torch.cat([weather_processed, orog_processed, self.lsm_tensor], dim=0)

        cerra_np = cerra_slice.isel(channel=0).to_numpy().squeeze()
        cerra_chw = torch.tensor(cerra_np, dtype=torch.float32).unsqueeze(0)
        
        if torch.isnan(cerra_chw).any():
            raise ValueError(f"NaNs detected in CERRA targets at index: {idx}")
        
        cerra_normalized = (cerra_chw - self.cerra_stats['mean']) / self.cerra_stats['std']
        
        return final_input_tensor, cerra_normalized

class FilteredDatasetWrapper(Dataset):
    def __init__(self, base_dataset: NetCDFSpatialDataset, channel_indices: List[int]) -> None:
        self.base_dataset = base_dataset
        self.channel_indices = channel_indices

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        return x[self.channel_indices, :, :], y
