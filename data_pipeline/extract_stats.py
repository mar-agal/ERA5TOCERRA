# ==============================================================================
# Format: BCHW | Environment: Kaggle (GPU/Multi-GPU Powered)
# Unified Pipeline: CERRA (GPU) -> ERA5 (Dask-Lazy) -> Orography (Static)
# ==============================================================================

import os
import json
import xarray as xr
import torch
import dask  

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ==============================================================================
# PART 1: CERRA DATASET (GPU POWERED)
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Execution Device: {device.type.upper()}")
if device.type == "cpu":
    print(" WARNING: No GPU detected! Running on CPU fallback.")

CERRA_TRAIN_PATH = "/kaggle/input/datasets/mariaagalioti/cerra-train/cerra_train.nc"
CERRA_JSON_PATH = "/kaggle/working/cerra_train_temp_stats.json"

if not os.path.exists(CERRA_TRAIN_PATH):
    raise FileNotFoundError(f" Training file not found at {CERRA_TRAIN_PATH}")

try:
    ds_cerra = xr.open_dataset(CERRA_TRAIN_PATH)
    var_name_cerra = "cerra_data" if "cerra_data" in ds_cerra.data_vars else list(ds_cerra.data_vars)[0]
    
    print(f"\n Dataset Shape (CERRA): {ds_cerra[var_name_cerra].shape}")
    print(" Loading Channel Index 0 into Host Memory...")
    raw_numpy = ds_cerra[var_name_cerra].isel(channel=0).values
except Exception as e:
    print(f" Failed to load NetCDF array: {str(e)}")
    raise

print(f" Moving data to {device.type.upper()} VRAM...")
temp_tensor = torch.tensor(raw_numpy, dtype=torch.float32, device=device)

print(" Calculating global metrics on GPU cores...")
global_mean = float(torch.mean(temp_tensor).item())
global_max  = float(torch.max(temp_tensor).item())
global_min  = float(torch.min(temp_tensor).item())
global_std  = float(torch.std(temp_tensor).item())

total_times_c = temp_tensor.shape[0]
w_dim, h_dim = temp_tensor.shape[1], temp_tensor.shape[2]

cerra_payload = {
    "dataset_metadata": {
        "filename": os.path.basename(CERRA_TRAIN_PATH),
        "target_variable": var_name_cerra,
        "channel_index": 0,
        "total_timesteps": int(total_times_c),
        "spatial_resolution": f"{w_dim}x{h_dim}"
    },
    "statistics": {
        "mean": global_mean,
        "max": global_max,
        "min": global_min,
        "std": global_std
    }
}

with open(CERRA_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(cerra_payload, f, indent=4)
print(f" CERRA JSON saved at {CERRA_JSON_PATH}")

ds_cerra.close()
del temp_tensor
del raw_numpy
torch.cuda.empty_cache()


# ==============================================================================
# PART 2: ERA5 DATASET (LAZY-STREAMING VIA DASK)
# ==============================================================================

ERA5_TRAIN_PATH = "/kaggle/input/datasets/mariaagalioti/era5-train/era5_train.nc"
ERA5_JSON_PATH = "/kaggle/working/era5_train_global_stats.json"

if not os.path.exists(ERA5_TRAIN_PATH):
    raise FileNotFoundError(f" ERA5 Training file not found at {ERA5_TRAIN_PATH}")

print("\n Opening ERA5 dataset with Lazy Dask Chunks (Time Block Size: 500)...")
ds_era5 = xr.open_dataset(ERA5_TRAIN_PATH, chunks={"time": 500})

var_name_era5 = "era5_data" if "era5_data" in ds_era5.data_vars else list(ds_era5.data_vars)[0]
tensor_pointer = ds_era5[var_name_era5]

shape_era5 = tensor_pointer.shape
total_times_e = shape_era5[0]
total_channels = shape_era5[1]
lat_dim, lon_dim = shape_era5[2], shape_era5[3]

channel_names = [str(c.values) for c in ds_era5.channel] if "channel" in ds_era5.coords else [f"channel_{i}" for i in range(total_channels)]

print(" Compiling mathematical operations into an optimized Dask Graph...")
mean_lazy = tensor_pointer.mean(dim=["time", "latitude", "longitude"])
max_lazy  = tensor_pointer.max(dim=["time", "latitude", "longitude"])
min_lazy  = tensor_pointer.min(dim=["time", "latitude", "longitude"])
std_lazy  = tensor_pointer.std(dim=["time", "latitude", "longitude"])

print(" Triggering Dask Streaming Engine (Shared Pass over Disk Blocks)...")
mean_res, max_res, min_res, std_res = dask.compute(mean_lazy, max_lazy, min_lazy, std_lazy)

means, maxs, mins, stds = mean_res.values, max_res.values, min_res.values, std_res.values

variables_statistics = {}
for i, name in enumerate(channel_names):
    variables_statistics[name] = {
        "channel_index": i,
        "mean": float(means[i]),
        "max": float(maxs[i]),
        "min": float(mins[i]),
        "std": float(stds[i])
    }

era5_payload = {
    "dataset_metadata": {
        "filename": os.path.basename(ERA5_TRAIN_PATH),
        "target_variable": var_name_era5,
        "total_timesteps": int(total_times_e),
        "spatial_resolution": f"{lat_dim}x{lon_dim}",
        "total_variables": total_channels
    },
    "variables": variables_statistics
}

with open(ERA5_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(era5_payload, f, indent=4)
print(f" ERA5 JSON saved at {ERA5_JSON_PATH}")

ds_era5.close()


# ==============================================================================
# PART 3: OROGRAPHY (STATIC DATA)
# ==============================================================================

OROG_INPUT_PATH = "/kaggle/input/datasets/mariaagalioti/static-data/CERRA_Greece_Static_B1_CHW.nc"
OROG_OUTPUT_JSON = "/kaggle/working/orography_stats.json"

if not os.path.exists(OROG_INPUT_PATH):
    raise FileNotFoundError(f" Orography file not found at {OROG_INPUT_PATH}")

print(f"\n Processing Orography from {OROG_INPUT_PATH}...")
ds_oro = xr.open_dataset(OROG_INPUT_PATH)
var_name_oro = list(ds_oro.data_vars)[0]
data_oro = ds_oro[var_name_oro].isel(channel=0)

orog_payload = {
    "dataset_metadata": {
        "filename": os.path.basename(OROG_INPUT_PATH),
        "target_variable": var_name_oro,
        "spatial_layout": "W x H"
    },
    "statistics": {
        "mean": float(data_oro.mean().item()),
        "max": float(data_oro.max().item()),
        "min": float(data_oro.min().item()),
        "std": float(data_oro.std().item()),
        "med": float(data_oro.median().item())
    }
}

with open(OROG_OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(orog_payload, f, indent=4)

print(f" Orography statistics saved at {OROG_OUTPUT_JSON}")
ds_oro.close()

print("\n" + "="*60)
print(" ALL PIPELINES EXECUTED SUCCESSFULLY WITHOUT ERRORS!")
print("="*60)