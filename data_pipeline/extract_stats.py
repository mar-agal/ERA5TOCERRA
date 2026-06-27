# ==============================================================================
# Format: BCHW | Environment: Kaggle (GPU/Multi-GPU Powered)
# ==============================================================================

import os
import json
import xarray as xr
import torch

# Force safe multi-process I/O access on Kaggle
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ------------------------------------------------------------------------------
# 1. DEVICE CHECK & CONFIGURATION
# ------------------------------------------------------------------------------
# Automatically detect and select the first available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Execution Device: {device.type.upper()}")

if device.type == "cpu":
    print(" WARNING: No GPU detected! Running on CPU fallback.")

CERRA_TRAIN_PATH = "/kaggle/input/datasets/mariaagalioti/cerra-train/cerra_train.nc"
OUTPUT_JSON_PATH = "/kaggle/working/cerra_train_temp_stats.json"

# ------------------------------------------------------------------------------
# 2. LAZY NETCDF INITIALIZATION
# ------------------------------------------------------------------------------
if not os.path.exists(CERRA_TRAIN_PATH):
    raise FileNotFoundError(f" Training file not found at {CERRA_TRAIN_PATH}")

try:
    # Open dataset without chunks first to get raw numpy arrays for PyTorch
    ds = xr.open_dataset(CERRA_TRAIN_PATH)
    var_name = "cerra_data" if "cerra_data" in ds.data_vars else list(ds.data_vars)[0]
    
    print(f" Dataset Shape: {ds[var_name].shape} (Dimensions: {ds[var_name].dims})")
    print("  Loading Channel Index 0 into Host Memory...")
    
    # Extract only Channel 0 as a raw numpy array (RAM-safe because it's just 1 channel)
    raw_numpy = ds[var_name].isel(channel=0).values

except Exception as e:
    print(f" Failed to load NetCDF array: {str(e)}")
    raise

# ------------------------------------------------------------------------------
# 3. GPU POWERED STATISTICAL COMPUTATION
# ------------------------------------------------------------------------------
print(f" Moving 5.7 Billion data points to {device.type.upper()} VRAM...")

# Convert to PyTorch Tensor and push directly to GPU
# We use float32 as it is perfectly optimized for GPU cores
temp_tensor = torch.tensor(raw_numpy, dtype=torch.float32, device=device)

print(" Calculating global metrics on GPU cores...")

# Fast parallel operations on GPU
global_mean = float(torch.mean(temp_tensor).item())
global_max  = float(torch.max(temp_tensor).item())
global_min  = float(torch.min(temp_tensor).item())
global_std  = float(torch.std(temp_tensor).item())

# ------------------------------------------------------------------------------
# 4. JSON METADATA EXPORT
# ------------------------------------------------------------------------------
total_times = temp_tensor.shape[0]
w_dim, h_dim = temp_tensor.shape[1], temp_tensor.shape[2]

metrics_payload = {
    "dataset_metadata": {
        "filename": os.path.basename(CERRA_TRAIN_PATH),
        "target_variable": var_name,
        "channel_index": 0,
        "channel_coordinate": "channel",
        "total_timesteps": int(total_times),
        "spatial_resolution": f"{w_dim}x{h_dim}"
    },
    "statistics": {
        "mean": global_mean,
        "max": global_max,
        "min": global_min,
        "std": global_std
    }
}

print(f" Saving configurations to JSON...")
try:
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(metrics_payload, json_file, indent=4)
        
    print("\n" + "="*60)
    print(" GPU COMPUTATION COMPLETE")
    print("="*60)
    print(f"🔹 Mean Temperature:       {metrics_payload['statistics']['mean']:.4f} K")
    print(f"🔹 Max Temperature:        {metrics_payload['statistics']['max']:.4f} K")
    print(f"🔹 Min Temperature:        {metrics_payload['statistics']['min']:.4f} K")
    print(f"🔹 Standard Deviation:     {metrics_payload['statistics']['std']:.4f} K")
    print("="*60)
    print(f" SUCCESS: JSON saved at {OUTPUT_JSON_PATH}")
    print("="*60)
except Exception as e:
    print(f" Failed to export JSON file: {str(e)}")

# Clear VRAM cache manually just to be safe for your next cells
# ==============================================================================
# OPTIMIZED LAZY-STREAMING MULTI-VARIABLE STATISTICS EXTRACTION FOR ERA5
# Format: BCHW | Dimensions: ('time', 'channel', 'latitude', 'longitude')
# Pipeline: High-Performance Shared Dask Graph Optimization
# ==============================================================================



# Force safe multi-process I/O access on Kaggle
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ------------------------------------------------------------------------------
# 1. PATHS DEFINITION
# ------------------------------------------------------------------------------
ERA5_TRAIN_PATH = "/kaggle/input/datasets/mariaagalioti/era5-train/era5_train.nc"
OUTPUT_JSON_PATH = "/kaggle/working/era5_train_global_stats.json"

# ------------------------------------------------------------------------------
# 2. LAZY NETCDF INITIALIZATION (CHUNKING ON TIME AXIS)
# ------------------------------------------------------------------------------
if not os.path.exists(ERA5_TRAIN_PATH):
    raise FileNotFoundError(f" ERA5 Training file not found at {ERA5_TRAIN_PATH}")

print(" Opening dataset with Lazy Dask Chunks (Time Block Size: 500)...")
ds = xr.open_dataset(ERA5_TRAIN_PATH, chunks={"time": 500})

var_name = "era5_data" if "era5_data" in ds.data_vars else list(ds.data_vars)[0]
tensor_pointer = ds[var_name]

shape = tensor_pointer.shape
print(f" Target Data Matrix Shape: {shape} (Dims: {tensor_pointer.dims})")

total_times = shape[0]
total_channels = shape[1]
lat_dim, lon_dim = shape[2], shape[3]

channel_names = []
if "channel" in ds.coords:
    channel_names = [str(c.values) for c in ds.channel]
else:
    channel_names = [f"channel_{i}" for i in range(total_channels)]

print(f"  Found Variables to process: {channel_names}")

# ------------------------------------------------------------------------------
# 3. DEFINE LAZY TASK GRAPH (NO COMPUTATION HAPPENING HERE)
# ------------------------------------------------------------------------------
print("\n📝 Compiling mathematical operations into an optimized Dask Graph...")

mean_lazy = tensor_pointer.mean(dim=["time", "latitude", "longitude"])
max_lazy  = tensor_pointer.max(dim=["time", "latitude", "longitude"])
min_lazy  = tensor_pointer.min(dim=["time", "latitude", "longitude"])
std_lazy  = tensor_pointer.std(dim=["time", "latitude", "longitude"])

# ------------------------------------------------------------------------------
# 4. EXECUTE SHARED GRAPH COMPUTE (SINGLE-PASS STREAMING)
# ------------------------------------------------------------------------------
print(" Triggering Dask Streaming Engine (1 Shared Pass over Disk Blocks)...")

# FIXED: We use dask.compute() to join Xarray lazy objects seamlessly!
mean_res, max_res, min_res, std_res = dask.compute(
    mean_lazy, 
    max_lazy, 
    min_lazy, 
    std_lazy
)

# Extract raw numpy arrays from calculated results
means = mean_res.values
maxs  = max_res.values
mins  = min_res.values
stds  = std_res.values

print("✅ Shared computation pipeline completed successfully!")

# ------------------------------------------------------------------------------
# 5. PARSE DATA PROFILE & JSON ARTIFACT EXPORT
# ------------------------------------------------------------------------------
variables_statistics = {}
for i, name in enumerate(channel_names):
    variables_statistics[name] = {
        "channel_index": i,
        "mean": float(means[i]),
        "max": float(maxs[i]),
        "min": float(mins[i]),
        "std": float(stds[i])
    }

metrics_payload = {
    "dataset_metadata": {
        "filename": os.path.basename(ERA5_TRAIN_PATH),
        "target_variable": var_name,
        "total_timesteps": int(total_times),
        "spatial_resolution": f"{lat_dim}x{lon_dim}",
        "total_variables": total_channels
    },
    "variables": variables_statistics
}

print(f"\n💾 Serializing profiles to configuration JSON...")
try:
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(metrics_payload, json_file, indent=4)
        
    print("\n" + "="*70)
    print(" ERA5 MULTI-VARIABLE EXTRACTOR COMPLETE (DASK-LAZY)")
    print("="*70)
    for var, stats in metrics_payload["variables"].items():
        print(f" Variable: {var:<15} | Mean: {stats['mean']:>10.4f} | Std: {stats['std']:>10.4f}")
    print("="*70)
    print(f" SUCCESS: JSON parameter map generated at: {OUTPUT_JSON_PATH}")
    print("="*70)
except Exception as e:
    print(f" Failed to export data profile: {str(e)}")

ds.close()
del temp_tensor
torch.cuda.empty_cache()
ds.close()

OROG_INPUT_PATH = "/kaggle/input/datasets/mariaagalioti/static-data/CERRA_Greece_Static_B1_CHW.nc"
OROG_OUTPUT_JSON = "/kaggle/working/orography_stats.json"

print(f"\nProcessing Orography from {OROG_INPUT_PATH}...")
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

print(f"✅ Orography statistics saved at {OROG_OUTPUT_JSON}")
ds_oro.close()
