import xarray as xr
import os
import yaml

# 1. Load configuration
config_path = "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 2. Setup paths from YAML
era5_path = os.path.join(config['paths']['combined'], "ERA5_2010_2021_BCHW.nc")
cerra_path = os.path.join(config['paths']['root'], "Cerra_2010_2021_BCHW.nc")
output_dir = config['paths']['split_output']

os.makedirs(output_dir, exist_ok=True)

# 3. Load data
print("Loading data...")
era5 = xr.open_dataset(era5_path)
cerra = xr.open_dataset(cerra_path)

# 4. Rechunking for efficiency
era5 = era5.chunk({'time': 365})
cerra = cerra.chunk({'time': 365})

# 5. Define splits
splits = {
    "train": slice("2010-01-01", "2019-12-31"),
    "val": slice("2020-01-01", "2020-12-31"),
    "test": slice("2021-01-01", "2021-12-31")
}

# 6. Process and save
for name, period in splits.items():
    print(f"Processing {name}...")
    
    era5_sub = era5.sel(time=period)
    cerra_sub = cerra.sel(time=period)
    
    # Automatically get variable names
    era5_vars = list(era5_sub.data_vars)
    cerra_vars = list(cerra_sub.data_vars)
    
    # Create encoding
    era5_encoding = {var: {'zlib': False} for var in era5_vars}
    cerra_encoding = {var: {'zlib': False} for var in cerra_vars}
    
    # Save to NetCDF
    era5_sub.to_netcdf(os.path.join(output_dir, f"era5_{name}.nc"), encoding=era5_encoding)
    cerra_sub.to_netcdf(os.path.join(output_dir, f"cerra_{name}.nc"), encoding=cerra_encoding)
    
    print(f"✅ {name} processing complete.")

era5.close()
cerra.close()
print("All splitting tasks finished successfully.")
