import sys
import pathlib
import gc
import numpy as np
import xarray as xr
import pandas as pd

current_dir = pathlib.Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.metrics import calculate_final_metrics_with_timeseries
import src.plots as plots_module

INPUT_DIR = pathlib.Path("/kaggle/input/datasets/mariaagalioti/dataset")
ERA5_TEST_PATH = INPUT_DIR / "era5_test.nc"
CERRA_TEST_PATH = INPUT_DIR / "cerra_test.nc"

BASE_DIR = pathlib.Path("/kaggle/working/outputs")
BASE_DIR.mkdir(parents=True, exist_ok=True)

BILINEAR_PATH = BASE_DIR / "bilinear_predictions_denorm_test.npy"
CERRA_NPY_PATH = BASE_DIR / "cerra_denorm_test.npy"
ERA5_NPY_PATH = BASE_DIR / "era5_denorm_test.npy"

TARGET_DATETIMES = [
    "2021-08-03 15:00:00",  
    "2021-08-10 15:00:00",
    "2021-02-16 06:00:00",
    "2021-11-15 09:00:00",
    "2021-05-15 12:00:00"
]

def run_interpolation_pipeline():
    print("📥 Opening NetCDF structures with strict time chunking...")
    ds_era5 = xr.open_dataset(ERA5_TEST_PATH, chunks={"time": 10})
    ds_cerra = xr.open_dataset(CERRA_TEST_PATH, chunks={"time": 10})

    era5_var = "t2m" if "t2m" in ds_era5.data_vars else list(ds_era5.data_vars)[0]
    cerra_var = "t2m" if "t2m" in ds_cerra.data_vars else list(ds_cerra.data_vars)[0]

    total_timesteps = len(ds_era5.time)
    
    lat_target = ds_cerra.lat.values if 'lat' in ds_cerra.coords else ds_cerra.latitude.values
    lon_target = ds_cerra.lon.values if 'lon' in ds_cerra.coords else ds_cerra.longitude.values
    H_cerra, W_cerra = lat_target.shape[0], lon_target.shape[0]
    
    era5_raw_var = ds_era5[era5_var]
    if "channel" in era5_raw_var.dims: era5_raw_var = era5_raw_var.isel(channel=0)
    elif "variable" in era5_raw_var.dims: era5_raw_var = era5_raw_var.isel(variable=0)
    
    lat_dim = 'lat' if 'lat' in era5_raw_var.coords else 'latitude'
    lon_dim = 'lon' if 'lon' in era5_raw_var.coords else 'longitude'
    H_era5, W_era5 = era5_raw_var[lat_dim].shape[0], era5_raw_var[lon_dim].shape[0]

    print("📦 Pre-allocating standard NumPy files on disk (open_memmap)...")
    bilinear_preds = np.lib.format.open_memmap(BILINEAR_PATH, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_cerra, W_cerra))
    cerra_truth = np.lib.format.open_memmap(CERRA_NPY_PATH, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_cerra, W_cerra))
    era5_raw = np.lib.format.open_memmap(ERA5_NPY_PATH, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_era5, W_era5))

    print("💡 Mapping requested Gregorian datetimes for case studies...")
    selected_timesteps = []
    netcdf_times = pd.DatetimeIndex(ds_era5.time.values)
    for dt_str in TARGET_DATETIMES:
        idx = np.abs(netcdf_times - pd.to_datetime(dt_str)).argmin()
        selected_timesteps.append(int(idx))
        print(f"🎯 Matched {dt_str} ──> Index {idx}")

    era5_data = ds_era5[era5_var]
    if "channel" in era5_data.dims: era5_data = era5_data.isel(channel=0)
    elif "variable" in era5_data.dims: era5_data = era5_data.isel(variable=0)
    
    cerra_data = ds_cerra[cerra_var]

    print(f"⏳ Running HPC Streaming Pipeline over {total_timesteps} steps...")
    batch_size = 50

    for start_idx in range(0, total_timesteps, batch_size):
        end_idx = min(start_idx + batch_size, total_timesteps)
        
        era5_batch = era5_data.isel(time=slice(start_idx, end_idx))
        cerra_batch = cerra_data.isel(time=slice(start_idx, end_idx))
        
        if 'lat' in ds_cerra.coords:
            interp_batch = era5_batch.interp(lat=lat_target, lon=lon_target, method="linear")
        else:
            interp_batch = era5_batch.interp(latitude=lat_target, longitude=lon_target, method="linear")
            
        interp_batch = interp_batch.compute()
        cerra_batch = cerra_batch.compute()
        era5_batch = era5_batch.compute()
        
        vals_b = np.asarray(interp_batch, dtype=np.float32)
        bilinear_preds[start_idx:end_idx] = np.expand_dims(vals_b, axis=1) if vals_b.ndim == 3 else vals_b
        
        vals_c = np.asarray(cerra_batch, dtype=np.float32)
        cerra_truth[start_idx:end_idx] = np.expand_dims(vals_c, axis=1) if vals_c.ndim == 3 else vals_c
        
        vals_e = np.asarray(era5_batch, dtype=np.float32)
        era5_raw[start_idx:end_idx] = np.expand_dims(vals_e, axis=1) if vals_e.ndim == 3 else vals_e

        bilinear_preds.flush()
        cerra_truth.flush()
        era5_raw.flush()

        del era5_batch, cerra_batch, interp_batch, vals_b, vals_c, vals_e
        gc.collect()

    print("💾 Valid NumPy files (.npy) successfully locked onto disk.")
    ds_era5.close()
    ds_cerra.close()

    del bilinear_preds, cerra_truth, era5_raw
    gc.collect()

    print("\n🧮 Launching metrics calculation modules...")
    # 🔥 ΕΔΩ ΕΙΝΑΙ Η ΜΕΓΑΛΗ ΑΛΛΑΓΗ: Επιβάλλουμε το "bilinear" για να μην παίρνει το default unet!
    calculate_final_metrics_with_timeseries(preds=np.load(BILINEAR_PATH), targets=np.load(CERRA_NPY_PATH), output_dir=str(BASE_DIR), model_name="bilinear")

    print("\n📊 Launching physical validation charts (PSD & Log-PDF)...")
    plots_module.generate_temperature_diagnostic_plots()

    print("\n🎬 Rendering specific case study maps...")
    low_res_arr = np.load(ERA5_NPY_PATH)
    gt_arr = np.load(CERRA_NPY_PATH)
    pred_arr = np.load(BILINEAR_PATH)
    
    plots_module.print_model_results(
        low_res=low_res_arr,
        ground_truth=gt_arr,
        models_list=[pred_arr],
        model_names=["Bilinear-Interpolation"],
        selected_timesteps=selected_timesteps,
        datetimes_list=TARGET_DATETIMES
    )

if __name__ == '__main__':
    run_interpolation_pipeline()
