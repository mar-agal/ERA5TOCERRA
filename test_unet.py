import torch
import yaml
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import src.plots as plots_module
from src.data_module import GreeceDownscalingDataModule
from src.models.unet import GreeceDownscalingModule
from src.metrics import calculate_final_metrics_with_timeseries

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    base_output_dir = cfg["savepreds_path"]
    os.makedirs(base_output_dir, exist_ok=True)

    era5_npy_path = os.path.join(base_output_dir, "era5_denorm_test.npy")
    unet_npy_path = os.path.join(base_output_dir, "unet_predictions_denorm_test.npy")
    cerra_npy_path = os.path.join(base_output_dir, "cerra_denorm_test.npy")
    
    your_interp_npy_path = os.path.join(base_output_dir, "bilinear_predictions_denorm_test.npy")

    print("📥 Loading DataModule for Independent Testing...")
    data_module = GreeceDownscalingDataModule(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        upsample=True,
        used_channels=cfg["used_channels"]
    )
    data_module.era5_test_path = "/kaggle/input/datasets/mariaagalioti/dataset/era5_test.nc"
    data_module.cerra_test_path = "/kaggle/input/datasets/mariaagalioti/dataset/cerra_test.nc"
    data_module.era5_stats_json = "/kaggle/input/datasets/mariaagalioti/stats-era-cerra/era5_train_global_stats.json"
    data_module.cerra_stats_json = "/kaggle/input/datasets/mariaagalioti/stats-era-cerra/cerra_train_temp_stats.json"
    data_module.orog_stats_json = "/kaggle/input/datasets/mariaagalioti/stats-era-cerra/orography_stats.json"

    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    total_timesteps = len(data_module.test_dataset)
    H_cerra, W_cerra = 256, 256
    H_era5, W_era5 = 52, 52

    print(f"📦 Pre-allocating NumPy files on disk for {total_timesteps} steps...")
    era5_raw = np.lib.format.open_memmap(era5_npy_path, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_era5, W_era5))
    unet_preds = np.lib.format.open_memmap(unet_npy_path, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_cerra, W_cerra))
    cerra_truth = np.lib.format.open_memmap(cerra_npy_path, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_cerra, W_cerra))

    print("🧠 Loading best trained weights into Single GPU...")
    weights_path = "/kaggle/working/weights/best-baseline-unet.ckpt"
    
    model = GreeceDownscalingModule(
        in_channels=cfg["img_in_channels"], 
        out_channels=cfg["img_out_channels"]
    )
    
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to("cuda")

    print("🎯 Opening original NetCDF file directly to extract exact timestamps...")
    test_nc = xr.open_dataset("/kaggle/input/datasets/mariaagalioti/dataset/era5_test.nc")
    time_values = test_nc.time.values
    test_nc.close()

    TARGET_DATETIMES = ["2021-08-03 15:00:00", "2021-08-10 15:00:00", "2021-02-16 06:00:00", "2021-11-15 09:00:00", "2021-05-15 12:00:00"]
    netcdf_times = pd.DatetimeIndex(time_values)
    selected_timesteps = [int(np.abs(netcdf_times - pd.to_datetime(dt_str)).argmin()) for dt_str in TARGET_DATETIMES]

    print("⏳ Streaming UNet inference directly to disk storage...")
    start_idx = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to("cuda").float(), batch[1].to("cuda").float()
            pred = model(x)
            
            cerra_mean, cerra_std = 289.0581359863281, 8.353236198425293
            era5_mean, era5_std = 289.2669677734375, 8.127739906311035
            
            pred_denorm = (pred * cerra_std) + cerra_mean
            y_denorm = (y * cerra_std) + cerra_mean
            
            x_lowres_256 = (x[:, [0], :, :] * era5_std) + era5_mean
            x_lowres_52 = torch.nn.functional.interpolate(x_lowres_256, size=(52, 52), mode='bilinear', align_corners=False)
            
            end_idx = start_idx + x.shape[0]
            
            era5_raw[start_idx:end_idx] = x_lowres_52.cpu().numpy()
            unet_preds[start_idx:end_idx] = pred_denorm.cpu().numpy()
            cerra_truth[start_idx:end_idx] = y_denorm.cpu().numpy()
            
            era5_raw.flush()
            unet_preds.flush()
            cerra_truth.flush()
            
            start_idx = end_idx
            del x, y, pred, pred_denorm, y_denorm, x_lowres_256, x_lowres_52
            torch.cuda.empty_cache()

    print("💾 Memmap structures closed and safely flushed.")
    del era5_raw, unet_preds, cerra_truth
    gc.collect()

    print("\n📊 Triggering centralized physical evaluation protocol...")
    final_preds = np.load(unet_npy_path, mmap_mode='r')
    final_targets = np.load(cerra_npy_path, mmap_mode='r')
    final_inputs = np.load(era5_npy_path, mmap_mode='r')
    
    if os.path.exists(your_interp_npy_path):
        print("📂 Found your bilinear_predictions_denorm_test.npy! Loading it for comparisons...")
        final_interp = np.load(your_interp_npy_path, mmap_mode='r')
        model_list_plots = [final_interp, final_preds]
        model_names_plots = ["Bilinear-Interp", "UNet-Baseline"]
    else:
        print("⚠️ Warning: bilinear_predictions_denorm_test.npy NOT found in output directory! Plotting UNet only.")
        model_list_plots = [final_preds]
        model_names_plots = ["UNet-Baseline"]
    
    calculate_final_metrics_with_timeseries(preds=final_preds, targets=final_targets, output_dir=base_output_dir)

    print("📈 Generating physical validation charts (PSD & Log-PDF)...")
    plots_module.generate_temperature_diagnostic_plots()

    print("🎬 Rendering specific case study maps up-right...")
    plots_module.print_model_results(
        low_res=final_inputs,
        ground_truth=final_targets,
        models_list=model_list_plots,
        model_names=model_names_plots,
        selected_timesteps=selected_timesteps,
        datetimes_list=TARGET_DATETIMES
    )
    print("\n🎉 HPC TEST PROTOCOL COMPLETE WITH EMBEDDED BILINEAR BASELINE!")
