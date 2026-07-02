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
    unet_psd_npy_path = os.path.join(base_output_dir, "unet_h_psd_predictions_denorm_test.npy") 
    cerra_npy_path = os.path.join(base_output_dir, "cerra_denorm_test.npy")
    your_interp_npy_path = os.path.join(base_output_dir, "bilinear_predictions_denorm_test.npy")

    print(" Loading DataModule for Independent Testing...")
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

    print(f" Pre-allocating NumPy files on disk for {total_timesteps} steps...")
    era5_raw = np.lib.format.open_memmap(era5_npy_path, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_era5, W_era5))
    unet_preds = np.lib.format.open_memmap(unet_npy_path, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_cerra, W_cerra))
    unet_psd_preds = np.lib.format.open_memmap(unet_psd_npy_path, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_cerra, W_cerra)) 
    cerra_truth = np.lib.format.open_memmap(cerra_npy_path, dtype='float32', mode='w+', shape=(total_timesteps, 1, H_cerra, W_cerra))

    cerra_mean, cerra_std = 289.0581359863281, 8.353236198425293
    era5_mean, era5_std = 289.2669677734375, 8.127739906311035

    # =========================================================================
    #  RUN 1: INFERENCE FOR BASELINE UNET
    # =========================================================================
    print(" Loading baseline UNet weights into GPU...")
    weights_path_baseline = "/kaggle/input/datasets/mariaagalioti/wieghts/best-baseline-unet.ckpt" 
    
    model = GreeceDownscalingModule(in_channels=cfg["img_in_channels"], out_channels=cfg["img_out_channels"])
    checkpoint = torch.load(weights_path_baseline, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to("cuda")

    print(" Streaming Baseline UNet inference directly to disk storage...")
    start_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to("cuda").float(), batch[1].to("cuda").float()
            pred = model(x)
            
            pred_denorm = (pred * cerra_std) + cerra_mean
            y_denorm = (y * cerra_std) + cerra_mean
            x_lowres_256 = (x[:, [0], :, :] * era5_std) + era5_mean
            x_lowres_52 = torch.nn.functional.interpolate(x_lowres_256, size=(52, 52), mode='bilinear', align_corners=False)
            
            end_idx = start_idx + x.shape[0]
            era5_raw[start_idx:end_idx] = x_lowres_52.cpu().numpy()
            unet_preds[start_idx:end_idx] = pred_denorm.cpu().numpy()
            cerra_truth[start_idx:end_idx] = y_denorm.cpu().numpy()
            
            start_idx = end_idx
            del x, y, pred, pred_denorm, y_denorm, x_lowres_256, x_lowres_52
    
    era5_raw.flush()
    unet_preds.flush()
    cerra_truth.flush()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    #  RUN 2: INFERENCE FOR NEW UNET (UNET-PSD)
    # =========================================================================
    print(" Loading NEW UNet-PSD weights into GPU...")
    weights_path_psd = "/kaggle/input/datasets/mariaagal/weights-unet-best/best-unet-psd-finetuned.ckpt"
    
    model = GreeceDownscalingModule(in_channels=cfg["img_in_channels"], out_channels=cfg["img_out_channels"])
    checkpoint = torch.load(weights_path_psd, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to("cuda")

    print(" Streaming UNet-PSD inference directly to disk storage...")
    start_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to("cuda").float()
            pred = model(x)
            pred_denorm = (pred * cerra_std) + cerra_mean
            
            end_idx = start_idx + x.shape[0]
            unet_psd_preds[start_idx:end_idx] = pred_denorm.cpu().numpy()
            
            start_idx = end_idx
            del x, pred, pred_denorm
    
    unet_psd_preds.flush()
    
    print(" Memmap structures closed and safely flushed.")
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # 📊 EVALUATION & PLOTS
    # =========================================================================
    print("\n Triggering centralized physical evaluation protocol...")
    final_preds = np.load(unet_npy_path, mmap_mode='r')
    final_psd_preds = np.load(unet_psd_npy_path, mmap_mode='r') 
    final_targets = np.load(cerra_npy_path, mmap_mode='r')
    final_inputs = np.load(era5_npy_path, mmap_mode='r')
    
    # Load the baseline bilinear interpolation array
    if os.path.exists(your_interp_npy_path):
        final_bilinear = np.load(your_interp_npy_path, mmap_mode='r')
        maps_models = [final_bilinear, final_preds, final_psd_preds]
        maps_names = ["Bilinear-Interpolation", "UNet-Baseline", "UNet-PSD"]
    else:
        print("Warning: Bilinear interpolation file not found. Falling back to 4 plots.")
        maps_models = [final_preds, final_psd_preds]
        maps_names = ["UNet-Baseline", "UNet-PSD"]
    
    calculate_final_metrics_with_timeseries(preds=final_preds, targets=final_targets, output_dir=base_output_dir, model_name="unet_baseline")
    calculate_final_metrics_with_timeseries(preds=final_psd_preds, targets=final_targets, output_dir=base_output_dir, model_name="unet_psd") 

    print(" Generating physical validation charts (PSD & Log-PDF)...")
    plots_module.generate_temperature_diagnostic_plots()

    print(" Rendering specific case study maps up-right...")
    TARGET_DATETIMES = ["2021-08-03 15:00:00", "2021-08-10 15:00:00", "2021-02-16 06:00:00", "2021-02-17 03:00:00", "2021-11-15 09:00:00", "2021-05-15 12:00:00"]
    test_nc = xr.open_dataset("/kaggle/input/datasets/mariaagalioti/dataset/era5_test.nc")
    netcdf_times = pd.DatetimeIndex(test_nc.time.values)
    test_nc.close()
    selected_timesteps = [int(np.abs(netcdf_times - pd.to_datetime(dt_str)).argmin()) for dt_str in TARGET_DATETIMES]

    plots_module.print_model_results(
        low_res=final_inputs,
        ground_truth=final_targets,
        models_list=maps_models,   
        model_names=maps_names,
        selected_timesteps=selected_timesteps,
        datetimes_list=TARGET_DATETIMES
    )
    print("\n HPC TEST PROTOCOL COMPLETE WITH BOTH UNET MODELS!")
