import os
import pathlib
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

var_name = "t2m"
base_dir = pathlib.Path("/kaggle/working/outputs")
cerra_path = base_dir / "cerra_denorm_test.npy"
era5_path = base_dir / "era5_denorm_test.npy"

model_paths: Dict[str, pathlib.Path] = {
    "Bilinear-Interpolation": base_dir / "bilinear_predictions_denorm_test.npy",
    "UNet-Baseline"         : base_dir / "unet_predictions_denorm_test.npy",
    "UNet-PSD"              : base_dir / "unet_h_psd_predictions_denorm_test.npy"
}

era5_dx_deg = 0.25                    
n_bins = 200                          
eps = 1e-12                           
out_dir = pathlib.Path("/kaggle/working/outputs/plots")
out_dir.mkdir(parents=True, exist_ok=True)

def get_psd(data: np.ndarray, dx: float, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    data = np.moveaxis(data, axis, -1)
    n = data.shape[-1]
    fft = np.fft.rfft(data, axis=-1) / n
    psd = 2.0 * (np.abs(fft) ** 2)
    k = np.fft.rfftfreq(n, d=dx)
    return k, psd

def load_stack(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        print(f"Warning: File path {path.name} was not discovered. Skipping entry.")
        return None
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    return arr

def generate_temperature_diagnostic_plots() -> None:
    print("Loading baseline ground truth references (CERRA & ERA5)...")
    cerra = load_stack(cerra_path)
    era5 = load_stack(era5_path)
    
    if cerra is None or era5 is None:
        raise FileNotFoundError("Critical Error: Missing core verification stacks (CERRA or ERA5).")
        
    n_ref_lon = cerra.shape[2]
    dx_ref = era5_dx_deg * (era5.shape[2] / n_ref_lon)

    print("Loading predictive candidate stacks dynamically...")
    model_stacks = {}
    for name, path in model_paths.items():
        stack = load_stack(path)
        if stack is not None:
            model_stacks[name] = stack

    print(f"\nConstructing Spatial Spectral & Climatological Diagnostics for: {var_name}")

    # ──────────────────────────────────────────────────────────────────
    # PSD PLOT (CERRA and Models Only - Excluding ERA5)
    # ──────────────────────────────────────────────────────────────────
    k_ref, psd_ref = get_psd(cerra, dx_ref, axis=2)
    psd_ref_mean = psd_ref.mean(axis=1).mean(axis=0)
    
    plt.figure(figsize=(7.5, 5.5))
    plt.loglog(k_ref, psd_ref_mean, lw=3, c="black", label="CERRA (Ground Truth)")
    for name, stack in model_stacks.items():
        _, psd_m = get_psd(stack, dx_ref, axis=2)
        psd_m_mean = psd_m.mean(axis=1).mean(axis=0)
        plt.loglog(k_ref, psd_m_mean, label=name, alpha=0.85, lw=1.5)
            
    plt.xlabel(r"Wavenumber $k$ (cycles deg$^{-1}$)", fontsize=11)
    plt.ylabel("Power Spectral Density (PSD)", fontsize=11)
    plt.title(f"PSD of {var_name}", fontsize=11, fontweight="bold")
    plt.legend(loc="lower left", frameon=True, facecolor="white")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{var_name}_psd_comparison.png", dpi=250, bbox_inches="tight")
    plt.close()

    # ──────────────────────────────────────────────────────────────────
    # FREQUENCY DISTRIBUTION PLOT (Fixed X-axis Alignment)
    # ──────────────────────────────────────────────────────────────────
    bin_width = 0.5
    
    raw_min_k = cerra.min()
    raw_max_k = cerra.max()
    
    raw_min_c = raw_min_k - 273.15
    raw_max_c = raw_max_k - 273.15
    
    bin_start_c = np.floor(raw_min_c / bin_width) * bin_width
    bin_end_c = np.ceil(raw_max_c / bin_width) * bin_width
    
    bins_c = np.arange(bin_start_c, bin_end_c + bin_width, bin_width)
    bins = bins_c + 273.15
    
    flat_ref = cerra.ravel()
    counts_ref, edges = np.histogram(flat_ref, bins=bins, density=False)
    centres_ref = 0.5 * (edges[:-1] + edges[1:])
    
    plt.figure(figsize=(7.5, 5.0))
    
    log_counts_ref = np.log(counts_ref + 1)
    plt.plot(centres_ref, log_counts_ref, lw=3, c="black", label="CERRA (Ground Truth)")
    
    for name, stack in model_stacks.items():
        counts_m, _ = np.histogram(stack.ravel(), bins=bins, density=False)
        plt.plot(centres_ref, np.log(counts_m + 1), label=name, alpha=0.85, lw=1.5)
        
    plt.xlabel(f"Temperature values ({var_name} in Kelvin Scale)", fontsize=11)
    plt.ylabel("Log(freq distrib)", fontsize=11)
    plt.title("2-m Temperature", fontsize=11, fontweight="bold")
    
    tick_start = np.ceil(bin_start_c / 10) * 10 + 273.15
    tick_end = np.floor(bin_end_c / 10) * 10 + 273.15
    plt.xticks(np.arange(tick_start, tick_end + 1, 10))
    
    plt.xlim(centres_ref[0], centres_ref[-1])
    plt.ylim(0.5, 16.5)
    
    plt.legend(ncol=1, loc="lower center", frameon=True, facecolor="white")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{var_name}_logpdf_comparison.png", dpi=250, bbox_inches="tight")
    plt.close()

def print_model_results(low_res, ground_truth, models_list, model_names=None, selected_timesteps=None, datetimes_list=None):
    if not isinstance(models_list, list):
        models_list = [models_list]
    if model_names is None:
        model_names = [f"predicted_{idx}" for idx in range(len(models_list))]
    if selected_timesteps is None:
        selected_timesteps = list(range(low_res.shape[0]))
        
    num_cols = 2 + len(models_list)

    for idx_in_list, i in enumerate(selected_timesteps):
        time_label = datetimes_list[idx_in_list] if (datetimes_list is not None and idx_in_list < len(datetimes_list)) else f"Timestep {i}"
        
        fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 5, 5.5))
        if num_cols == 1:
            axes = [axes]
            
        ground_truth_c = ground_truth[i, 0, :, : ] - 273.15
        
        if low_res.ndim == 4:
            low_res_c = low_res[i, 0, :, :] - 273.15
        else:
            low_res_c = low_res[i, :, :] - 273.15
        
        all_maps_in_row = [low_res_c, ground_truth_c]
        for model_arr in models_list:
            all_maps_in_row.append(model_arr[i, 0, :, :] - 273.15)
            
        v_min = min(m.min() for m in all_maps_in_row)
        v_max = max(m.max() for m in all_maps_in_row)

        ax = axes[0]
        ax.imshow(low_res_c, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
        ax.axis('off')
        ax.set_title(f"{time_label}\nLast Low Res (ERA5)")
        
        col_idx = 1
        for model_arr, name in zip(models_list, model_names):
            ax = axes[col_idx]
            model_c = model_arr[i, 0, :, :] - 273.15
            ax.imshow(model_c, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
            ax.axis('off')
            ax.set_title(f"{time_label}\n{name}")
            col_idx += 1
            
        ax = axes[col_idx]
        im = ax.imshow(ground_truth_c, cmap='viridis', origin='lower', vmin=v_min, vmax=v_max)
        ax.axis('off')
        ax.set_title(f"{time_label}\nGround Truth (CERRA)")       

        plt.tight_layout()
        
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', location='bottom', pad=0.05, shrink=0.85, fraction=0.046)
        cbar.set_label(r'Temperature ($T_{2m}$ in $^\circ$C)', fontsize=11, fontweight='bold')
        
        safe_time_str = str(time_label).replace(" ", "_").replace(":", "-")
        save_path = out_dir / f"case_study_{safe_time_str}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        
    print(f"Saved {len(selected_timesteps)} case study maps with Gregorian headers.")
