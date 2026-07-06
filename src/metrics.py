import os
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

def calculate_final_metrics_with_timeseries(preds: np.ndarray, targets: np.ndarray, output_dir: str, model_name: str = "unet") -> list:
    if preds.ndim == 4: preds = preds.squeeze(1)
    if targets.ndim == 4: targets = targets.squeeze(1)

    total_timesteps = preds.shape[0]

    mae = float(np.mean(np.abs(preds - targets)))
    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(preds - targets))

    global_min = min(float(targets.min()), float(preds.min()))
    global_max = max(float(targets.max()), float(preds.max()))
    global_denom = global_max - global_min if (global_max - global_min) > 0 else 1e-8

    ssim_time_series = []
    psnr_time_series = []

    for i in range(total_timesteps):
        targets_norm = (targets[i] - global_min) / global_denom
        preds_norm = (preds[i] - global_min) / global_denom

        ssim_val = ssim_fn(targets_norm, preds_norm, data_range=1.0)
        ssim_time_series.append(ssim_val)

        timestep_norm_mse = np.mean((preds_norm - targets_norm) ** 2)
        psnr_val = 10 * np.log10(1.0 / (timestep_norm_mse + 1e-8))
        psnr_time_series.append(psnr_val)

    mean_ssim = float(np.mean(ssim_time_series))
    mean_psnr = float(np.mean(psnr_time_series))

    np.save(os.path.join(output_dir, f"{model_name}_ssim_timeseries.npy"), np.array(ssim_time_series))
    np.save(os.path.join(output_dir, f"{model_name}_psnr_timeseries.npy"), np.array(psnr_time_series))

    summary_file = os.path.join(output_dir, f"{model_name}_metrics_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"[{model_name.upper()} Test Evaluation Summary]\n")
        f.write(f"Global_MAE: {mae:.6f}\n")
        f.write(f"Global_MSE: {mse:.6f}\n")
        f.write(f"Global_RMSE: {rmse:.6f}\n")
        f.write(f"Global_BIAS: {bias:.6f}\n")
        f.write(f"Global_SSIM: {mean_ssim:.6f}\n")
        f.write(f"Global_PSNR: {mean_psnr:.6f}\n")

    print("=" * 65)
    print(f" METEOROLOGICAL EVALUATION REPORT ({model_name.upper()})")
    print("=" * 65)
    print(f" ▶ Global Mean Absolute Error (MAE)   : {mae:.4f} K")
    print(f" ▶ Global Mean Squared Error (MSE)    : {mse:.4f} K²")
    print(f" ▶ Global Root Mean Squared Error (RMSE): {rmse:.4f} K")
    print(f" ▶ Systematic Model Variance Bias (BIAS): {bias:.4f} K")
    print(f" ▶ Global Structural Similarity (SSIM): {mean_ssim:.4f}")
    print(f" ▶ Global Peak Signal-to-Noise (PSNR) : {mean_psnr:.4f} dB")
    print("=" * 65)

    return ssim_time_series
