import torch
import torch.nn as nn
import math

class FourierLossCarlo(nn.Module):
    """
    2D Power Spectral Density (PSD) and Mean Squared Error (MSE) Joint Loss Function.
    
    Adapted from the official implementation by Carlo Saccardi for physics-consistent 
    meteorological climate downscaling:
    Source: https://github.com/CarloSaccardi/PSD-Downscaling/
    Branch: CNN-UNet
    """
    def __init__(self, spatial_resolution_km: float = 5.5, epsilon_floor: float = 1e-8) -> None:
        super(FourierLossCarlo, self).__init__()
        self.dx = spatial_resolution_km  
        self.eps = epsilon_floor         

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes spatial pixel-wise MSE alongside spectral power distribution loss.
        """
        sr_psd = self.getpsd(sr)
        hr_psd = self.getpsd(hr)
        
        diff_log_psd = torch.log(sr_psd + self.eps) - torch.log(hr_psd + self.eps)
        w = self.make_high_freq_weights(sr)
        
        w = w.expand_as(diff_log_psd)
        
        mse_loss = torch.mean((sr - hr) ** 2)
        psd_loss = torch.sqrt(torch.mean(w * (diff_log_psd ** 2)))
        
        return mse_loss, psd_loss

    def getpsd(self, image: torch.Tensor) -> torch.Tensor:
        """
        Computes the 2D power spectral density (PSD) of real-valued images using rFFTN.
        """
        v_ft = torch.fft.rfftn(image, dim=(2, 3))
        N = image.shape[-2] * image.shape[-1]
        psd = (v_ft.real**2 + v_ft.imag**2) / (N * self.dx)
        return psd
    
    def make_high_freq_weights(self, image: torch.Tensor) -> torch.Tensor:
        """
        Creates a parabolic 2D wavenumber weight mask to emphasize high spatial frequencies.
        """
        _, _, H, W = image.shape
        W_rfft = W // 2 + 1

        dk_h = 2 * math.pi / (H * self.dx)
        dk_w = 2 * math.pi / (W * self.dx)

        k_h = torch.arange(H, device=image.device).reshape(H, 1) * dk_h
        k_w = torch.arange(W_rfft, device=image.device).reshape(1, W_rfft) * dk_w

        k_grid = torch.sqrt(k_h ** 2 + k_w ** 2)
        
        max_grid = k_grid.max()
        if max_grid > 0:
            weights = (k_grid / max_grid).pow(2)
        else:
            weights = torch.ones_like(k_grid)

        return weights.unsqueeze(0).unsqueeze(0)


