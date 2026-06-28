import torch
import torch.nn as nn
import torch.nn.functional as F

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



def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """
    Delays the activation of a loss term.

    In GAN training, the discriminator is usually activated after a warm-up
    period. Before that point, the adversarial term is set to zero so that the
    generator first learns a stable reconstruction.
    """
    if global_step < threshold:
        return value
    return weight


def hinge_loss(logits_real, logits_fake):
    """
    Hinge loss for the PatchGAN discriminator.

    Real samples are encouraged to have scores greater than +1.
    Fake samples are encouraged to have scores lower than -1.
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


class PatchGANPSDGeneratorLoss(nn.Module):
    """
    Generator and discriminator loss for U-Net + conditional PatchGAN + PSD training.

    PatchGAN adversarial training logic adapted from:
    DSIP-FBK/DiffScaler
    https://github.com/DSIP-FBK/DiffScaler/blob/main/src/models/components/gan.py

    Modified for ERA5-to-CERRA temperature downscaling by adding a PSD
    physical consistency term.

    Generator:
        L_G = MAE + lambda_psd * PSD
              + adaptive_weight * disc_factor * adversarial_loss

    Discriminator:
        L_D = disc_factor * HingeLoss(D(real), D(fake))
    """

    def __init__(
        self,
        discriminator: nn.Module,
        lambda_psd: float = 0.05,
        disc_start: int = 50000,
        disc_factor: float = 1.0,
        disc_weight: float = 0.2,
    ):
        super().__init__()

        self.discriminator = discriminator
        self.lambda_psd = lambda_psd
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        # Same delayed-discriminator logic as DiffScaler.
        # If your Lightning global_step already counts once per batch,
        # remove the "* 2". If you follow DiffScaler exactly, keep it.
        self.discriminator_iter_start = disc_start * 2

        # Use your existing Carlo PSD loss exactly as defined above.
        self.psd_criterion = FourierLossCarlo()

    def reconstruction_loss(self, pred, target):
        """
        Pixel-wise reconstruction loss.

        MAE is used instead of MSE because the DiffScaler GAN setup uses MAE
        and it usually produces less smoothing than MSE in image-to-image tasks.
        """
        return torch.mean(torch.abs(pred - target))

    def generator_adversarial_loss(self, logits_fake):
        """
        Generator adversarial loss.

        The discriminator outputs a patch-wise score map.
        The generator wants fake patches to receive high discriminator scores.
        """
        return -torch.mean(logits_fake)

    def calculate_adaptive_weight(self, reconstruction_loss, adversarial_loss, last_layer):
        """
        Balance reconstruction/physics loss and adversarial loss dynamically.

        It compares the gradient magnitude of:
            reconstruction_loss = MAE + lambda_psd * PSD
        against:
            adversarial_loss = -mean(D(fake))

        The final adversarial weight is:
            disc_weight * ||grad_reconstruction|| / (||grad_adversarial|| + 1e-4)
        """
        reconstruction_grads = torch.autograd.grad(
            reconstruction_loss,
            last_layer,
            retain_graph=True,
        )[0]

        adversarial_grads = torch.autograd.grad(
            adversarial_loss,
            last_layer,
            retain_graph=True,
        )[0]

        adaptive_weight = torch.norm(reconstruction_grads) / (
            torch.norm(adversarial_grads) + 1e-4
        )

        adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
        return adaptive_weight * self.discriminator_weight

    def generator_loss(self, condition, pred, target, global_step, last_layer):
        """
        Compute generator loss.

        condition: ERA5 predictors + static fields, shape [B, 7, H, W]
        pred:      generated CERRA temperature, shape [B, 1, H, W]
        target:    real CERRA temperature, shape [B, 1, H, W]

        Conditional fake input:
            [condition, pred] -> [B, 8, H, W]
        """
        mae = self.reconstruction_loss(pred, target)

        _, psd_loss = self.psd_criterion(pred, target)

        reconstruction_plus_physics = mae + self.lambda_psd * psd_loss

        fake_input = torch.cat([condition, pred], dim=1)
        logits_fake = self.discriminator(fake_input.contiguous())

        adv_loss = self.generator_adversarial_loss(logits_fake)

        try:
            adaptive_weight = self.calculate_adaptive_weight(
                reconstruction_plus_physics,
                adv_loss,
                last_layer=last_layer,
            )
        except RuntimeError:
            adaptive_weight = torch.tensor(0.0, device=pred.device)

        disc_factor = adopt_weight(
            self.disc_factor,
            global_step,
            threshold=self.discriminator_iter_start,
            value=0.0,
        )

        total_loss = reconstruction_plus_physics + adaptive_weight * disc_factor * adv_loss

        log = {
            "generator/total_loss": total_loss.detach(),
            "generator/mae": mae.detach(),
            "generator/psd_loss": psd_loss.detach(),
            "generator/adv_loss": adv_loss.detach(),
            "generator/adaptive_weight": adaptive_weight.detach(),
            "generator/disc_factor": torch.tensor(disc_factor, device=pred.device),
        }

        return total_loss, log

    def discriminator_loss(self, condition, pred, target, global_step):
        """
        Compute discriminator loss.

        Real pair:
            [condition, target]

        Fake pair:
            [condition, pred.detach()]

        The detach prevents discriminator training from updating the generator.
        """
        real_input = torch.cat([condition, target], dim=1)
        fake_input = torch.cat([condition, pred.detach()], dim=1)

        logits_real = self.discriminator(real_input.contiguous())
        logits_fake = self.discriminator(fake_input.contiguous())

        disc_factor = adopt_weight(
            self.disc_factor,
            global_step,
            threshold=self.discriminator_iter_start,
            value=0.0,
        )

        d_loss = disc_factor * hinge_loss(logits_real, logits_fake)

        log = {
            "discriminator/loss": d_loss.detach(),
            "discriminator/logits_real": logits_real.detach().mean(),
            "discriminator/logits_fake": logits_fake.detach().mean(),
            "discriminator/disc_factor": torch.tensor(disc_factor, device=pred.device),
        }

        return d_loss, log


