from typing import Any

import torch
from lightning import LightningModule

from src.models.unet.unet_architecture import DownscalingUnet
from src.models.unet_gan.patchgan_discriminator import PatchGANDiscriminator, weights_init
from src.losses import PatchGANPSDGeneratorLoss


class GreeceDownscalingGANModule(LightningModule):
    """
    LightningModule for ERA5-to-CERRA downscaling using a U-Net generator
    and a conditional PatchGAN discriminator.

    This module follows the GAN training logic adapted from:
    DSIP-FBK/DiffScaler
    https://github.com/DSIP-FBK/DiffScaler/blob/main/src/models/gan_module.py

    Main training logic:
    - manual optimization is used,
    - the generator is updated first,
    - the discriminator is updated second,
    - both optimizers use Adam with betas=(0.5, 0.9),
    - the discriminator is activated after a warm-up period inside the loss.

    The only methodological addition is the PSD term inside the generator loss.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        base_learning_rate: float = 4.5e-6,
        lambda_psd: float = 0.05,
        disc_start: int = 50000,
        disc_weight: float = 0.2,
        disc_factor: float = 1.0,
    ):
        super().__init__()

        # Save all constructor arguments in the checkpoint.
        # This makes the experiment reproducible and allows Lightning to reload
        # the model configuration together with the trained weights.
        self.save_hyperparameters()

        # Manual optimization is required because GAN training uses two optimizers:
        # one optimizer for the generator and one optimizer for the discriminator.
        # Lightning will not call backward() and optimizer.step() automatically.
        self.automatic_optimization = False

        # Generator: U-Net that maps the input predictors to the target field.
        # In this project:
        #   input  x: ERA5 predictors + static fields, shape [B, 7, H, W]
        #   output y_pred: predicted CERRA temperature, shape [B, 1, H, W]
        self.net = DownscalingUnet(in_ch=in_channels, out_ch=out_channels)

        # Conditional PatchGAN discriminator.
        # It receives the condition and the temperature field concatenated together:
        #   real pair: [x, y]
        #   fake pair: [x, y_pred]
        #
        # Therefore the number of discriminator input channels is:
        #   in_channels + out_channels = 7 + 1 = 8
        self.discriminator = PatchGANDiscriminator(
            input_nc=in_channels + out_channels,
            ndf=64,
            n_layers=3,
        ).apply(weights_init)

        # Combined loss object.
        # This contains:
        #   generator loss: MAE + lambda_psd * PSD + adaptive adversarial loss
        #   discriminator loss: hinge loss
        #
        # The delayed discriminator activation is handled inside this loss object
        # through disc_start and adopt_weight().
        self.loss = PatchGANPSDGeneratorLoss(
            discriminator=self.discriminator,
            lambda_psd=lambda_psd,
            disc_start=disc_start,
            disc_weight=disc_weight,
            disc_factor=disc_factor,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the generator only.

        Args:
            x: Input tensor [B, 7, H, W].

        Returns:
            Predicted CERRA temperature field [B, 1, H, W].
        """
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        """
        One GAN training step.

        The dataset returns:
            x: normalized ERA5 predictors + static fields
            y: normalized CERRA target

        The training order follows the original GAN logic:
            1. update generator / U-Net
            2. update discriminator / PatchGAN
        """

        # Retrieve the two optimizers returned by configure_optimizers().
        unet_opt, d_opt = self.optimizers()

       
        x, y = batch[0].float(), batch[1].float()

        # Generate the high-resolution prediction.
        y_pred = self(x)

        # ================================================================
        # 1. Generator / U-Net update
        # ================================================================
        self.toggle_optimizer(unet_opt)

        # Generator loss:
        #   MAE + lambda_psd * PSD + adaptive_weight * disc_factor * GAN loss
        #
        # condition=x is used because this is a conditional PatchGAN:
        # the discriminator evaluates whether y_pred is realistic given x.
        unetloss, log_dict_unet = self.loss.generator_loss(
            condition=x,
            pred=y_pred,
            target=y,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
        )

        self.log(
            "unetgan_loss",
            unetloss,
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Manual backward and optimizer step for the generator.
        unet_opt.zero_grad()
        self.manual_backward(unetloss)
        unet_opt.step()

        self.untoggle_optimizer(unet_opt)

        # ================================================================
        # 2. Discriminator / PatchGAN update
        # ================================================================
        self.toggle_optimizer(d_opt)

        # Discriminator loss:
        #   hinge loss between real pairs [x, y] and fake pairs [x, y_pred].
        #
        # Inside discriminator_loss(), y_pred is detached so the discriminator
        # update does not modify the generator.
        discloss, log_dict_disc = self.loss.discriminator_loss(
            condition=x,
            pred=y_pred,
            target=y,
            global_step=self.global_step,
        )

        self.log(
            "disc_loss",
            discloss,
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Manual backward and optimizer step for the discriminator.
        d_opt.zero_grad()
        self.manual_backward(discloss)
        d_opt.step()

        self.untoggle_optimizer(d_opt)

        # Log all individual loss components:
        # MAE, PSD, adversarial loss, adaptive weight, discriminator loss, etc.
        self.log_dict(
            {**log_dict_unet, **log_dict_disc},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Validation step.

        No optimizer update is performed here.
        The method only computes the generator and discriminator losses
        for monitoring.
        """
        return self._shared_eval_step(batch, suffix="val")

    def test_step(self, batch: Any, batch_idx: int):
        """
        Test step.

        This logs normalized-space GAN losses.
        Your final physical evaluation can still be done separately using
        denormalized predictions, as in your existing test.py pipeline.
        """
        return self._shared_eval_step(batch, suffix="test")

    def _shared_eval_step(self, batch: Any, suffix: str = "val"):
        """
        Shared validation/test logic.

        This avoids duplicating the same evaluation code in validation_step()
        and test_step().
        """
        x, y = batch[0].float(), batch[1].float()
        y_pred = self(x)

        # Compute generator-side loss terms.
        unetloss, log_dict_unet = self.loss.generator_loss(
            condition=x,
            pred=y_pred,
            target=y,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
        )

        # Compute discriminator-side hinge loss.
        discloss, log_dict_disc = self.loss.discriminator_loss(
            condition=x,
            pred=y_pred,
            target=y,
            global_step=self.global_step,
        )

        self.log(
            f"{suffix}/unetgan_loss",
            unetloss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            f"{suffix}/disc_loss",
            discloss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        logs = {}
        logs.update({f"{suffix}/{k}": v for k, v in log_dict_unet.items()})
        logs.update({f"{suffix}/{k}": v for k, v in log_dict_disc.items()})

        self.log_dict(
            logs,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return logs

    def configure_optimizers(self):
        """
        Create one optimizer for the generator and one for the discriminator.

        The learning rate follows the same scaling rule as the prototype:
            lr = base_learning_rate
                 * batch_size
                 * number_of_devices
                 * accumulate_grad_batches
        """

        bs = self.trainer.datamodule.hparams.batch_size
        agb = self.trainer.accumulate_grad_batches
        ngpu = self.trainer.num_devices

        self.learning_rate = agb * ngpu * bs * self.hparams.base_learning_rate

        # Generator optimizer.
        unet_opt = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
            foreach=True,
        )

        # Discriminator optimizer.
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
            foreach=True,
        )

        return [unet_opt, d_opt], []

    def get_last_layer(self):
        """
        Return the final generator layer used for adaptive adversarial weighting.

        The adaptive weight compares gradients of:
            reconstruction/PSD loss
        and:
            adversarial loss

        with respect to this last layer.
        """
        return self.net.last_layer().weight

    def on_test_epoch_end(self):
        pass
