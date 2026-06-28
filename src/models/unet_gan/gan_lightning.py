from typing import Any, Dict, Optional

import torch
from lightning import LightningModule

from src.models.unet.unet_architecture import DownscalingUnet
from src.models.unet_gan.patchgan_discriminator import (
    PatchGANDiscriminator,
    weights_init,
)
from src.losses import PatchGANPSDGeneratorLoss


class GreeceDownscalingGANModule(LightningModule):
    """
    LightningModule for conditional U-Net + PatchGAN training.

    Generator:
        U-Net: x -> predicted CERRA temperature

    Discriminator:
        PatchGAN: concat([x, temperature]) -> patch-wise real/fake logits

    Generator loss:
        MAE + lambda_psd * PSD + adaptive_weight * adversarial_loss

    Discriminator loss:
        Hinge loss
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        learning_rate: float = 1e-4,
        discriminator_learning_rate: Optional[float] = None,
        lambda_psd: float = 0.05,
        disc_start: int = 50000,
        disc_weight: float = 0.2,
        disc_factor: float = 1.0,
        weight_decay: float = 0.0,
        savepreds_path: str = "/kaggle/working/outputs",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.generator = DownscalingUnet(
            in_ch=in_channels,
            out_ch=out_channels,
        )

        self.discriminator = PatchGANDiscriminator(
            input_nc=in_channels + out_channels,
            ndf=64,
            n_layers=3,
        )
        self.discriminator.apply(weights_init)

        self.loss = PatchGANPSDGeneratorLoss(
            discriminator=self.discriminator,
            lambda_psd=lambda_psd,
            disc_start=disc_start,
            disc_factor=disc_factor,
            disc_weight=disc_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def get_last_layer(self) -> torch.Tensor:
        return self.generator.last_layer().weight

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        generator_optimizer, discriminator_optimizer = self.optimizers()

        x, y = batch[0].float(), batch[1].float()

        pred = self(x)

        # -------------------------
        # 1. Generator update
        # -------------------------
        self.toggle_optimizer(generator_optimizer)

        generator_loss, generator_log = self.loss.generator_loss(
            condition=x,
            pred=pred,
            target=y,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
        )

        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()

        self.untoggle_optimizer(generator_optimizer)

        # -------------------------
        # 2. Discriminator update
        # -------------------------
        self.toggle_optimizer(discriminator_optimizer)

        discriminator_loss, discriminator_log = self.loss.discriminator_loss(
            condition=x,
            pred=pred.detach(),
            target=y,
            global_step=self.global_step,
        )

        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()

        self.untoggle_optimizer(discriminator_optimizer)

        logs = {
            **generator_log,
            **discriminator_log,
        }

        self.log("train/generator_loss", generator_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/discriminator_loss", discriminator_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return {
            "loss": generator_loss.detach(),
            "generator_loss": generator_loss.detach(),
            "discriminator_loss": discriminator_loss.detach(),
        }

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch[0].float(), batch[1].float()

        pred = self(x)

        generator_loss, generator_log = self.loss.generator_loss(
            condition=x,
            pred=pred,
            target=y,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
        )

        discriminator_loss, discriminator_log = self.loss.discriminator_loss(
            condition=x,
            pred=pred,
            target=y,
            global_step=self.global_step,
        )

        logs = {
            **{f"val/{k}": v for k, v in generator_log.items()},
            **{f"val/{k}": v for k, v in discriminator_log.items()},
        }

        self.log("val/generator_loss", generator_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/discriminator_loss", discriminator_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(logs, on_epoch=True, prog_bar=False, sync_dist=True)

        return {
            "val_generator_loss": generator_loss.detach(),
            "val_discriminator_loss": discriminator_loss.detach(),
        }

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch[0].float(), batch[1].float()

        pred = self(x)

        generator_loss, generator_log = self.loss.generator_loss(
            condition=x,
            pred=pred,
            target=y,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
        )

        discriminator_loss, discriminator_log = self.loss.discriminator_loss(
            condition=x,
            pred=pred,
            target=y,
            global_step=self.global_step,
        )

        logs = {
            **{f"test/{k}": v for k, v in generator_log.items()},
            **{f"test/{k}": v for k, v in discriminator_log.items()},
        }

        self.log("test/generator_loss", generator_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/discriminator_loss", discriminator_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(logs, on_epoch=True, prog_bar=False, sync_dist=True)

        return {
            "test_generator_loss": generator_loss.detach(),
            "test_discriminator_loss": discriminator_loss.detach(),
        }

    def configure_optimizers(self):
        discriminator_lr = (
            self.hparams.discriminator_learning_rate
            if self.hparams.discriminator_learning_rate is not None
            else self.hparams.learning_rate
        )

        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.9),
            weight_decay=self.hparams.weight_decay,
        )

        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self.hparams.weight_decay,
        )

        return [generator_optimizer, discriminator_optimizer], []
