import torch
import torch.nn as nn
from lightning import LightningModule

from src.models.unet.unet_architecture import DownscalingUnet
from src.losses import HuberPSDLoss


class GreeceDownscalingHuberPSDModule(LightningModule):
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        init_lambda: float = 0.0,
        max_lambda: float = 0.1,
        anneal_epochs: int = 20,
        savepreds_path: str = "/kaggle/working/outputs"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DownscalingUnet(
            in_ch=in_channels,
            out_ch=out_channels
        )

        self.loss_criterion = HuberPSDLoss(
            spatial_resolution_km=5.5
        )

        self.mse_criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _get_lambda_psd(self) -> float:
        if self.hparams.anneal_epochs <= 0:
            return self.hparams.max_lambda

        if self.current_epoch >= self.hparams.anneal_epochs:
            return self.hparams.max_lambda

        progress = self.current_epoch / self.hparams.anneal_epochs

        lambda_psd = self.hparams.init_lambda + (
            self.hparams.max_lambda - self.hparams.init_lambda
        ) * (progress ** 2)

        return lambda_psd

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
        lambda_psd = self._get_lambda_psd()

        loss_total, huber_loss, psd_loss = self.loss_criterion(
            pred,
            target,
            lambda_psd=lambda_psd
        )

        self.log("loss/huber", huber_loss, on_epoch=True, prog_bar=False)
        self.log("loss/psd", psd_loss, on_epoch=True, prog_bar=False)
        self.log("loss/lambda_psd", lambda_psd, on_epoch=True, prog_bar=True)

        return loss_total

    def training_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        pred = self(x)

        loss = self._compute_loss(pred, y)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        pred = self(x)

        loss = self._compute_loss(pred, y)

        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        pred = self(x)

        mse = self.mse_criterion(pred, y)

        self.log(
            "test/pure_mse",
            mse,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        return mse

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
