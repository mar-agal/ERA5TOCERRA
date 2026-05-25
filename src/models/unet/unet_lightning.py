import os
import torch
import torch.nn as nn
from lightning import LightningModule
from src.models.unet.unet_architecture import DownscalingUnet
from src.losses import FourierLossCarlo

class GreeceDownscalingModule(LightningModule):
    def __init__(
        self, 
        in_channels: int = 7, 
        out_channels: int = 1, 
        learning_rate: float = 2e-4, 
        weight_decay: float = 1e-5,
        use_psd_loss: bool = True, 
        init_lambda: float = 0.0,
        max_lambda: float = 0.1,
        anneal_epochs: int = 20, 
        savepreds_path: str = "/kaggle/working/outputs"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DownscalingUnet(in_ch=in_channels, out_ch=out_channels)
        self.mse_criterion = nn.MSELoss()
        
        if self.hparams.use_psd_loss:
            self.psd_criterion = FourierLossCarlo(spatial_resolution_km=5.5)

    def forward(self, x):
        return self.model(x)

    def _compute_blended_loss(self, pred: torch.Tensor, target: torch.Tensor):
        if not self.hparams.use_psd_loss:
            return self.mse_criterion(pred, target)
            
        term_space, term_amp = self.psd_criterion(pred, target)
        if self.current_epoch >= self.hparams.anneal_epochs:
            lambda_psd = self.hparams.max_lambda
        else:
            progress = self.current_epoch / self.hparams.anneal_epochs
            lambda_psd = self.hparams.init_lambda + (self.hparams.max_lambda - self.hparams.init_lambda) * (progress ** 2)
            
        loss_total = term_space + lambda_psd * term_amp
        return loss_total

    def training_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        pred = self(x)
        loss = self._compute_blended_loss(pred, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        pred = self(x)
        loss = self._compute_blended_loss(pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[0].float(), batch[1].float()
        pred = self(x)
        loss = self.mse_criterion(pred, y)
        self.log("test/pure_mse", loss, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
