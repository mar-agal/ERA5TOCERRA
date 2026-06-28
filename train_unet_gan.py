import os
import time
import yaml
import torch

from lightning import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data_module import GreeceDownscalingDataModule
from src.models.unet_gan.gan_lightning import GreeceDownscalingGANModule


class LogEvery10Epochs(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_rank != 0:
            return

        current_epoch = trainer.current_epoch

        if current_epoch % 10 == 0 or current_epoch == trainer.max_epochs - 1:
            val_g_loss = trainer.callback_metrics.get("val/unetgan_loss")
            val_d_loss = trainer.callback_metrics.get("val/disc_loss")

            if val_g_loss is not None:
                print(f"\n📢 [EPOCH {current_epoch:03d}] val/unetgan_loss: {val_g_loss:.6f}")

            if val_d_loss is not None:
                print(f"📢 [EPOCH {current_epoch:03d}] val/disc_loss: {val_d_loss:.6f}")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    weights_dir = "/kaggle/working/weights_gan"
    os.makedirs(weights_dir, exist_ok=True)

    print("📥 Initializing DataModule...")
    data_module = GreeceDownscalingDataModule(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        upsample=True,
        used_channels=cfg["used_channels"],
    )

    data_module.era5_train_path = "/kaggle/input/datasets/mariaagalioti/era5-train/era5_train.nc"
    data_module.cerra_train_path = "/kaggle/input/datasets/mariaagalioti/cerra-train/cerra_train.nc"
    data_module.era5_val_path = "/kaggle/input/datasets/mariaagalioti/dataset/era5_val.nc"
    data_module.cerra_val_path = "/kaggle/input/datasets/mariaagalioti/dataset/cerra_val.nc"
    data_module.era5_test_path = "/kaggle/input/datasets/mariaagalioti/dataset/era5_test.nc"
    data_module.cerra_test_path = "/kaggle/input/datasets/mariaagalioti/dataset/cerra_test.nc"

    data_module.era5_stats_json = "/kaggle/input/datasets/mariaagalioti/stats-era-cerra/era5_train_global_stats.json"
    data_module.cerra_stats_json = "/kaggle/input/datasets/mariaagalioti/stats-era-cerra/cerra_train_temp_stats.json"
    data_module.orog_stats_json = "/kaggle/input/datasets/mariaagalioti/stats-era-cerra/orography_stats.json"

    print("🏗️ Building U-Net + PatchGAN model...")
    model = GreeceDownscalingGANModule(
        in_channels=cfg["img_in_channels"],
        out_channels=cfg["img_out_channels"],
        base_learning_rate=float(cfg.get("base_learning_rate", 4.5e-6)),
        lambda_psd=float(cfg.get("lambda_psd", 0.05)),
        disc_start=int(cfg.get("disc_start", 50000)),
        disc_weight=float(cfg.get("disc_weight", 0.2)),
        disc_factor=float(cfg.get("disc_factor", 1.0)),
    )

    # Best model according to validation generator loss.
    best_checkpoint = ModelCheckpoint(
        monitor="val/unetgan_loss",
        dirpath=weights_dir,
        filename="best-unetgan-{epoch:03d}-{val_unetgan_loss:.6f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    # Extra safety checkpoint every N training steps.
    # This protects you if Kaggle stops in the middle of a long epoch.
    step_checkpoint = ModelCheckpoint(
        dirpath=weights_dir,
        filename="step-unetgan-{epoch:03d}-{step}",
        every_n_train_steps=int(cfg.get("save_every_n_steps", 2000)),
        save_top_k=-1,
        save_last=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    epoch_logger = LogEvery10Epochs()

    callbacks_list = [
        best_checkpoint,
        step_checkpoint,
        lr_monitor,
        epoch_logger,
    ]

    trainer = Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=2,
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks_list,
        precision=cfg["precision"],
        enable_progress_bar=False,
        log_every_n_steps=50,
    )

    last_ckpt_path = os.path.join(weights_dir, "last.ckpt")

    if os.path.exists(last_ckpt_path):
        print(f" Found last checkpoint. Resuming from: {last_ckpt_path}")
        ckpt_path = last_ckpt_path
    else:
        print(" No last checkpoint found. Starting GAN training from scratch.")
        ckpt_path = None

    print("\n🔥 Launching U-Net + PatchGAN training over 2 GPUs...")
    start_time = time.time()

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )

    end_time = time.time()

    if trainer.is_global_zero:
        total_seconds = end_time - start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        print(f"\n GAN TRAINING COMPLETE! Total Time: {hours}h {minutes}m {seconds}s")
        print(f" Best checkpoint: {best_checkpoint.best_model_path}")
        print(f" Last checkpoint: {last_ckpt_path}")
