import os
import time
import yaml
import torch

from lightning import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data_module import GreeceDownscalingDataModule
from src.models.unet.unet_psd_h_lightning import GreeceDownscalingHuberPSDModule


class LogEveryEpoch(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train/loss_epoch")
        val_loss = trainer.callback_metrics.get("val/loss")
        lambda_psd = trainer.callback_metrics.get("loss/lambda_psd")

        msg = f"\n [EPOCH {epoch:03d}]"

        if train_loss is not None:
            msg += f" Train Loss: {train_loss:.6f} |"

        if val_loss is not None:
            msg += f" Val Loss: {val_loss:.6f} |"

        if lambda_psd is not None:
            msg += f" Lambda PSD: {lambda_psd:.6f}"

        print(msg)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print(" Initializing DataModule...")

    data_module = GreeceDownscalingDataModule(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        upsample=True,
        used_channels=cfg["used_channels"],
        seed=cfg["seed"]
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

    print(" Building UNet-HuberPSD model...")

    model = GreeceDownscalingHuberPSDModule(
        in_channels=cfg["img_in_channels"],
        out_channels=cfg["img_out_channels"],
        learning_rate=float(cfg["lr"]),
        init_lambda=float(cfg["init_lambda"]),
        max_lambda=float(cfg["max_lambda"]),
        anneal_epochs=int(cfg["anneal_epochs"]),
        savepreds_path=cfg["savepreds_path"]
    )

    weights_dir = "/kaggle/working/weights/huber_psd"
    os.makedirs(weights_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=weights_dir,
        filename="best-unet-huber-psd",
        save_top_k=1,
        mode="min",
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    epoch_logger = LogEveryEpoch()

    callbacks_list = [
        checkpoint_callback,
        lr_monitor,
        epoch_logger
    ]

    trainer = Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks_list,
        precision=cfg["precision"],
        enable_progress_bar=False,
        log_every_n_steps=1000
    )

    resume_path = os.path.join(weights_dir, "last.ckpt")

    if os.path.exists(resume_path):
        print(f" Found last checkpoint. Resuming from: {resume_path}")
    else:
        print(" No checkpoint found. Starting from scratch.")
        resume_path = None

    print("\n Launching UNet-HuberPSD training...")
    start_time = time.time()

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_path
    )

    end_time = time.time()

    if trainer.is_global_zero:
        total_seconds = end_time - start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        print(f"\n TRAINING COMPLETE! Total Time: {hours}h {minutes}m {seconds}s")
