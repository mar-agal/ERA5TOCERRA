import torch
import yaml
import os
import time
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from src.data_module import GreeceDownscalingDataModule
from src.models.unet import GreeceDownscalingModule

class LogEvery10Epochs(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
        current_epoch = trainer.current_epoch
        if current_epoch % 10 == 0 or current_epoch == trainer.max_epochs - 1:
            val_loss = trainer.callback_metrics.get("val/loss")
            if val_loss is not None:
                print(f"\n [EPOCH {current_epoch:03d}] Current Validation Loss: {val_loss:.6f}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print(" Initializing DataModule...")
    data_module = GreeceDownscalingDataModule(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        upsample=True,
        used_channels=cfg["used_channels"]
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

    print(" Building Modular UNet Framework...")
    model = GreeceDownscalingModule(
        in_channels=cfg["img_in_channels"],
        out_channels=cfg["img_out_channels"],
        learning_rate=float(cfg["lr"]),
        use_psd_loss=cfg["use_psd_loss"],
        anneal_epochs=cfg["anneal_epochs"],
        savepreds_path=cfg["savepreds_path"]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="/kaggle/working/weights",
        filename="best-baseline-unet",
        save_top_k=1,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    epoch_logger = LogEvery10Epochs()

    callbacks_list = [checkpoint_callback, lr_monitor, epoch_logger]
    
    if not cfg["use_psd_loss"]:
        print(" MSE Only detected: Activating EarlyStopping with patience=12...")
        early_stop = EarlyStopping(monitor="val/loss", patience=12, mode="min", verbose=True)
        callbacks_list.append(early_stop)
    else:
        print(" PSD Blended Loss detected: EarlyStopping disabled to protect annealing curves.")

    trainer = Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=2,                                                                                                                                                 
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks_list,
        precision=cfg["precision"],
        enable_progress_bar=False,
        log_every_n_steps=50
    )

    print("\n Launching execution loop over 2 GPUs...")
    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    end_time = time.time()
    
    if trainer.is_global_zero:
        total_seconds = end_time - start_time
        hours, minutes, seconds = int(total_seconds // 3600), int((total_seconds % 3600) // 60), int(total_seconds % 60)
        print(f"\n TRAINING COMPLETE! Total Time: {hours}h {minutes}m {seconds}s")
