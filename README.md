# ERA5 to CERRA via U-Net

### Temperature super-resolution

### U-Net architecture

### From ERA5 to CERRA

### Over the Eastern Mediterranean

### Temporal resolution of 3 hours

<p align="center">
  <img src="images/overview.png" width="850">
</p>

---

# Installation

This project is designed to run on **Kaggle**.

Clone the repository

```bash
git clone [https://github.com/<your_username>/<repository_name>.git](https://github.com/mar-agal/ERA5TOCERRA)
%cd /kaggle/working/ERA5TOCERRA
```

Install the required dependency

```bash
pip install lightning
```

Run the training

```bash
python train_unet.py
```

Evaluate the trained model

```bash
python test_unet.py
```

---

# Project Structure

This repository contains several key components that are integral to the project. Below is an overview of each file and its purpose.

## train_unet.py

**Description**

Main training script used in this project. It trains the U-Net model using PyTorch Lightning.

---

## test_unet.py

**Description**

Evaluates the trained U-Net (basic) model on the test dataset.

---

## test_interpolation.py

**Description**

Evaluates interpolation baselines and compares them with the U-Net predictions.

---

## src/data_module.py

**Description**

PyTorch Lightning DataModule responsible for loading the ERA5 and CERRA datasets during training and evaluation.

---

## src/dataset.py

**Description**

Dataset implementation used for loading paired ERA5 and CERRA samples.

---

## src/losses.py

**Description**

Contains the loss functions used during model optimization.

---

## src/metrics.py

**Description**

Implements the evaluation metrics used throughout the experiments.

---

## src/plots.py

**Description**

Utilities used for plotting predictions, comparisons and evaluation figures (psd curves, friquency distribution).

---

## src/models/unet/

### unet_architecture.py

**Description**

Defines the U-Net architecture used throughout this project.

### unet_lightning.py

**Description**

PyTorch Lightning implementation of the U-Net training pipeline.

### unet_psd_h_lightning.py

**Description**

Variant of the Lightning module including Power Spectral Density (PSD) analysis.

---

## src/models/unet_gan/

### gan_lightning.py

### patchgan_discriminator.py

**Description**

Contains an experimental GAN-based implementation for climate super-resolution.

These files are included for future development but **were not used in the experiments presented in this repository**.

---

## data_pipeline/

The complete preprocessing workflow is documented in

```
data_pipeline/README.md
```

It includes:

- ERA5 and CERRA data acquisition
- reprojection onto a common grid
- temporal train / validation / test split
- normalization statistics extraction

---

# Data Availability

A dataset containing the processed ERA5 and CERRA tensors in NumPy format is freely available on Kaggle.

The trained U-Net model weights are also available as a Kaggle Dataset.

The normalization statistics used during training are available as a separate Kaggle Dataset.

The raw climate data used in this project is available from the Copernicus Climate Data Store (CDS) for both ERA5 and CERRA.

### ERA5

https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

### CERRA

https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels

---

# Results

<p align="center">
  <img src="images/results.png" width="900">
</p>

Example comparison between the ERA5 input, the U-Net prediction and the corresponding CERRA target.

---

# Environment

The experiments were conducted using Kaggle Notebooks with:

- Python 3.12.13
- PyTorch
- PyTorch Lightning
- NVIDIA Tesla T4 GPU
