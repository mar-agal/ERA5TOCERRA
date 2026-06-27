# data_pipeline overview

A collection of modules for processing climate data to support super-resolution downscaling tasks. This pipeline handles:

- Acquisition of ERA5 (low-resolution) and CERRA (high-resolution) data via CDS API
- Reprojection and alignment of both datasets to a common 256×256 grid over Greece
- Temporal splitting into train/validation/test sets
- Statistical analysis for normalization

## Pipeline Modules

### **`data_pipeline/Climate_Data_Acquisition_and_Transformation.ipynb`**

**Primary ingestion and transformation module**

This notebook serves as the entry point for the data pipeline, handling the transition from raw climate data to processed inputs.

- **Data Acquisition:** Automates the download of GRIB files from the Copernicus Climate Data Store (CDS) API:
    - **ERA5 variables:** Temperature ($t$), zonal wind ($u$), meridional wind ($v$), geopotential ($z$) at 850hPa, and dewpoint temperature ($2d$).
    - **CERRA variables:** 2m temperature ($t2m$), orography ($orog$), and land-sea mask ($lsm$).
- **Data Transformation:** Converts downloaded GRIB files into NetCDF format and structures them into (B, C, W, H) tensors, where:
    - **B (Batch):** Temporal/batch dimension.
    - **C (Channels):** Atmospheric variables/features.
    - **W, H (Width, Height):** Spatial grid dimensions (256×256).
- **Grid Alignment:** Utilizes **CDO (Climate Data Operators)** to perform bilinear remapping (`cdo remapbil`), projecting both ERA5 and CERRA datasets onto a uniform target grid.

#### Reprojection CERRA to match ERA-5
To ensure spatial consistency for the model, all data is mapped to a regular Lat/Lon grid defined in `config/cyl_greece.txt` (256×256 pixels):

- **Target Grid Configuration:**
    - **Start:** Lat 17.0, Lon 33.0.
    - **End:** Lat 29.75, Lon 45.75.
    - **Coordinate Calculation:** 
        - Lat: $17.0 + (255 \times 0.05) = 29.75$
        - Lon: $33.0 + (255 \times 0.05) = 45.75$
- **Super-resolution Ratio:** The super-resolution ratio between ERA-5 and CERRA is 52×52 to 256×256, which is an upscaling factor of approximately **x4.9**.
### `data_pipeline/config.yaml`

Centralized configuration file containing:
- Geographical boundaries
- Grid specifications
- Path definitions

### `data_pipeline/split.py`

Temporal data partitioning script that organizes files into:

- **Train** (2010–2019) — long-term atmospheric patterns
- **Validation** (2020) — hyperparameter tuning
- **Test** (2021) — final model evaluation

### `data_pipeline/extract_stats.py`

Statistical processing engine that computes global statistics (mean, std, min, max, median) across the training dataset using Dask/PyTorch. Results are saved in https://www.kaggle.com/datasets/mariaagalioti/stats-era-cerra for input normalization.

