# Temporal environmental data — Phase 1 starter

Target project directory on Windows:

    D:\temporal_environmental_data

Sites:
- Jolo: 6.1098, 120.9775
- Masinloc: 15.55672, 119.93036
- Palapag: 12.63476, 125.00978

## 1) Create environment

Open Anaconda Prompt or a VS Code terminal:

    conda create -n temporal_env python=3.12 -y
    conda activate temporal_env
    conda install -c conda-forge xarray netcdf4 h5netcdf pandas numpy scipy matplotlib openpyxl -y

## 2) Copy the `scripts` folder to

    D:\temporal_environmental_data\scripts

Then run:

    python D:\temporal_environmental_data\scripts\00_setup_project.py

Copy `00_metadata\sites.csv` into:

    D:\temporal_environmental_data\00_metadata\sites.csv

## 3) Run ERSST Phase 1

    python D:\temporal_environmental_data\scripts\01_ersst_extract.py

The first run downloads the NOAA PSL ERSSTv5 monthly NetCDF (~158 MB) and caches it under:

    D:\temporal_environmental_data\01_raw\ERSST\sst.mnmean.nc

Outputs are written under `03_extracted`, `04_tables`, and `05_figures`.

## Scientific provenance

ERSSTv5 is classified in the output as:

    observation_based_reconstruction

It is NOT labeled a direct observation.

ERSST is 2° x 2°, so this script intentionally uses the nearest ocean grid cell rather
than pretending a 70-km buffer mean is supported at that resolution. The actual ERSST
grid-cell center and site-to-grid distance are stored for each site.

The 70-km buffer will be applied to datasets with adequate spatial resolution later.
