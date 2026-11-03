"""
Phase 1: NOAA ERSSTv5 extraction for Jolo, Masinloc, and Palapag.
Period: 1900-2018 inclusive.

This version uses NOAA PSL's NetCDF Subset Service (NCSS), downloading only
a Philippine bounding box for 1900-2018 instead of the full ~157.5 MB global file.

ERSSTv5 is a 2° x 2° Level-4 observation-based reconstructed SST product.
Because a 70-km buffer is smaller than the ERSST grid spacing, this script
uses the nearest ERSST ocean grid cell for each site and records its center
and site-to-grid-center distance. Do NOT describe these values as a 70-km mean.
"""

from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress
import requests

ROOT = Path(r"D:\temporal_environmental_data")
SITES_FILE = ROOT / "00_metadata" / "sites.csv"
RAW_DIR = ROOT / "01_raw" / "ERSST"
OUT_DIR = ROOT / "03_extracted"
TABLE_DIR = ROOT / "04_tables"
FIG_DIR = ROOT / "05_figures"
LOG_DIR = ROOT / "07_logs"

for d in [RAW_DIR, OUT_DIR, TABLE_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START_YEAR = 1900
END_YEAR = 2018

# NOAA PSL NetCDF Subset Service.
NCSS_URL = (
    "https://psl.noaa.gov/thredds/ncss/grid/"
    "Datasets/noaa.ersst.v5/sst.mnmean.nc"
)

# Bounding box deliberately larger than all three sites so their nearest 2° cells
# are guaranteed to be available.
BBOX = {
    "north": 18.0,
    "south": 4.0,
    "west": 118.0,
    "east": 128.0,
}

NC_FILE = RAW_DIR / "ersstv5_philippines_1900_2018.nc"
PART_FILE = RAW_DIR / "ersstv5_philippines_1900_2018.nc.part"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def to_360(lon):
    return lon % 360


def download_subset(max_attempts=5):
    params = {
        "var": "sst",
        "north": BBOX["north"],
        "south": BBOX["south"],
        "west": BBOX["west"],
        "east": BBOX["east"],
        "time_start": f"{START_YEAR}-01-01T00:00:00Z",
        "time_end": f"{END_YEAR}-12-31T23:59:59Z",
        "accept": "netcdf4",
    }

    for attempt in range(1, max_attempts + 1):
        try:
            if PART_FILE.exists():
                PART_FILE.unlink()

            print(f"Downloading NOAA ERSSTv5 Philippine subset (attempt {attempt}/{max_attempts})...")
            with requests.get(
                NCSS_URL,
                params=params,
                stream=True,
                timeout=(30, 300),
            ) as r:
                r.raise_for_status()

                content_type = r.headers.get("Content-Type", "")
                total = int(r.headers.get("Content-Length", 0))

                print(f"Server content type: {content_type}")
                if total:
                    print(f"Expected subset size: {total / 1024 / 1024:.2f} MB")

                downloaded = 0
                with open(PART_FILE, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = 100 * downloaded / total
                                print(
                                    f"\rDownloaded {downloaded / 1024 / 1024:.2f} MB "
                                    f"({pct:.1f}%)",
                                    end="",
                                    flush=True,
                                )

                print()

            if total and PART_FILE.stat().st_size != total:
                raise IOError(
                    f"Incomplete download: got {PART_FILE.stat().st_size} bytes, "
                    f"expected {total}"
                )

            # Validate before promoting the temporary file.
            test = xr.open_dataset(PART_FILE)
            if "sst" not in test.variables:
                test.close()
                raise RuntimeError("Downloaded file does not contain expected 'sst' variable.")
            test.close()

            PART_FILE.replace(NC_FILE)
            print(f"Validated and saved: {NC_FILE}")
            return

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if PART_FILE.exists():
                PART_FILE.unlink()
            if attempt == max_attempts:
                raise
            time.sleep(3 * attempt)


print("=== ERSSTv5 Phase 1 — NOAA subset version ===")

if not SITES_FILE.exists():
    raise FileNotFoundError(
        f"{SITES_FILE} not found. Run scripts\\00_setup_project.py first."
    )

# Remove the incomplete full-global file left by the original downloader, if present.
old_global = RAW_DIR / "sst.mnmean.nc"
if old_global.exists():
    try:
        old_size = old_global.stat().st_size
        if old_size < 150_000_000:
            print(
                f"Removing incomplete old global download "
                f"({old_size / 1024 / 1024:.2f} MB): {old_global}"
            )
            old_global.unlink()
    except OSError:
        pass

if not NC_FILE.exists():
    download_subset()
else:
    print(f"Using cached subset: {NC_FILE}")

print("Opening NetCDF subset...")
ds = xr.open_dataset(NC_FILE)

required = {"sst", "lat", "lon", "time"}
missing = required.difference(set(ds.variables) | set(ds.coords))
if missing:
    raise RuntimeError(f"Missing expected ERSST variables/coordinates: {missing}")

print("Subset dimensions:", dict(ds.sizes))
print(
    f"Subset time coverage: "
    f"{pd.to_datetime(ds.time.values[0]).date()} to "
    f"{pd.to_datetime(ds.time.values[-1]).date()}"
)

sst = ds["sst"].sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
sites = pd.read_csv(SITES_FILE)

monthly_frames = []
provenance_rows = []

for row in sites.itertuples(index=False):
    site = row.site
    site_lat = float(row.latitude)
    site_lon = float(row.longitude)
    site_lon_360 = to_360(site_lon)

    point = sst.sel(lat=site_lat, lon=site_lon_360, method="nearest").load()

    grid_lat = float(point["lat"].values)
    grid_lon_360 = float(point["lon"].values)
    grid_lon_display = grid_lon_360 if grid_lon_360 <= 180 else grid_lon_360 - 360
    distance_km = haversine_km(site_lat, site_lon, grid_lat, grid_lon_display)

    print(
        f"{site}: site=({site_lat:.5f}, {site_lon:.5f}) -> "
        f"ERSST cell=({grid_lat:.2f}, {grid_lon_display:.2f}), "
        f"distance={distance_km:.1f} km"
    )

    df = point.to_dataframe(name="sst_degC").reset_index()
    df["site"] = site
    df["site_latitude"] = site_lat
    df["site_longitude"] = site_lon
    df["grid_latitude"] = grid_lat
    df["grid_longitude"] = grid_lon_display
    df["grid_center_distance_km"] = distance_km
    df["dataset"] = "NOAA ERSSTv5"
    df["variable"] = "sea_surface_temperature"
    df["unit"] = "degC"
    df["data_class"] = "observation_based_reconstruction"
    df["processing_level"] = "NOAA Level 4"
    df["spatial_resolution"] = "2 degree x 2 degree"
    df["spatial_extraction"] = "nearest_ocean_grid_cell"
    df["requested_buffer_km"] = 70
    df["buffer_applied"] = False
    df["source_basis"] = (
        "in_situ marine observations with statistical reconstruction; "
        "not a direct point observation"
    )
    monthly_frames.append(df)

    provenance_rows.append(
        {
            "site": site,
            "site_latitude": site_lat,
            "site_longitude": site_lon,
            "ersst_grid_latitude": grid_lat,
            "ersst_grid_longitude": grid_lon_display,
            "site_to_grid_center_km": distance_km,
            "requested_buffer_km": 70,
            "buffer_applied": False,
            "spatial_extraction": "nearest_ocean_grid_cell",
            "effective_spatial_support": "2 degree ERSST grid cell",
            "dataset": "NOAA ERSSTv5",
            "data_class": "observation_based_reconstruction",
            "processing_level": "NOAA Level 4",
            "period_used": f"{START_YEAR}-{END_YEAR}",
        }
    )

monthly = pd.concat(monthly_frames, ignore_index=True)
monthly["year"] = pd.to_datetime(monthly["time"]).dt.year
monthly["month"] = pd.to_datetime(monthly["time"]).dt.month

monthly_cols = [
    "site", "site_latitude", "site_longitude",
    "year", "month", "time", "sst_degC",
    "grid_latitude", "grid_longitude", "grid_center_distance_km",
    "dataset", "variable", "unit", "data_class", "processing_level",
    "spatial_resolution", "spatial_extraction",
    "requested_buffer_km", "buffer_applied", "source_basis",
]
monthly = monthly[monthly_cols].sort_values(["site", "year", "month"])
monthly.to_csv(OUT_DIR / "ersstv5_monthly_1900_2018.csv", index=False)

annual = (
    monthly.groupby(["site", "site_latitude", "site_longitude", "year"], as_index=False)
    .agg(
        sst_mean_degC=("sst_degC", "mean"),
        sst_sd_monthly_degC=("sst_degC", "std"),
        sst_min_monthly_degC=("sst_degC", "min"),
        sst_max_monthly_degC=("sst_degC", "max"),
        n_months=("sst_degC", "count"),
        grid_latitude=("grid_latitude", "first"),
        grid_longitude=("grid_longitude", "first"),
        grid_center_distance_km=("grid_center_distance_km", "first"),
    )
)

annual["sst_seasonal_range_degC"] = (
    annual["sst_max_monthly_degC"] - annual["sst_min_monthly_degC"]
)
annual["dataset"] = "NOAA ERSSTv5"
annual["data_class"] = "observation_based_reconstruction"
annual["spatial_extraction"] = "nearest_ocean_grid_cell"
annual["requested_buffer_km"] = 70
annual["buffer_applied"] = False
annual.to_csv(OUT_DIR / "ersstv5_annual_1900_2018.csv", index=False)

provenance = pd.DataFrame(provenance_rows)
provenance.to_csv(TABLE_DIR / "ersstv5_site_grid_provenance.csv", index=False)

trend_rows = []
for site, g in annual.groupby("site"):
    g = g.dropna(subset=["sst_mean_degC"])
    lr = linregress(g["year"], g["sst_mean_degC"])
    trend_rows.append(
        {
            "site": site,
            "variable": "annual_mean_SST",
            "dataset": "NOAA ERSSTv5",
            "data_class": "observation_based_reconstruction",
            "period": f"{START_YEAR}-{END_YEAR}",
            "n_years": len(g),
            "slope_degC_per_year": lr.slope,
            "slope_degC_per_decade": lr.slope * 10,
            "intercept": lr.intercept,
            "r_squared": lr.rvalue ** 2,
            "p_value": lr.pvalue,
            "stderr_slope_per_year": lr.stderr,
        }
    )

trend = pd.DataFrame(trend_rows)
trend.to_csv(TABLE_DIR / "ersstv5_trend_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))

for site, g in annual.groupby("site", sort=False):
    g = g.sort_values("year").copy()
    g["sst_roll10"] = g["sst_mean_degC"].rolling(
        window=10, center=True, min_periods=5
    ).mean()

    line, = ax.plot(
        g["year"],
        g["sst_mean_degC"],
        linewidth=0.8,
        alpha=0.35,
        label=f"{site} annual",
    )
    ax.plot(
        g["year"],
        g["sst_roll10"],
        linewidth=2.2,
        label=f"{site} 10-y mean",
        color=line.get_color(),
    )

ax.axvline(1940, linestyle="--", linewidth=1)
ax.set_title("Sea-surface temperature trajectories at three Philippine locations")
ax.set_xlabel("Year")
ax.set_ylabel("Sea-surface temperature (°C)")
ax.set_xlim(START_YEAR, END_YEAR)
ax.legend(frameon=False, ncol=2, fontsize=8)
ax.grid(alpha=0.2)
fig.tight_layout()

fig.savefig(FIG_DIR / "Fig_ERSSTv5_SST_1900_2018.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "Fig_ERSSTv5_SST_1900_2018.pdf", bbox_inches="tight")
plt.close(fig)

ds.close()

print("\nDONE.")
print(f"Monthly:    {OUT_DIR / 'ersstv5_monthly_1900_2018.csv'}")
print(f"Annual:     {OUT_DIR / 'ersstv5_annual_1900_2018.csv'}")
print(f"Provenance: {TABLE_DIR / 'ersstv5_site_grid_provenance.csv'}")
print(f"Trends:     {TABLE_DIR / 'ersstv5_trend_summary.csv'}")
print(f"Figure PNG: {FIG_DIR / 'Fig_ERSSTv5_SST_1900_2018.png'}")
print(f"Figure PDF: {FIG_DIR / 'Fig_ERSSTv5_SST_1900_2018.pdf'}")
