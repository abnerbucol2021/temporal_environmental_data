"""
EN4 Phase 2B — full 1900-2018 extraction for 3 Philippine sites
===============================================================

Dataset:
  Met Office EN.4.2.2 objective analyses, G10 bias-correction member.

Sites:
  Jolo      6.1098, 120.9775
  Masinloc 15.55672, 119.93036
  Palapag  12.63476, 125.00978

Strategy:
  - Process one year at a time.
  - Download annual ZIP only when needed.
  - Extract its 12 monthly NetCDF files.
  - Extract ALL 42 depth levels for all 3 sites.
  - Retain temperature/salinity uncertainty and observation weights.
  - Save one checkpoint CSV per year.
  - Delete the large annual ZIP and extracted NetCDF files after successful
    checkpoint creation (configurable), keeping peak disk use low.
  - Resume safely: already completed annual checkpoint files are skipped.
  - Combine all checkpoints at the end.

IMPORTANT DATA CLASS:
  observation_constrained_objective_analysis

The gridded EN4 fields are not direct point observations.

Outputs:
  03_extracted/EN4_yearly/en4_YYYY_monthly_all_depths.csv
  03_extracted/en4_monthly_1900_2018_all_depths.csv
  03_extracted/en4_annual_1900_2018_all_depths.csv
  04_tables/en4_site_grid_provenance.csv
  04_tables/en4_depth_levels.csv
  04_tables/en4_data_coverage_summary.csv
  07_logs/en4_full_run_log.txt
"""

from pathlib import Path
import math
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(r"D:\temporal_environmental_data")
SITES_FILE = ROOT / "00_metadata" / "sites.csv"

RAW_ROOT = ROOT / "01_raw" / "EN4"
OUT_DIR = ROOT / "03_extracted"
YEARLY_DIR = OUT_DIR / "EN4_yearly"
TABLE_DIR = ROOT / "04_tables"
LOG_DIR = ROOT / "07_logs"

for d in [RAW_ROOT, OUT_DIR, YEARLY_DIR, TABLE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START_YEAR = 1900
END_YEAR = 2018
MEMBER = "g10"

# Delete each annual ~243 MB ZIP and unzipped NetCDFs after that year's
# checkpoint is verified. Set False only if you intentionally want to keep raw files.
CLEAN_RAW_AFTER_SUCCESS = True

BASE_URL = (
    "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/"
    "EN.4.2.2"
)

LOG_FILE = LOG_DIR / "en4_full_run_log.txt"


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def normalize_lon_for_dataset(site_lon, dataset_lons):
    lmin = float(np.nanmin(dataset_lons))
    lmax = float(np.nanmax(dataset_lons))
    if lmax > 180 and site_lon < 0:
        return site_lon % 360
    if lmin >= 0 and site_lon < 0:
        return site_lon % 360
    return site_lon


def display_lon(lon):
    lon = float(lon)
    return lon - 360 if lon > 180 else lon


def download_zip(url, zip_file, part_file, max_attempts=5):
    # Validate existing ZIP first.
    if zip_file.exists():
        try:
            with zipfile.ZipFile(zip_file, "r") as z:
                if z.testzip() is None:
                    log(f"Using cached valid ZIP: {zip_file.name}")
                    return
        except zipfile.BadZipFile:
            pass
        log(f"Deleting invalid cached ZIP: {zip_file}")
        zip_file.unlink(missing_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(1, max_attempts + 1):
        try:
            part_file.unlink(missing_ok=True)
            log(f"Downloading {zip_file.name} (attempt {attempt}/{max_attempts})")

            with requests.get(
                url,
                stream=True,
                timeout=(30, 300),
                headers=headers,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                if total:
                    log(f"Expected size: {total / 1024 / 1024:.2f} MB")

                downloaded = 0
                with open(part_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = 100 * downloaded / total
                                print(
                                    f"\r  {downloaded/1024/1024:.1f} MB "
                                    f"({pct:.1f}%)",
                                    end="",
                                    flush=True,
                                )
                print()

            if total and part_file.stat().st_size != total:
                raise IOError(
                    f"Incomplete download: {part_file.stat().st_size} "
                    f"of {total} bytes"
                )

            with zipfile.ZipFile(part_file, "r") as z:
                bad = z.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Corrupted member: {bad}")

            part_file.replace(zip_file)
            log(f"Validated: {zip_file}")
            return

        except Exception as e:
            log(f"Attempt {attempt} failed: {e}")
            part_file.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise
            time.sleep(3 * attempt)


def identify_names(ds):
    def find_name(candidates, available):
        lower_map = {x.lower(): x for x in available}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        raise KeyError(f"Could not identify {candidates} in {list(available)}")

    all_names = list(ds.coords) + list(ds.variables)
    vars_ = list(ds.data_vars)

    return {
        "depth": find_name(["depth"], all_names),
        "lat": find_name(["lat", "latitude"], all_names),
        "lon": find_name(["lon", "longitude"], all_names),
        "temperature": find_name(["temperature", "temp"], vars_),
        "salinity": find_name(["salinity", "sal"], vars_),
        "temperature_uncertainty": find_name(
            ["temperature_uncertainty"], vars_
        ),
        "salinity_uncertainty": find_name(
            ["salinity_uncertainty"], vars_
        ),
        "temperature_observation_weights": find_name(
            ["temperature_observation_weights"], vars_
        ),
        "salinity_observation_weights": find_name(
            ["salinity_observation_weights"], vars_
        ),
    }


def scalar_or_nan(value):
    arr = np.asarray(value).squeeze()
    try:
        return float(arr)
    except Exception:
        return np.nan


def safe_cleanup(path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


sites = pd.read_csv(SITES_FILE)

log("")
log("=" * 72)
log(f"EN4 full run started: {START_YEAR}-{END_YEAR}, member={MEMBER}")
log(f"Cleanup raw after successful year: {CLEAN_RAW_AFTER_SUCCESS}")
log("=" * 72)

# Preserve depth table from the successful 1900 test if already available.
depth_table_written = (TABLE_DIR / "en4_depth_levels.csv").exists()

provenance_accumulator = []

for year in range(START_YEAR, END_YEAR + 1):
    checkpoint = YEARLY_DIR / f"en4_{year}_monthly_all_depths.csv"

    # A valid completed checkpoint for 3 sites x 12 months x 42 depths = 1512 rows.
    if checkpoint.exists():
        try:
            chk = pd.read_csv(checkpoint)
            if (
                len(chk) >= 3 * 12 * 42
                and chk["site"].nunique() == 3
                and chk["month"].nunique() == 12
            ):
                log(f"[{year}] checkpoint complete; skipping download.")
                continue
            else:
                log(f"[{year}] incomplete checkpoint detected; rebuilding.")
                checkpoint.unlink()
        except Exception:
            log(f"[{year}] unreadable checkpoint detected; rebuilding.")
            checkpoint.unlink(missing_ok=True)

    year_dir = RAW_ROOT / str(year)
    extract_dir = year_dir / "unzipped"
    year_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_name = f"EN.4.2.2.analyses.{MEMBER}.{year}.zip"
    zip_file = year_dir / zip_name
    part_file = year_dir / (zip_name + ".part")
    url = f"{BASE_URL}/{zip_name}"

    log("")
    log(f"--- YEAR {year} ---")
    download_zip(url, zip_file, part_file)

    log(f"[{year}] extracting ZIP...")
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(extract_dir)

    nc_files = sorted(extract_dir.rglob("*.nc"))
    if len(nc_files) != 12:
        raise RuntimeError(
            f"[{year}] expected 12 monthly NetCDF files, found {len(nc_files)}"
        )

    rows = []

    for month_index, nc in enumerate(nc_files, start=1):
        ds = xr.open_dataset(nc, decode_cf=True)
        names = identify_names(ds)

        depths = np.asarray(ds[names["depth"]].values, dtype=float).ravel()

        if not depth_table_written:
            pd.DataFrame(
                {
                    "level_index_0based": np.arange(len(depths)),
                    "depth_m": depths,
                }
            ).to_csv(TABLE_DIR / "en4_depth_levels.csv", index=False)
            depth_table_written = True

        dataset_lons = np.asarray(ds[names["lon"]].values)

        # Derive month robustly from file time coordinate where possible.
        try:
            month = int(pd.to_datetime(ds["time"].values[0]).month)
        except Exception:
            month = month_index

        for s in sites.itertuples(index=False):
            site_lat = float(s.latitude)
            site_lon = float(s.longitude)
            query_lon = normalize_lon_for_dataset(site_lon, dataset_lons)

            point = ds.sel(
                {
                    names["lat"]: site_lat,
                    names["lon"]: query_lon,
                },
                method="nearest",
            ).load()

            grid_lat = float(point[names["lat"]].values)
            grid_lon_raw = float(point[names["lon"]].values)
            grid_lon = display_lon(grid_lon_raw)
            dist_km = haversine_km(
                site_lat, site_lon, grid_lat, grid_lon
            )

            # Extract all 42 depth levels vectorized.
            temp = np.asarray(
                point[names["temperature"]].values
            ).squeeze().astype(float)
            sal = np.asarray(
                point[names["salinity"]].values
            ).squeeze().astype(float)
            temp_unc = np.asarray(
                point[names["temperature_uncertainty"]].values
            ).squeeze().astype(float)
            sal_unc = np.asarray(
                point[names["salinity_uncertainty"]].values
            ).squeeze().astype(float)
            temp_w = np.asarray(
                point[names["temperature_observation_weights"]].values
            ).squeeze().astype(float)
            sal_w = np.asarray(
                point[names["salinity_observation_weights"]].values
            ).squeeze().astype(float)

            # Defensive temperature conversion.
            if np.nanmedian(temp) > 100:
                temp = temp - 273.15
                temp_decode_note = "manual_kelvin_to_celsius_applied"
            else:
                temp_decode_note = "CF_decoded_temperature"

            for i, depth_m in enumerate(depths):
                rows.append(
                    {
                        "site": s.site,
                        "site_latitude": site_lat,
                        "site_longitude": site_lon,
                        "year": year,
                        "month": month,
                        "depth_level_index": i,
                        "depth_m": float(depth_m),
                        "potential_temperature_degC": float(temp[i])
                        if np.isfinite(temp[i]) else np.nan,
                        "practical_salinity": float(sal[i])
                        if np.isfinite(sal[i]) else np.nan,
                        "temperature_uncertainty": float(temp_unc[i])
                        if np.isfinite(temp_unc[i]) else np.nan,
                        "salinity_uncertainty": float(sal_unc[i])
                        if np.isfinite(sal_unc[i]) else np.nan,
                        "temperature_observation_weight": float(temp_w[i])
                        if np.isfinite(temp_w[i]) else np.nan,
                        "salinity_observation_weight": float(sal_w[i])
                        if np.isfinite(sal_w[i]) else np.nan,
                        "grid_latitude": grid_lat,
                        "grid_longitude": grid_lon,
                        "grid_center_distance_km": dist_km,
                        "dataset": "EN.4.2.2",
                        "bias_correction_member": MEMBER,
                        "data_class": (
                            "observation_constrained_objective_analysis"
                        ),
                        "spatial_resolution": "1 degree",
                        "spatial_extraction": "nearest_grid_cell",
                        "requested_buffer_km": 70,
                        "buffer_applied": False,
                        "temperature_decode_note": temp_decode_note,
                        "source_file": nc.name,
                    }
                )

            provenance_accumulator.append(
                {
                    "site": s.site,
                    "site_latitude": site_lat,
                    "site_longitude": site_lon,
                    "en4_grid_latitude": grid_lat,
                    "en4_grid_longitude": grid_lon,
                    "site_to_grid_center_km": dist_km,
                    "dataset": "EN.4.2.2",
                    "bias_correction_member": MEMBER,
                    "data_class": (
                        "observation_constrained_objective_analysis"
                    ),
                    "spatial_resolution": "1 degree",
                    "requested_buffer_km": 70,
                    "buffer_applied": False,
                }
            )

        ds.close()

    yearly = pd.DataFrame(rows).sort_values(
        ["site", "month", "depth_level_index"]
    )

    expected_rows = 3 * 12 * 42
    if len(yearly) != expected_rows:
        raise RuntimeError(
            f"[{year}] expected {expected_rows} rows but produced {len(yearly)}"
        )

    yearly.to_csv(checkpoint, index=False)

    # Verify written checkpoint before deleting raw files.
    verify = pd.read_csv(checkpoint)
    if len(verify) != expected_rows:
        raise RuntimeError(
            f"[{year}] checkpoint verification failed after write."
        )

    log(
        f"[{year}] checkpoint saved: {checkpoint.name} "
        f"({len(yearly)} rows)"
    )

    if CLEAN_RAW_AFTER_SUCCESS:
        log(f"[{year}] cleaning large raw ZIP/NetCDF files...")
        safe_cleanup(extract_dir)
        safe_cleanup(zip_file)
        safe_cleanup(part_file)

        # Remove now-empty year directory.
        try:
            year_dir.rmdir()
        except OSError:
            pass

# Combine all annual checkpoints.
log("")
log("Combining annual checkpoints...")

year_files = [
    YEARLY_DIR / f"en4_{y}_monthly_all_depths.csv"
    for y in range(START_YEAR, END_YEAR + 1)
]

missing = [str(p) for p in year_files if not p.exists()]
if missing:
    raise RuntimeError(
        f"Cannot combine: {len(missing)} annual checkpoints are missing."
    )

monthly = pd.concat(
    (pd.read_csv(p) for p in year_files),
    ignore_index=True,
)

monthly = monthly.sort_values(
    ["site", "year", "month", "depth_level_index"]
)

monthly_out = OUT_DIR / "en4_monthly_1900_2018_all_depths.csv"
monthly.to_csv(monthly_out, index=False)

log(f"Combined monthly rows: {len(monthly):,}")
log(f"Saved: {monthly_out}")

# Annual summaries by site and exact EN4 depth.
annual = (
    monthly.groupby(
        [
            "site",
            "site_latitude",
            "site_longitude",
            "year",
            "depth_level_index",
            "depth_m",
        ],
        as_index=False,
    )
    .agg(
        temperature_mean_degC=("potential_temperature_degC", "mean"),
        temperature_sd_monthly_degC=("potential_temperature_degC", "std"),
        temperature_min_degC=("potential_temperature_degC", "min"),
        temperature_max_degC=("potential_temperature_degC", "max"),
        salinity_mean=("practical_salinity", "mean"),
        salinity_sd_monthly=("practical_salinity", "std"),
        salinity_min=("practical_salinity", "min"),
        salinity_max=("practical_salinity", "max"),
        temperature_uncertainty_mean=(
            "temperature_uncertainty", "mean"
        ),
        salinity_uncertainty_mean=(
            "salinity_uncertainty", "mean"
        ),
        temperature_observation_weight_mean=(
            "temperature_observation_weight", "mean"
        ),
        salinity_observation_weight_mean=(
            "salinity_observation_weight", "mean"
        ),
        n_temperature_months=("potential_temperature_degC", "count"),
        n_salinity_months=("practical_salinity", "count"),
        grid_latitude=("grid_latitude", "first"),
        grid_longitude=("grid_longitude", "first"),
        grid_center_distance_km=("grid_center_distance_km", "first"),
    )
)

annual["dataset"] = "EN.4.2.2"
annual["bias_correction_member"] = MEMBER
annual["data_class"] = "observation_constrained_objective_analysis"
annual["spatial_resolution"] = "1 degree"
annual["spatial_extraction"] = "nearest_grid_cell"
annual["requested_buffer_km"] = 70
annual["buffer_applied"] = False

annual_out = OUT_DIR / "en4_annual_1900_2018_all_depths.csv"
annual.to_csv(annual_out, index=False)
log(f"Annual rows: {len(annual):,}")
log(f"Saved: {annual_out}")

# Site/grid provenance.
prov_cols = [
    "site",
    "site_latitude",
    "site_longitude",
    "grid_latitude",
    "grid_longitude",
    "grid_center_distance_km",
]

prov = (
    monthly[prov_cols]
    .drop_duplicates()
    .rename(
        columns={
            "grid_latitude": "en4_grid_latitude",
            "grid_longitude": "en4_grid_longitude",
            "grid_center_distance_km": "site_to_grid_center_km",
        }
    )
    .sort_values("site")
)

prov["dataset"] = "EN.4.2.2"
prov["bias_correction_member"] = MEMBER
prov["data_class"] = "observation_constrained_objective_analysis"
prov["spatial_resolution"] = "1 degree"
prov["period_used"] = f"{START_YEAR}-{END_YEAR}"

prov_out = TABLE_DIR / "en4_site_grid_provenance.csv"
prov.to_csv(prov_out, index=False)

# Coverage / observational-influence summary.
coverage = (
    monthly.groupby(["site", "year"], as_index=False)
    .agg(
        temp_weight_mean_all_depths=(
            "temperature_observation_weight", "mean"
        ),
        sal_weight_mean_all_depths=(
            "salinity_observation_weight", "mean"
        ),
        temp_uncertainty_mean_all_depths=(
            "temperature_uncertainty", "mean"
        ),
        sal_uncertainty_mean_all_depths=(
            "salinity_uncertainty", "mean"
        ),
        valid_temp_values=("potential_temperature_degC", "count"),
        valid_sal_values=("practical_salinity", "count"),
    )
)

coverage_out = TABLE_DIR / "en4_data_coverage_summary.csv"
coverage.to_csv(coverage_out, index=False)

log("")
log("EN4 FULL EXTRACTION COMPLETE.")
log(f"Monthly all-depth file: {monthly_out}")
log(f"Annual all-depth file:  {annual_out}")
log(f"Provenance:             {prov_out}")
log(f"Coverage summary:       {coverage_out}")
log("")
log("Expected combined monthly rows:")
log(f"  3 sites x 119 years x 12 months x 42 depths = {3*119*12*42:,}")
log(f"Actual combined monthly rows: {len(monthly):,}")
