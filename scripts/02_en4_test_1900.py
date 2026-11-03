"""
EN4 Phase 2A — test year 1900 only
----------------------------------
Downloads Met Office EN.4.2.2 G10 objective analyses for 1900,
unpacks the 12 monthly NetCDF files, inspects dimensions/variables/depths,
and extracts temperature + salinity at the nearest grid cell for:

Jolo      6.1098, 120.9775
Masinloc 15.55672, 119.93036
Palapag  12.63476, 125.00978

This is a TEST script. It does NOT yet launch the full 1900–2018 run.

Data provenance:
  dataset       = EN.4.2.2
  bias member   = g10
  data_class    = observation_constrained_objective_analysis
  grid          = 1 degree
  vertical      = 42 standard levels

Outputs:
  01_raw/EN4/1900/...
  03_extracted/en4_test_1900_monthly.csv
  04_tables/en4_test_1900_grid_provenance.csv
  04_tables/en4_depth_levels.csv
  07_logs/en4_test_1900_structure.txt
"""

from pathlib import Path
import math
import zipfile
import time
import requests
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(r"D:\temporal_environmental_data")
SITES_FILE = ROOT / "00_metadata" / "sites.csv"

RAW_DIR = ROOT / "01_raw" / "EN4" / "1900"
OUT_DIR = ROOT / "03_extracted"
TABLE_DIR = ROOT / "04_tables"
LOG_DIR = ROOT / "07_logs"

for d in [RAW_DIR, OUT_DIR, TABLE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

YEAR = 1900
MEMBER = "g10"
ZIP_NAME = f"EN.4.2.2.analyses.{MEMBER}.{YEAR}.zip"
URL = (
    "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/"
    f"EN.4.2.2/{ZIP_NAME}"
)

ZIP_FILE = RAW_DIR / ZIP_NAME
PART_FILE = RAW_DIR / (ZIP_NAME + ".part")
EXTRACT_DIR = RAW_DIR / "unzipped"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def normalize_lon_for_dataset(site_lon, dataset_lons):
    """Return site longitude compatible with either -180..180 or 0..360 datasets."""
    lmin = float(np.nanmin(dataset_lons))
    lmax = float(np.nanmax(dataset_lons))
    if lmin >= 0 and site_lon < 0:
        return site_lon % 360
    if lmax > 180 and site_lon < 0:
        return site_lon % 360
    return site_lon


def display_lon(lon):
    lon = float(lon)
    return lon - 360 if lon > 180 else lon


def download_file(max_attempts=5):
    if ZIP_FILE.exists():
        try:
            with zipfile.ZipFile(ZIP_FILE, "r") as z:
                bad = z.testzip()
                if bad is None:
                    print(f"Using cached valid ZIP: {ZIP_FILE}")
                    return
        except zipfile.BadZipFile:
            print("Cached ZIP is invalid; deleting and re-downloading.")
            ZIP_FILE.unlink()

    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(1, max_attempts + 1):
        try:
            if PART_FILE.exists():
                PART_FILE.unlink()

            print(f"Downloading EN4 {YEAR} objective analyses (attempt {attempt}/{max_attempts})...")
            print(URL)

            with requests.get(
                URL,
                stream=True,
                timeout=(30, 300),
                headers=headers,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0))
                ctype = r.headers.get("Content-Type", "")
                print(f"Content type: {ctype}")
                if total:
                    print(f"Expected download size: {total / 1024 / 1024:.2f} MB")

                downloaded = 0
                with open(PART_FILE, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                print(
                                    f"\rDownloaded {downloaded/1024/1024:.2f} MB "
                                    f"({pct:.1f}%)",
                                    end="",
                                    flush=True,
                                )
            print()

            if total and PART_FILE.stat().st_size != total:
                raise IOError(
                    f"Incomplete download: {PART_FILE.stat().st_size} bytes "
                    f"of expected {total}"
                )

            with zipfile.ZipFile(PART_FILE, "r") as z:
                bad = z.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Corrupted member: {bad}")

            PART_FILE.replace(ZIP_FILE)
            print(f"Validated ZIP saved: {ZIP_FILE}")
            return

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if PART_FILE.exists():
                PART_FILE.unlink()
            if attempt == max_attempts:
                raise
            time.sleep(3 * attempt)


print("=== EN4 Phase 2A: test year 1900 ===")

if not SITES_FILE.exists():
    raise FileNotFoundError(f"Missing site file: {SITES_FILE}")

download_file()

print("Unpacking NetCDF files...")
with zipfile.ZipFile(ZIP_FILE, "r") as z:
    z.extractall(EXTRACT_DIR)

nc_files = sorted(EXTRACT_DIR.rglob("*.nc"))
if len(nc_files) == 0:
    raise RuntimeError("No NetCDF files were found after unzipping.")

print(f"Found {len(nc_files)} NetCDF files.")
for f in nc_files:
    print(" ", f.name)

# Inspect first month.
first_file = nc_files[0]
ds0 = xr.open_dataset(first_file, decode_cf=True)

structure_lines = []
structure_lines.append(f"File: {first_file}")
structure_lines.append("")
structure_lines.append("DIMENSIONS:")
structure_lines.append(str(dict(ds0.sizes)))
structure_lines.append("")
structure_lines.append("COORDINATES:")
for name in ds0.coords:
    da = ds0[name]
    structure_lines.append(
        f"{name}: dims={da.dims}, shape={da.shape}, attrs={dict(da.attrs)}"
    )
structure_lines.append("")
structure_lines.append("DATA VARIABLES:")
for name in ds0.data_vars:
    da = ds0[name]
    structure_lines.append(
        f"{name}: dims={da.dims}, shape={da.shape}, attrs={dict(da.attrs)}"
    )

log_file = LOG_DIR / "en4_test_1900_structure.txt"
log_file.write_text("\n".join(structure_lines), encoding="utf-8")

print("\n--- Variables found ---")
print(list(ds0.data_vars))
print("--- Coordinates found ---")
print(list(ds0.coords))
print("--- Dimensions ---")
print(dict(ds0.sizes))

# Robust coordinate-name lookup.
def find_name(candidates, available):
    lower_map = {x.lower(): x for x in available}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not identify any of {candidates} in {list(available)}")


all_names = list(ds0.coords) + list(ds0.variables)

depth_name = find_name(["depth", "Depth", "DEPTH"], all_names)
lat_name = find_name(["lat", "latitude", "LATITUDE"], all_names)
lon_name = find_name(["lon", "longitude", "LONGITUDE"], all_names)
temp_name = find_name(["temperature", "TEMP", "temp"], list(ds0.data_vars))
sal_name = find_name(["salinity", "PSAL", "sal"], list(ds0.data_vars))

temp_unc_name = None
sal_unc_name = None
for candidate in ["temperature_uncertainty", "TEMP_UNCERTAINTY"]:
    if candidate in ds0.variables:
        temp_unc_name = candidate
for candidate in ["salinity_uncertainty", "SALINITY_UNCERTAINTY"]:
    if candidate in ds0.variables:
        sal_unc_name = candidate

depths = np.asarray(ds0[depth_name].values, dtype=float).ravel()

depth_table = pd.DataFrame(
    {
        "level_index_0based": np.arange(len(depths)),
        "depth_m": depths,
    }
)
depth_table.to_csv(TABLE_DIR / "en4_depth_levels.csv", index=False)

print(f"\nEN4 has {len(depths)} depth levels.")
print("First 15 depth levels (m):")
print(depths[:15])

# Target ecological depths: use nearest actual EN4 standard levels.
requested_depths = [5, 15, 25, 50]
actual_depths = {}
for target in requested_depths:
    idx = int(np.nanargmin(np.abs(depths - target)))
    actual_depths[target] = float(depths[idx])

print("\nRequested -> nearest actual EN4 depth:")
for req, actual in actual_depths.items():
    print(f"  {req:>3} m -> {actual:.3f} m")

ds0.close()

sites = pd.read_csv(SITES_FILE)
rows = []
provenance_rows = []

for nc in nc_files:
    ds = xr.open_dataset(nc, decode_cf=True)

    # Determine month from filename (expected YYYYMM near filename end).
    digits = "".join(ch for ch in nc.stem if ch.isdigit())
    month = None
    if len(digits) >= 6:
        # Find 1900MM occurrence preferentially.
        pos = digits.find(str(YEAR))
        if pos >= 0 and len(digits) >= pos + 6:
            month = int(digits[pos + 4:pos + 6])
    if month is None or month < 1 or month > 12:
        # Fall back to sequential order.
        month = nc_files.index(nc) + 1

    dataset_lons = np.asarray(ds[lon_name].values)

    for s in sites.itertuples(index=False):
        site_lat = float(s.latitude)
        site_lon = float(s.longitude)
        query_lon = normalize_lon_for_dataset(site_lon, dataset_lons)

        # Nearest 1° grid cell. We record distance and do not pretend
        # this is a true 70-km mean unless later spatial support permits it.
        point = ds.sel(
            {
                lat_name: site_lat,
                lon_name: query_lon,
            },
            method="nearest",
        )

        grid_lat = float(point[lat_name].values)
        grid_lon_raw = float(point[lon_name].values)
        grid_lon = display_lon(grid_lon_raw)
        dist_km = haversine_km(site_lat, site_lon, grid_lat, grid_lon)

        for req_depth, actual_depth in actual_depths.items():
            pz = point.sel({depth_name: actual_depth}, method="nearest")

            temp = float(np.asarray(pz[temp_name].values).squeeze())
            sal = float(np.asarray(pz[sal_name].values).squeeze())

            temp_unc = np.nan
            sal_unc = np.nan

            if temp_unc_name is not None:
                temp_unc = float(
                    np.asarray(pz[temp_unc_name].values).squeeze()
                )
            if sal_unc_name is not None:
                sal_unc = float(
                    np.asarray(pz[sal_unc_name].values).squeeze()
                )

            # xarray normally applies NetCDF scale_factor/add_offset automatically.
            # EN4 FAQ notes analysis temperature is stored in K with an offset;
            # decoded values should be checked for plausibility.
            if temp > 100:
                temp_c = temp - 273.15
                temp_decode_note = "manual_kelvin_to_celsius_applied"
            else:
                temp_c = temp
                temp_decode_note = "CF_decoding_produced_celsius_scale"

            if np.isfinite(temp_unc) and temp_unc > 100:
                # Uncertainty is a difference, so do NOT subtract 273.15.
                # This branch only flags implausible decoding.
                temp_unc_note = "CHECK_UNCERTAINTY_UNITS"
            else:
                temp_unc_note = "as_decoded"

            rows.append(
                {
                    "site": s.site,
                    "site_latitude": site_lat,
                    "site_longitude": site_lon,
                    "year": YEAR,
                    "month": month,
                    "requested_depth_m": req_depth,
                    "actual_en4_depth_m": actual_depth,
                    "potential_temperature_degC": temp_c,
                    "practical_salinity": sal,
                    "temperature_uncertainty": temp_unc,
                    "salinity_uncertainty": sal_unc,
                    "grid_latitude": grid_lat,
                    "grid_longitude": grid_lon,
                    "grid_center_distance_km": dist_km,
                    "dataset": "EN.4.2.2",
                    "bias_correction_member": MEMBER,
                    "data_class": "observation_constrained_objective_analysis",
                    "spatial_resolution": "1 degree",
                    "spatial_extraction": "nearest_grid_cell_test",
                    "requested_buffer_km": 70,
                    "buffer_applied": False,
                    "temperature_decode_note": temp_decode_note,
                    "temperature_uncertainty_note": temp_unc_note,
                    "source_file": nc.name,
                }
            )

        provenance_rows.append(
            {
                "site": s.site,
                "site_latitude": site_lat,
                "site_longitude": site_lon,
                "en4_grid_latitude": grid_lat,
                "en4_grid_longitude": grid_lon,
                "site_to_grid_center_km": dist_km,
                "dataset": "EN.4.2.2",
                "bias_correction_member": MEMBER,
                "data_class": "observation_constrained_objective_analysis",
                "spatial_resolution": "1 degree",
                "requested_buffer_km": 70,
                "buffer_applied": False,
                "test_year": YEAR,
            }
        )

    ds.close()

monthly = pd.DataFrame(rows).sort_values(
    ["site", "month", "requested_depth_m"]
)
monthly.to_csv(
    OUT_DIR / "en4_test_1900_monthly.csv",
    index=False,
)

prov = (
    pd.DataFrame(provenance_rows)
    .drop_duplicates(
        subset=["site", "en4_grid_latitude", "en4_grid_longitude"]
    )
    .sort_values("site")
)
prov.to_csv(
    TABLE_DIR / "en4_test_1900_grid_provenance.csv",
    index=False,
)

print("\n--- Site grid cells ---")
for r in prov.itertuples(index=False):
    print(
        f"{r.site}: site=({r.site_latitude:.5f}, {r.site_longitude:.5f}) -> "
        f"EN4 cell=({r.en4_grid_latitude:.2f}, {r.en4_grid_longitude:.2f}), "
        f"distance={r.site_to_grid_center_km:.1f} km"
    )

print("\n--- Quick plausibility summary ---")
summary = (
    monthly.groupby(["site", "requested_depth_m"], as_index=False)
    .agg(
        temp_mean_degC=("potential_temperature_degC", "mean"),
        salinity_mean=("practical_salinity", "mean"),
        temp_unc_mean=("temperature_uncertainty", "mean"),
        sal_unc_mean=("salinity_uncertainty", "mean"),
        n_months=("month", "nunique"),
    )
)
print(summary.to_string(index=False))

print("\nDONE.")
print(f"Raw ZIP:       {ZIP_FILE}")
print(f"Unpacked:      {EXTRACT_DIR}")
print(f"Structure log: {log_file}")
print(f"Depth table:   {TABLE_DIR / 'en4_depth_levels.csv'}")
print(f"Monthly test:  {OUT_DIR / 'en4_test_1900_monthly.csv'}")
print(f"Provenance:    {TABLE_DIR / 'en4_test_1900_grid_provenance.csv'}")
