"""
05_population_extract_5arcmin.py
================================

Extract historical human population around:
  Jolo      6.1098, 120.9775
  Masinloc 15.55672, 119.93036
  Palapag  12.63476, 125.00978

Dataset intended:
  ISIMIP3a InputData / socioeconomic / pop / histsoc
  population_histsoc_5arcmin_annual
  1850-1900 + 1901-2021

Study period:
  1900-2018

Spatial summaries:
  circular buffers of 25, 50, and 70 km
  grid-cell-center inclusion using haversine distance

Evidence classification:
  historical_population_reconstruction

IMPORTANT:
The previous broad ISIMIP cutout ZIP contains some duplicate basenames from
multiple ISIMIP generations. This script checks duplicate 5-arcmin members
before using them. If same-name duplicates differ in CRC or size, the script
ABORTS rather than silently mixing products.
"""

from pathlib import Path
from collections import defaultdict
import csv
import math
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(r"D:\temporal_environmental_data")

RAW_DIR = ROOT / "01_raw" / "HYDE_population"
ZIP_PATH = RAW_DIR / "population_ISIMIP3a_Philippines_subset.zip"
CLEAN_DIR = RAW_DIR / "ISIMIP3a_histsoc_5arcmin_clean"

EXTRACTED_OUT = ROOT / "03_extracted"
TABLE_OUT = ROOT / "04_tables"
FIG_OUT = ROOT / "05_figures"
LOG_OUT = ROOT / "07_logs"

for p in (CLEAN_DIR, EXTRACTED_OUT, TABLE_OUT, FIG_OUT, LOG_OUT):
    p.mkdir(parents=True, exist_ok=True)

SITES = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}

BUFFERS_KM = [25, 50, 70]
START_YEAR = 1900
END_YEAR = 2018

TARGET_STEMS = [
    "population_histsoc_5arcmin_annual_1850_1900",
    "population_histsoc_5arcmin_annual_1901_2021",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    R = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(a))


def find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables and ds[name].ndim == 1:
            return name
    return None


def find_population_var(ds, lat_name, lon_name, time_name):
    """Identify the population data variable without assuming its exact name."""
    preferred = [
        "population",
        "pop",
        "total_population",
        "population_count",
    ]
    for name in preferred:
        if name in ds.data_vars:
            return name

    ranked = []
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        score = 0
        lname = name.lower()
        if "population" in lname:
            score += 20
        if lname == "pop" or lname.startswith("pop"):
            score += 10
        if time_name in dims:
            score += 5
        if lat_name in dims and lon_name in dims:
            score += 5
        if score > 0:
            ranked.append((score, name))

    if not ranked:
        raise RuntimeError(
            "Could not identify the population variable. "
            f"Available variables: {list(ds.data_vars)}"
        )

    ranked.sort(reverse=True)
    return ranked[0][1]


def years_from_time(da):
    """Robustly obtain integer years from datetime64/cftime/numeric CF time."""
    try:
        return np.asarray(da.dt.year.values, dtype=int)
    except Exception:
        pass

    vals = np.asarray(da.values)

    years = []
    for v in vals:
        if hasattr(v, "year"):
            years.append(int(v.year))
        else:
            text = str(v)
            try:
                years.append(int(text[:4]))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not decode year from time value {v!r}"
                ) from exc
    return np.asarray(years, dtype=int)


print("=" * 78)
print("ISIMIP3a HISTORICAL POPULATION EXTRACTION — 5 ARCMIN")
print("=" * 78)
print(f"Input ZIP: {ZIP_PATH}")
print(f"Study period: {START_YEAR}-{END_YEAR}")
print(f"Buffers: {BUFFERS_KM} km")
print()

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Input ZIP not found:\n{ZIP_PATH}\n"
        "Do not re-run this extraction until the population cutout ZIP exists."
    )

# -------------------------------------------------------------------------
# 1. Locate intended 5-arcmin archive members and detect duplicate conflicts.
# -------------------------------------------------------------------------
print("STEP 1 — Inspecting ZIP members and checking duplicate filenames")
print("-" * 78)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    infos = z.infolist()

    matched = {}
    for stem in TARGET_STEMS:
        hits = [
            info for info in infos
            if stem in Path(info.filename).name
            and info.filename.lower().endswith(".nc")
        ]

        if not hits:
            raise RuntimeError(
                f"Required archive member not found for:\n  {stem}"
            )

        print(f"\n{stem}")
        print(f"  matching ZIP entries: {len(hits)}")

        signatures = defaultdict(list)
        for i, info in enumerate(hits, start=1):
            sig = (info.CRC, info.file_size)
            signatures[sig].append(info)
            print(
                f"  [{i}] {info.filename}\n"
                f"      size={info.file_size:,} bytes  CRC={info.CRC:08x}"
            )

        if len(signatures) > 1:
            print()
            print("ERROR: same-name candidate files are NOT byte-identical.")
            print("This broad ZIP mixed different ISIMIP products.")
            print(
                "Extraction is being stopped intentionally so we do not "
                "silently use the wrong historical population dataset."
            )
            print()
            print("Paste this diagnostic output back into ChatGPT.")
            sys.exit(3)

        if len(hits) > 1:
            print("  Duplicate entries are byte-identical: SAFE to use one copy.")

        matched[stem] = hits[0]

    # Cleanly extract only one validated copy of each required file.
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    clean_files = []

    for stem, info in matched.items():
        basename = Path(info.filename).name
        dest = CLEAN_DIR / basename

        with z.open(info, "r") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

        clean_files.append(dest)
        print(f"\nExtracted clean file:\n  {dest}")

# -------------------------------------------------------------------------
# 2. Open both files and identify coordinates/variable.
# -------------------------------------------------------------------------
print()
print("STEP 2 — Validating NetCDF structure")
print("-" * 78)

datasets = []
metadata_rows = []

for nc in clean_files:
    ds = xr.open_dataset(nc, decode_cf=True)

    lat_name = find_coord(ds, ["lat", "latitude", "y"])
    lon_name = find_coord(ds, ["lon", "longitude", "x"])
    time_name = find_coord(ds, ["time", "year"])

    if lat_name is None or lon_name is None or time_name is None:
        ds.close()
        raise RuntimeError(
            f"Could not identify lat/lon/time coordinates in {nc.name}.\n"
            f"Coordinates: {list(ds.coords)}\nVariables: {list(ds.variables)}"
        )

    pop_var = find_population_var(ds, lat_name, lon_name, time_name)
    years = years_from_time(ds[time_name])

    units = ds[pop_var].attrs.get("units", "")
    long_name = ds[pop_var].attrs.get("long_name", "")
    standard_name = ds[pop_var].attrs.get("standard_name", "")

    print(f"\nFile: {nc.name}")
    print(f"  dimensions: {dict(ds.sizes)}")
    print(f"  population variable: {pop_var}")
    print(f"  units: {units!r}")
    print(f"  long_name: {long_name!r}")
    print(f"  standard_name: {standard_name!r}")
    print(f"  latitude coord: {lat_name}")
    print(f"  longitude coord: {lon_name}")
    print(f"  time coord: {time_name}")
    print(f"  years: {years.min()} to {years.max()}")
    print(
        f"  lat range: {float(ds[lat_name].min()):.4f} "
        f"to {float(ds[lat_name].max()):.4f}"
    )
    print(
        f"  lon range: {float(ds[lon_name].min()):.4f} "
        f"to {float(ds[lon_name].max()):.4f}"
    )

    metadata_rows.append({
        "file": nc.name,
        "population_variable": pop_var,
        "units": units,
        "long_name": long_name,
        "standard_name": standard_name,
        "lat_coord": lat_name,
        "lon_coord": lon_name,
        "time_coord": time_name,
        "first_year": int(years.min()),
        "last_year": int(years.max()),
        "n_lat": int(ds.sizes[lat_name]),
        "n_lon": int(ds.sizes[lon_name]),
        "n_time": int(ds.sizes[time_name]),
    })

    datasets.append(
        (ds, nc, lat_name, lon_name, time_name, pop_var, years)
    )

# Ensure both files use compatible structure.
base = datasets[0]
for current in datasets[1:]:
    if (
        current[2] != base[2]
        or current[3] != base[3]
        or current[4] != base[4]
        or current[5] != base[5]
    ):
        for x in datasets:
            x[0].close()
        raise RuntimeError(
            "The two historical files do not use compatible "
            "coordinate/variable names."
        )

lat_name = base[2]
lon_name = base[3]
time_name = base[4]
pop_var = base[5]

# -------------------------------------------------------------------------
# 3. Concatenate the two files and restrict to 1900-2018.
# -------------------------------------------------------------------------
print()
print("STEP 3 — Combining years 1900-2018")
print("-" * 78)

pieces = []
for ds, nc, latn, lonn, timen, pvar, years in datasets:
    da = ds[pvar]
    keep = np.where((years >= START_YEAR) & (years <= END_YEAR))[0]
    if keep.size:
        piece = da.isel({timen: keep}).load()
        # Attach simple integer year coordinate to simplify later processing.
        selected_years = years[keep]
        piece = piece.assign_coords(year=(timen, selected_years))
        pieces.append(piece)

for ds, *_ in datasets:
    ds.close()

if not pieces:
    raise RuntimeError("No population data were found for 1900-2018.")

pop = xr.concat(pieces, dim=time_name)

all_years = np.asarray(pop["year"].values, dtype=int)

# Remove accidental duplicate years if any.
_, unique_idx = np.unique(all_years, return_index=True)
unique_idx = np.sort(unique_idx)
pop = pop.isel({time_name: unique_idx})
all_years = np.asarray(pop["year"].values, dtype=int)

order = np.argsort(all_years)
pop = pop.isel({time_name: order})
all_years = np.asarray(pop["year"].values, dtype=int)

expected_years = np.arange(START_YEAR, END_YEAR + 1)
missing_years = sorted(set(expected_years) - set(all_years))

print(f"Combined years: {all_years.min()} to {all_years.max()}")
print(f"Number of annual steps: {len(all_years)}")
print(f"Expected annual steps: {len(expected_years)}")

if missing_years:
    raise RuntimeError(
        "Missing years in combined dataset:\n"
        + ", ".join(map(str, missing_years))
    )

if len(all_years) != len(expected_years):
    raise RuntimeError(
        "Unexpected year count after de-duplication."
    )

# -------------------------------------------------------------------------
# 4. Prepare spatial grid.
# -------------------------------------------------------------------------
print()
print("STEP 4 — Calculating 25/50/70-km spatial buffers")
print("-" * 78)

lat_vals = np.asarray(pop[lat_name].values, dtype=float)
lon_vals = np.asarray(pop[lon_name].values, dtype=float)

# Convert 0..360 to -180..180 only for distance calculations, if necessary.
lon_for_dist = ((lon_vals + 180.0) % 360.0) - 180.0

lon2d, lat2d = np.meshgrid(lon_for_dist, lat_vals)

rows = []
buffer_provenance = []

# Determine dimension order and normalize to time, lat, lon.
pop_tll = pop.transpose(time_name, lat_name, lon_name)
arr = np.asarray(pop_tll.values, dtype=float)

for site, (site_lat, site_lon) in SITES.items():
    dist = haversine_km(site_lat, site_lon, lat2d, lon2d)

    nearest_flat = int(np.nanargmin(dist))
    nearest_i, nearest_j = np.unravel_index(nearest_flat, dist.shape)
    nearest_dist = float(dist[nearest_i, nearest_j])
    nearest_lat = float(lat_vals[nearest_i])
    nearest_lon = float(lon_for_dist[nearest_j])

    print(
        f"\n{site}: nearest 5' cell center "
        f"({nearest_lat:.4f}, {nearest_lon:.4f}), "
        f"{nearest_dist:.2f} km"
    )

    for radius in BUFFERS_KM:
        mask = dist <= radius
        n_grid_cells = int(mask.sum())

        if n_grid_cells == 0:
            raise RuntimeError(
                f"No grid-cell centers fell within {radius} km of {site}."
            )

        # Extract all cells selected by the spatial mask for every year.
        selected = arr[:, mask]

        n_nonmissing = np.sum(np.isfinite(selected), axis=1)

        # Human population around a marine site is total people in selected cells.
        # NaN cells (commonly ocean / unavailable) contribute zero to the total.
        totals = np.nansum(selected, axis=1)

        # If literally every selected value is NaN for a year, flag result NaN.
        all_nan = n_nonmissing == 0
        totals = totals.astype(float)
        totals[all_nan] = np.nan

        print(
            f"  {radius:>2} km: {n_grid_cells} grid-cell centers selected"
        )

        for idx, year in enumerate(all_years):
            rows.append({
                "site": site,
                "latitude": site_lat,
                "longitude": site_lon,
                "year": int(year),
                "buffer_km": int(radius),
                "population_sum": float(totals[idx])
                    if np.isfinite(totals[idx]) else np.nan,
                "n_grid_cells": n_grid_cells,
                "n_nonmissing_cells": int(n_nonmissing[idx]),
                "dataset": "ISIMIP3a histsoc",
                "variable": pop_var,
                "data_class": "historical_population_reconstruction",
                "spatial_resolution": "5 arcmin (~9 km at equator)",
                "buffer_method": "grid_cell_center_haversine_inclusion",
            })

        buffer_provenance.append({
            "site": site,
            "site_latitude": site_lat,
            "site_longitude": site_lon,
            "buffer_km": radius,
            "n_grid_cells": n_grid_cells,
            "nearest_grid_latitude": nearest_lat,
            "nearest_grid_longitude": nearest_lon,
            "nearest_grid_distance_km": nearest_dist,
            "spatial_resolution": "5 arcmin",
            "buffer_method": "grid_cell_center_haversine_inclusion",
            "data_class": "historical_population_reconstruction",
        })

df = pd.DataFrame(rows)

# -------------------------------------------------------------------------
# 5. Derived summary tables.
# -------------------------------------------------------------------------
print()
print("STEP 5 — Writing extraction and summary tables")
print("-" * 78)

long_csv = EXTRACTED_OUT / "population_ISIMIP3a_5arcmin_buffers_1900_2018.csv"
df.to_csv(long_csv, index=False)

wide = df.pivot(
    index=["site", "latitude", "longitude", "year"],
    columns="buffer_km",
    values="population_sum",
).reset_index()

wide.columns.name = None
wide = wide.rename(columns={
    25: "population_25km",
    50: "population_50km",
    70: "population_70km",
})

wide_csv = EXTRACTED_OUT / "population_ISIMIP3a_5arcmin_annual_1900_2018.csv"
wide.to_csv(wide_csv, index=False)

prov_csv = TABLE_OUT / "population_ISIMIP3a_5arcmin_buffer_provenance.csv"
pd.DataFrame(buffer_provenance).to_csv(prov_csv, index=False)

meta_csv = TABLE_OUT / "population_ISIMIP3a_5arcmin_file_metadata.csv"
pd.DataFrame(metadata_rows).to_csv(meta_csv, index=False)

summary_rows = []

for (site, radius), g in df.groupby(["site", "buffer_km"]):
    g = g.sort_values("year")

    v1900 = g.loc[g.year == 1900, "population_sum"]
    v2018 = g.loc[g.year == 2018, "population_sum"]

    pop1900 = float(v1900.iloc[0]) if len(v1900) else np.nan
    pop2018 = float(v2018.iloc[0]) if len(v2018) else np.nan

    early = g.loc[
        (g.year >= 1900) & (g.year <= 1919),
        "population_sum"
    ].mean()

    modern = g.loc[
        (g.year >= 1999) & (g.year <= 2018),
        "population_sum"
    ].mean()

    absolute_change = pop2018 - pop1900

    if np.isfinite(pop1900) and pop1900 != 0:
        percent_change = 100.0 * absolute_change / pop1900
        fold_change = pop2018 / pop1900
    else:
        percent_change = np.nan
        fold_change = np.nan

    if np.isfinite(early) and early != 0:
        period_percent_change = 100.0 * (modern - early) / early
        period_fold_change = modern / early
    else:
        period_percent_change = np.nan
        period_fold_change = np.nan

    summary_rows.append({
        "site": site,
        "buffer_km": int(radius),
        "population_1900": pop1900,
        "population_2018": pop2018,
        "absolute_change_1900_2018": absolute_change,
        "percent_change_1900_2018": percent_change,
        "fold_change_1900_2018": fold_change,
        "mean_population_1900_1919": float(early),
        "mean_population_1999_2018": float(modern),
        "percent_change_20yr_period_means": period_percent_change,
        "fold_change_20yr_period_means": period_fold_change,
        "data_class": "historical_population_reconstruction",
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = TABLE_OUT / "population_ISIMIP3a_5arcmin_change_summary.csv"
summary_df.to_csv(summary_csv, index=False)

# -------------------------------------------------------------------------
# 6. Basic trend table (OLS slope, R2) for descriptive use.
# -------------------------------------------------------------------------
trend_rows = []

for (site, radius), g in df.groupby(["site", "buffer_km"]):
    g = g.dropna(subset=["population_sum"]).sort_values("year")

    x = g["year"].to_numpy(dtype=float)
    y = g["population_sum"].to_numpy(dtype=float)

    if len(g) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope * x

        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        slope = intercept = r2 = np.nan

    trend_rows.append({
        "site": site,
        "buffer_km": int(radius),
        "n_years": int(len(g)),
        "ols_slope_people_per_year": float(slope),
        "ols_slope_people_per_decade": float(slope * 10.0),
        "ols_intercept": float(intercept),
        "r_squared": float(r2),
        "note": (
            "Descriptive OLS only; Mann-Kendall/Sen slope can be added "
            "during formal statistical analysis."
        ),
    })

trend_df = pd.DataFrame(trend_rows)
trend_csv = TABLE_OUT / "population_ISIMIP3a_5arcmin_trend_summary.csv"
trend_df.to_csv(trend_csv, index=False)

# -------------------------------------------------------------------------
# 7. Figure.
# -------------------------------------------------------------------------
fig_png = FIG_OUT / "Fig_population_ISIMIP3a_5arcmin_1900_2018.png"
fig_pdf = FIG_OUT / "Fig_population_ISIMIP3a_5arcmin_1900_2018.pdf"

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 10),
        sharex=True,
        constrained_layout=True,
    )

    for ax, site in zip(axes, SITES):
        site_df = df[df.site == site]

        for radius in BUFFERS_KM:
            g = site_df[site_df.buffer_km == radius].sort_values("year")
            ax.plot(
                g["year"],
                g["population_sum"],
                label=f"{radius} km",
                linewidth=1.6,
            )

        ax.set_title(site)
        ax.set_ylabel("Population (people)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Year")

    fig.suptitle(
        "Historical population around Philippine marine sites\n"
        "ISIMIP3a histsoc, 5 arcmin, grid-cell-center buffers",
        fontsize=12,
    )

    fig.savefig(fig_png, dpi=300)
    fig.savefig(fig_pdf)
    plt.close(fig)

    print(f"Figure PNG: {fig_png}")
    print(f"Figure PDF: {fig_pdf}")

except Exception as exc:
    print()
    print("WARNING: figure creation skipped.")
    print(f"Reason: {exc}")

# -------------------------------------------------------------------------
# 8. Human-readable log.
# -------------------------------------------------------------------------
log_txt = LOG_OUT / "population_ISIMIP3a_5arcmin_extraction_log.txt"

with open(log_txt, "w", encoding="utf-8") as f:
    f.write("ISIMIP3a historical population extraction\n")
    f.write("=" * 60 + "\n")
    f.write(f"Study period: {START_YEAR}-{END_YEAR}\n")
    f.write("Resolution: 5 arcmin\n")
    f.write("Data class: historical_population_reconstruction\n")
    f.write(
        "Spatial method: grid-cell-center inclusion using haversine distance\n"
    )
    f.write(
        "Population measure: sum of gridded population counts in each buffer\n"
    )
    f.write(
        "NaN grid cells contribute zero unless every selected grid cell is NaN\n"
    )
    f.write("\nSites:\n")
    for site, coords in SITES.items():
        f.write(f"  {site}: {coords[0]}, {coords[1]}\n")
    f.write("\nBuffers: 25, 50, 70 km\n")
    f.write("\nOutput files:\n")
    for p in [
        long_csv,
        wide_csv,
        prov_csv,
        meta_csv,
        summary_csv,
        trend_csv,
        fig_png,
        fig_pdf,
    ]:
        f.write(f"  {p}\n")

print()
print("=" * 78)
print("EXTRACTION COMPLETE")
print("=" * 78)
print(f"Long annual table:  {long_csv}")
print(f"Wide annual table:  {wide_csv}")
print(f"Change summary:     {summary_csv}")
print(f"Trend summary:      {trend_csv}")
print(f"Buffer provenance:  {prov_csv}")
print(f"File metadata:      {meta_csv}")
print(f"Log:                {log_txt}")
print()
print(f"Expected long rows: 3 sites × 3 buffers × 119 years = {3*3*119}")
print(f"Actual long rows:   {len(df)}")
print()
print("Preview — change summary:")
print(summary_df.to_string(index=False))
