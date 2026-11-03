"""
05_population_extract_5arcmin_CLEAN.py
======================================

Extract 1900-2018 historical population around Jolo, Masinloc, and Palapag
from the CLEAN ISIMIP3a histsoc 5-arcmin Philippines subset.

Inputs:
D:\temporal_environmental_data\01_raw\HYDE_population\
    ISIMIP3a_histsoc_5arcmin_clean\

Outputs:
03_extracted:
    population_ISIMIP3a_5arcmin_buffers_1900_2018.csv
    population_ISIMIP3a_5arcmin_annual_1900_2018.csv

04_tables:
    population_ISIMIP3a_5arcmin_buffer_provenance.csv
    population_ISIMIP3a_5arcmin_file_metadata.csv
    population_ISIMIP3a_5arcmin_change_summary.csv
    population_ISIMIP3a_5arcmin_trend_summary.csv

05_figures:
    Fig_population_ISIMIP3a_5arcmin_1900_2018.png
    Fig_population_ISIMIP3a_5arcmin_1900_2018.pdf

Method:
  - population counts summed inside circular buffers
  - 25, 50, 70 km
  - haversine distance from site to grid-cell centers
  - data class = historical_population_reconstruction
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(r"D:\temporal_environmental_data")
INPUT_DIR = (
    ROOT / "01_raw" / "HYDE_population" /
    "ISIMIP3a_histsoc_5arcmin_clean"
)

EXTRACTED_OUT = ROOT / "03_extracted"
TABLE_OUT = ROOT / "04_tables"
FIG_OUT = ROOT / "05_figures"
LOG_OUT = ROOT / "07_logs"

for p in (EXTRACTED_OUT, TABLE_OUT, FIG_OUT, LOG_OUT):
    p.mkdir(parents=True, exist_ok=True)

SITES = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}

BUFFERS_KM = [25, 50, 70]
START_YEAR = 1900
END_YEAR = 2018


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(a))


def choose_coord(ds, names):
    for name in names:
        if name in ds.coords:
            return name
    return None


def choose_population_variable(ds, time_name, lat_name, lon_name):
    exact = ["population", "pop", "total_population"]

    for name in exact:
        if name in ds.data_vars:
            return name

    candidates = []
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        if (
            time_name in dims
            and lat_name in dims
            and lon_name in dims
        ):
            score = 0
            lname = name.lower()
            if "population" in lname:
                score += 20
            if lname.startswith("pop"):
                score += 10
            candidates.append((score, name))

    candidates.sort(reverse=True)

    if not candidates:
        raise RuntimeError(
            f"Cannot identify population variable. "
            f"Variables are: {list(ds.data_vars)}"
        )

    return candidates[0][1]


print("=" * 78)
print("ISIMIP3a HISTORICAL POPULATION — CLEAN 5-ARCMIN EXTRACTION")
print("=" * 78)
print(f"Input folder: {INPUT_DIR}")
print(f"Period: {START_YEAR}-{END_YEAR}")
print(f"Buffers: {BUFFERS_KM} km")

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"\nClean input folder not found:\n{INPUT_DIR}\n\n"
        "Run 04b_population_download_ISIMIP3a_5arcmin_ONLY.py first."
    )

files_early = sorted(INPUT_DIR.glob(
    "*population_histsoc_5arcmin_annual_1850_1900*.nc"
))
files_late = sorted(INPUT_DIR.glob(
    "*population_histsoc_5arcmin_annual_1901_2021*.nc"
))

if len(files_early) != 1 or len(files_late) != 1:
    raise RuntimeError(
        "\nExpected exactly these two clean inputs:\n"
        "  one 1850_1900 file\n"
        "  one 1901_2021 file\n"
        f"Found early={len(files_early)}, late={len(files_late)}"
    )

nc_files = [files_early[0], files_late[0]]

print("\nInput files:")
for p in nc_files:
    print(" ", p.name)

opened = []
metadata = []

for nc in nc_files:
    ds = xr.open_dataset(nc, decode_cf=True)

    lat_name = choose_coord(ds, ["lat", "latitude"])
    lon_name = choose_coord(ds, ["lon", "longitude"])
    time_name = choose_coord(ds, ["time"])

    if not all([lat_name, lon_name, time_name]):
        ds.close()
        raise RuntimeError(
            f"Could not identify lat/lon/time in {nc.name}."
        )

    pop_var = choose_population_variable(
        ds, time_name, lat_name, lon_name
    )

    try:
        years = np.asarray(ds[time_name].dt.year.values, dtype=int)
    except Exception as exc:
        ds.close()
        raise RuntimeError(
            f"Could not decode annual time coordinate in {nc.name}."
        ) from exc

    attrs = ds[pop_var].attrs

    print(f"\n{nc.name}")
    print("  dimensions:", dict(ds.sizes))
    print("  variables:", list(ds.data_vars))
    print(f"  selected population variable: {pop_var}")
    print(f"  units: {attrs.get('units', '')!r}")
    print(f"  long_name: {attrs.get('long_name', '')!r}")
    print(f"  years: {years.min()} to {years.max()}")
    print(
        f"  latitude: {float(ds[lat_name].min()):.4f} "
        f"to {float(ds[lat_name].max()):.4f}"
    )
    print(
        f"  longitude: {float(ds[lon_name].min()):.4f} "
        f"to {float(ds[lon_name].max()):.4f}"
    )

    metadata.append({
        "file": nc.name,
        "population_variable": pop_var,
        "units": attrs.get("units", ""),
        "long_name": attrs.get("long_name", ""),
        "first_year": int(years.min()),
        "last_year": int(years.max()),
        "n_time": int(ds.sizes[time_name]),
        "n_lat": int(ds.sizes[lat_name]),
        "n_lon": int(ds.sizes[lon_name]),
        "data_class": "historical_population_reconstruction",
        "dataset": "ISIMIP3a histsoc",
        "resolution": "5 arcmin",
    })

    opened.append(
        (ds, lat_name, lon_name, time_name, pop_var, years)
    )

# Structural compatibility
base = opened[0]
for other in opened[1:]:
    if other[1:5] != base[1:5]:
        for x in opened:
            x[0].close()
        raise RuntimeError(
            "Early and late files use incompatible coordinate/variable names."
        )

lat_name, lon_name, time_name, pop_var = base[1:5]

pieces = []

for ds, latn, lonn, timen, pvar, years in opened:
    keep = np.where(
        (years >= START_YEAR) & (years <= END_YEAR)
    )[0]

    if keep.size:
        da = ds[pvar].isel({timen: keep}).load()
        da = da.assign_coords(
            extraction_year=(timen, years[keep])
        )
        pieces.append(da)

for item in opened:
    item[0].close()

pop = xr.concat(pieces, dim=time_name)

years = np.asarray(
    pop["extraction_year"].values,
    dtype=int
)

# Remove duplicate years and sort.
_, idx = np.unique(years, return_index=True)
idx = np.sort(idx)
pop = pop.isel({time_name: idx})

years = np.asarray(
    pop["extraction_year"].values,
    dtype=int
)

order = np.argsort(years)
pop = pop.isel({time_name: order})
years = np.asarray(
    pop["extraction_year"].values,
    dtype=int
)

expected = np.arange(START_YEAR, END_YEAR + 1)

if not np.array_equal(years, expected):
    missing = sorted(set(expected) - set(years))
    extra = sorted(set(years) - set(expected))
    raise RuntimeError(
        f"Annual coverage is not exactly {START_YEAR}-{END_YEAR}.\n"
        f"Missing: {missing}\nExtra: {extra}"
    )

print("\nAnnual series assembled successfully.")
print(f"  first year: {years[0]}")
print(f"  last year:  {years[-1]}")
print(f"  annual steps: {len(years)}")

# Normalize dimension order.
pop = pop.transpose(time_name, lat_name, lon_name)

lat = np.asarray(pop[lat_name].values, dtype=float)
lon = np.asarray(pop[lon_name].values, dtype=float)
lon_distance = ((lon + 180.0) % 360.0) - 180.0

lon2d, lat2d = np.meshgrid(lon_distance, lat)

values = np.asarray(pop.values, dtype=float)

rows = []
prov_rows = []

print("\nSpatial extraction:")

for site, (site_lat, site_lon) in SITES.items():
    distance = haversine_km(
        site_lat, site_lon, lat2d, lon2d
    )

    flat = int(np.nanargmin(distance))
    ii, jj = np.unravel_index(flat, distance.shape)

    nearest_lat = float(lat[ii])
    nearest_lon = float(lon_distance[jj])
    nearest_distance = float(distance[ii, jj])

    print(
        f"\n{site}"
        f"\n  nearest grid center: "
        f"({nearest_lat:.4f}, {nearest_lon:.4f})"
        f"\n  distance: {nearest_distance:.2f} km"
    )

    for radius in BUFFERS_KM:
        mask = distance <= radius
        n_cells = int(mask.sum())

        if n_cells == 0:
            raise RuntimeError(
                f"No grid centers inside {radius} km for {site}."
            )

        selected = values[:, mask]

        finite_count = np.sum(
            np.isfinite(selected), axis=1
        )

        totals = np.nansum(
            selected, axis=1
        ).astype(float)

        totals[finite_count == 0] = np.nan

        print(
            f"  {radius} km: "
            f"{n_cells} grid centers selected"
        )

        for k, year in enumerate(years):
            rows.append({
                "site": site,
                "latitude": site_lat,
                "longitude": site_lon,
                "year": int(year),
                "buffer_km": radius,
                "population_sum": (
                    float(totals[k])
                    if np.isfinite(totals[k])
                    else np.nan
                ),
                "n_grid_cells": n_cells,
                "n_nonmissing_cells": int(finite_count[k]),
                "dataset": "ISIMIP3a histsoc",
                "variable": pop_var,
                "data_class":
                    "historical_population_reconstruction",
                "spatial_resolution": "5 arcmin",
                "buffer_method":
                    "grid_cell_center_haversine_inclusion",
            })

        prov_rows.append({
            "site": site,
            "site_latitude": site_lat,
            "site_longitude": site_lon,
            "buffer_km": radius,
            "n_grid_cells": n_cells,
            "nearest_grid_latitude": nearest_lat,
            "nearest_grid_longitude": nearest_lon,
            "nearest_grid_distance_km":
                nearest_distance,
            "dataset": "ISIMIP3a histsoc",
            "data_class":
                "historical_population_reconstruction",
            "spatial_resolution": "5 arcmin",
            "buffer_method":
                "grid_cell_center_haversine_inclusion",
        })

df = pd.DataFrame(rows)

# Main long table
long_csv = (
    EXTRACTED_OUT /
    "population_ISIMIP3a_5arcmin_buffers_1900_2018.csv"
)
df.to_csv(long_csv, index=False)

# Main wide annual table
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

wide_csv = (
    EXTRACTED_OUT /
    "population_ISIMIP3a_5arcmin_annual_1900_2018.csv"
)
wide.to_csv(wide_csv, index=False)

# Metadata / provenance
meta_csv = (
    TABLE_OUT /
    "population_ISIMIP3a_5arcmin_file_metadata.csv"
)
pd.DataFrame(metadata).to_csv(meta_csv, index=False)

prov_csv = (
    TABLE_OUT /
    "population_ISIMIP3a_5arcmin_buffer_provenance.csv"
)
pd.DataFrame(prov_rows).to_csv(prov_csv, index=False)

# Change summary
summary = []

for (site, radius), g in df.groupby(
    ["site", "buffer_km"]
):
    g = g.sort_values("year")

    p1900 = float(
        g.loc[g.year == 1900, "population_sum"].iloc[0]
    )
    p2018 = float(
        g.loc[g.year == 2018, "population_sum"].iloc[0]
    )

    early = float(
        g.loc[
            g.year.between(1900, 1919),
            "population_sum"
        ].mean()
    )

    recent = float(
        g.loc[
            g.year.between(1999, 2018),
            "population_sum"
        ].mean()
    )

    absolute = p2018 - p1900

    percent = (
        absolute / p1900 * 100.0
        if p1900 != 0 else np.nan
    )

    fold = (
        p2018 / p1900
        if p1900 != 0 else np.nan
    )

    period_pct = (
        (recent - early) / early * 100.0
        if early != 0 else np.nan
    )

    period_fold = (
        recent / early
        if early != 0 else np.nan
    )

    summary.append({
        "site": site,
        "buffer_km": radius,
        "population_1900": p1900,
        "population_2018": p2018,
        "absolute_change_1900_2018": absolute,
        "percent_change_1900_2018": percent,
        "fold_change_1900_2018": fold,
        "mean_population_1900_1919": early,
        "mean_population_1999_2018": recent,
        "percent_change_20yr_period_means":
            period_pct,
        "fold_change_20yr_period_means":
            period_fold,
    })

summary_df = pd.DataFrame(summary)

summary_csv = (
    TABLE_OUT /
    "population_ISIMIP3a_5arcmin_change_summary.csv"
)
summary_df.to_csv(summary_csv, index=False)

# Descriptive OLS
trends = []

for (site, radius), g in df.groupby(
    ["site", "buffer_km"]
):
    g = g.dropna(
        subset=["population_sum"]
    ).sort_values("year")

    x = g.year.to_numpy(dtype=float)
    y = g.population_sum.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0 else np.nan
    )

    trends.append({
        "site": site,
        "buffer_km": radius,
        "n_years": len(g),
        "ols_slope_people_per_year": slope,
        "ols_slope_people_per_decade": slope * 10.0,
        "r_squared": r2,
        "note":
            "Descriptive OLS; formal Mann-Kendall and "
            "Sen slope can be added later.",
    })

trend_df = pd.DataFrame(trends)

trend_csv = (
    TABLE_OUT /
    "population_ISIMIP3a_5arcmin_trend_summary.csv"
)
trend_df.to_csv(trend_csv, index=False)

# Figure
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 10),
        sharex=True,
        constrained_layout=True,
    )

    for ax, site in zip(axes, SITES.keys()):
        sdf = df[df.site == site]

        for radius in BUFFERS_KM:
            g = sdf[
                sdf.buffer_km == radius
            ].sort_values("year")

            ax.plot(
                g.year,
                g.population_sum,
                label=f"{radius} km",
                linewidth=1.5,
            )

        ax.set_title(site)
        ax.set_ylabel("Population")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Year")

    fig.suptitle(
        "Historical population around Philippine marine sites\n"
        "ISIMIP3a histsoc, 5 arcmin"
    )

    png = (
        FIG_OUT /
        "Fig_population_ISIMIP3a_5arcmin_1900_2018.png"
    )
    pdf = (
        FIG_OUT /
        "Fig_population_ISIMIP3a_5arcmin_1900_2018.pdf"
    )

    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

except Exception as exc:
    print(f"\nWARNING: figure creation failed: {exc}")

print("\n" + "=" * 78)
print("EXTRACTION COMPLETE")
print("=" * 78)
print(f"Annual wide table:\n  {wide_csv}")
print(f"Annual long table:\n  {long_csv}")
print(f"Change summary:\n  {summary_csv}")
print(f"Trend summary:\n  {trend_csv}")
print(f"Provenance:\n  {prov_csv}")
print(f"Metadata:\n  {meta_csv}")
print()
print("Expected long rows: 1071")
print(f"Actual long rows:   {len(df)}")
print()
print("CHANGE SUMMARY")
print(summary_df.to_string(index=False))
