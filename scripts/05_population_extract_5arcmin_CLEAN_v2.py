"""
05_population_extract_5arcmin_CLEAN_v2.py
=========================================

Extract 1900-2018 historical population around Jolo, Masinloc, and Palapag
from the CLEAN ISIMIP3a histsoc 5-arcmin Philippines subset.

PATCH v2:
- Opens NetCDF with decode_times=False because ISIMIP population files use
  units such as "years since 1850-1-1 12:00:00", which xarray/cftime do not
  decode as standard CF time.
- Reconstructs integer calendar years from the time units + numeric offsets.
- Performs explicit checks that the reconstructed years are sensible.

Input:
D:\temporal_environmental_data\01_raw\HYDE_population\
    ISIMIP3a_histsoc_5arcmin_clean\

Study period:
1900-2018

Buffers:
25, 50, 70 km

Data class:
historical_population_reconstruction
"""

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(r"D:\temporal_environmental_data")

INPUT_DIR = (
    ROOT / "01_raw" / "HYDE_population"
    / "ISIMIP3a_histsoc_5arcmin_clean"
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
        if name in ds.variables and ds[name].ndim == 1:
            return name
    return None


def choose_population_variable(ds, time_name, lat_name, lon_name):
    preferred = [
        "population",
        "pop",
        "total_population",
        "population_count",
    ]

    for name in preferred:
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
            lname = name.lower()
            score = 0

            if "population" in lname:
                score += 20
            if lname == "pop" or lname.startswith("pop"):
                score += 10

            candidates.append((score, name))

    if not candidates:
        raise RuntimeError(
            "Could not identify population variable.\n"
            f"Available data variables: {list(ds.data_vars)}"
        )

    candidates.sort(reverse=True)

    return candidates[0][1]


def decode_isimip_years(time_da):
    """
    Convert non-standard ISIMIP time values such as:
        units = 'years since 1850-1-1 12:00:00'
        values = 0, 1, 2, ...

    to integer years 1850, 1851, 1852, ...

    Also supports a direct integer-year time axis.
    """

    vals = np.asarray(time_da.values, dtype=float)
    units = str(time_da.attrs.get("units", "")).strip()

    # Case 1: "years since YYYY-..."
    match = re.search(
        r"years?\s+since\s+([0-9]{4})",
        units,
        flags=re.IGNORECASE,
    )

    if match:
        reference_year = int(match.group(1))

        rounded = np.rint(vals)

        if not np.allclose(vals, rounded, atol=1e-6, rtol=0):
            raise RuntimeError(
                "Time offsets are not whole years.\n"
                f"units={units!r}\n"
                f"first values={vals[:10]}"
            )

        years = reference_year + rounded.astype(int)

        return years, reference_year, units

    # Case 2: time coordinate already consists of years like 1850, 1851...
    if (
        np.all(np.isfinite(vals))
        and np.nanmin(vals) >= 1500
        and np.nanmax(vals) <= 2500
    ):
        rounded = np.rint(vals)

        if np.allclose(vals, rounded, atol=1e-6, rtol=0):
            years = rounded.astype(int)

            return years, None, units

    raise RuntimeError(
        "Unsupported time encoding.\n"
        f"units={units!r}\n"
        f"first values={vals[:10]}"
    )


print("=" * 78)
print("ISIMIP3a HISTORICAL POPULATION — CLEAN 5-ARCMIN EXTRACTION v2")
print("=" * 78)

print(f"Input folder: {INPUT_DIR}")
print(f"Study period: {START_YEAR}-{END_YEAR}")
print(f"Buffers: {BUFFERS_KM} km")
print()

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Clean input directory does not exist:\n{INPUT_DIR}"
    )

early_files = sorted(
    INPUT_DIR.glob(
        "*population_histsoc_5arcmin_annual_1850_1900*.nc"
    )
)

late_files = sorted(
    INPUT_DIR.glob(
        "*population_histsoc_5arcmin_annual_1901_2021*.nc"
    )
)

if len(early_files) != 1 or len(late_files) != 1:
    raise RuntimeError(
        "Expected exactly two clean ISIMIP3a files.\n"
        f"1850-1900 matches: {len(early_files)}\n"
        f"1901-2021 matches: {len(late_files)}"
    )

nc_files = [early_files[0], late_files[0]]

print("Input files:")
for nc in nc_files:
    print(f"  {nc.name}")

opened = []
metadata_rows = []

print()
print("STEP 1 — Inspecting NetCDF structure and reconstructing years")
print("-" * 78)

for nc in nc_files:

    # Important patch:
    ds = xr.open_dataset(
        nc,
        decode_times=False,
    )

    lat_name = choose_coord(
        ds,
        ["lat", "latitude"],
    )

    lon_name = choose_coord(
        ds,
        ["lon", "longitude"],
    )

    time_name = choose_coord(
        ds,
        ["time"],
    )

    if not all([lat_name, lon_name, time_name]):
        ds.close()

        raise RuntimeError(
            f"Could not identify lat/lon/time in {nc.name}.\n"
            f"coords={list(ds.coords)}"
        )

    pop_var = choose_population_variable(
        ds,
        time_name,
        lat_name,
        lon_name,
    )

    years, reference_year, time_units = decode_isimip_years(
        ds[time_name]
    )

    attrs = ds[pop_var].attrs

    print()
    print(nc.name)
    print(f"  dimensions: {dict(ds.sizes)}")
    print(f"  data variables: {list(ds.data_vars)}")
    print(f"  selected population variable: {pop_var}")
    print(f"  units: {attrs.get('units', '')!r}")
    print(f"  long_name: {attrs.get('long_name', '')!r}")
    print(f"  time units: {time_units!r}")
    print(
        "  raw time first/last: "
        f"{float(ds[time_name].values[0])} / "
        f"{float(ds[time_name].values[-1])}"
    )
    print(
        f"  reconstructed years: "
        f"{int(years.min())} to {int(years.max())}"
    )
    print(
        f"  latitude: "
        f"{float(ds[lat_name].min()):.4f} to "
        f"{float(ds[lat_name].max()):.4f}"
    )
    print(
        f"  longitude: "
        f"{float(ds[lon_name].min()):.4f} to "
        f"{float(ds[lon_name].max()):.4f}"
    )

    metadata_rows.append({
        "file": nc.name,
        "population_variable": pop_var,
        "population_units": attrs.get("units", ""),
        "population_long_name": attrs.get("long_name", ""),
        "time_units_original": time_units,
        "time_reference_year": reference_year,
        "first_reconstructed_year": int(years.min()),
        "last_reconstructed_year": int(years.max()),
        "n_time": int(ds.sizes[time_name]),
        "n_lat": int(ds.sizes[lat_name]),
        "n_lon": int(ds.sizes[lon_name]),
        "dataset": "ISIMIP3a histsoc",
        "data_class": "historical_population_reconstruction",
        "resolution": "5 arcmin",
    })

    opened.append(
        (
            ds,
            nc,
            lat_name,
            lon_name,
            time_name,
            pop_var,
            years,
        )
    )


# Check structure compatibility
base_lat = opened[0][2]
base_lon = opened[0][3]
base_time = opened[0][4]
base_var = opened[0][5]

for item in opened[1:]:

    if (
        item[2] != base_lat
        or item[3] != base_lon
        or item[4] != base_time
        or item[5] != base_var
    ):
        for x in opened:
            x[0].close()

        raise RuntimeError(
            "The two files do not use compatible "
            "coordinate/variable names."
        )


lat_name = base_lat
lon_name = base_lon
time_name = base_time
pop_var = base_var


print()
print("STEP 2 — Combining 1900-2018")
print("-" * 78)

pieces = []

for (
    ds,
    nc,
    latn,
    lonn,
    timen,
    pvar,
    years,
) in opened:

    keep = np.where(
        (years >= START_YEAR)
        & (years <= END_YEAR)
    )[0]

    if keep.size:

        da = ds[pvar].isel(
            {timen: keep}
        ).load()

        da = da.assign_coords(
            extraction_year=(
                timen,
                years[keep],
            )
        )

        pieces.append(da)


for item in opened:
    item[0].close()


if not pieces:
    raise RuntimeError(
        "No data found for 1900-2018."
    )


pop = xr.concat(
    pieces,
    dim=time_name,
)

years = np.asarray(
    pop["extraction_year"].values,
    dtype=int,
)


# Remove duplicate years if present
_, first_idx = np.unique(
    years,
    return_index=True,
)

first_idx = np.sort(first_idx)

pop = pop.isel(
    {time_name: first_idx}
)

years = np.asarray(
    pop["extraction_year"].values,
    dtype=int,
)


# Sort chronologically
order = np.argsort(years)

pop = pop.isel(
    {time_name: order}
)

years = np.asarray(
    pop["extraction_year"].values,
    dtype=int,
)


expected_years = np.arange(
    START_YEAR,
    END_YEAR + 1,
)


if not np.array_equal(
    years,
    expected_years,
):
    missing = sorted(
        set(expected_years)
        - set(years)
    )

    extra = sorted(
        set(years)
        - set(expected_years)
    )

    raise RuntimeError(
        "Annual time reconstruction failed.\n"
        f"Missing years: {missing}\n"
        f"Extra years: {extra}\n"
        f"First 10 recovered: {years[:10]}\n"
        f"Last 10 recovered: {years[-10:]}"
    )


print(
    f"Annual coverage confirmed: "
    f"{years[0]}-{years[-1]}"
)

print(
    f"Annual steps: {len(years)} "
    f"(expected {len(expected_years)})"
)


print()
print("STEP 3 — Preparing 5-arcmin spatial grid")
print("-" * 78)


pop = pop.transpose(
    time_name,
    lat_name,
    lon_name,
)


lat = np.asarray(
    pop[lat_name].values,
    dtype=float,
)

lon = np.asarray(
    pop[lon_name].values,
    dtype=float,
)


# Longitude for distance calculations
lon_distance = (
    (lon + 180.0) % 360.0
) - 180.0


lon2d, lat2d = np.meshgrid(
    lon_distance,
    lat,
)


values = np.asarray(
    pop.values,
    dtype=float,
)


print(
    f"Grid: {len(lat)} latitude x "
    f"{len(lon)} longitude cells"
)


rows = []
prov_rows = []


print()
print("STEP 4 — Population within 25/50/70-km buffers")
print("-" * 78)


for site, (
    site_lat,
    site_lon,
) in SITES.items():

    distance = haversine_km(
        site_lat,
        site_lon,
        lat2d,
        lon2d,
    )

    nearest_flat = int(
        np.nanargmin(distance)
    )

    ii, jj = np.unravel_index(
        nearest_flat,
        distance.shape,
    )

    nearest_lat = float(lat[ii])
    nearest_lon = float(
        lon_distance[jj]
    )

    nearest_distance = float(
        distance[ii, jj]
    )


    print()
    print(site)

    print(
        f"  nearest grid center: "
        f"({nearest_lat:.4f}, "
        f"{nearest_lon:.4f})"
    )

    print(
        f"  site-to-center distance: "
        f"{nearest_distance:.2f} km"
    )


    for radius in BUFFERS_KM:

        mask = distance <= radius

        n_cells = int(
            mask.sum()
        )

        if n_cells == 0:
            raise RuntimeError(
                f"No grid centers within "
                f"{radius} km of {site}."
            )


        selected = values[:, mask]

        n_nonmissing = np.sum(
            np.isfinite(selected),
            axis=1,
        )


        totals = np.nansum(
            selected,
            axis=1,
        ).astype(float)


        # If all cells are NaN, total should be NaN, not zero.
        totals[
            n_nonmissing == 0
        ] = np.nan


        print(
            f"  {radius:>2} km: "
            f"{n_cells} grid centers "
            f"selected"
        )


        for k, year in enumerate(
            years
        ):

            rows.append({
                "site": site,
                "latitude": site_lat,
                "longitude": site_lon,
                "year": int(year),
                "buffer_km": int(radius),
                "population_sum": (
                    float(totals[k])
                    if np.isfinite(
                        totals[k]
                    )
                    else np.nan
                ),
                "n_grid_cells":
                    n_cells,
                "n_nonmissing_cells":
                    int(n_nonmissing[k]),
                "dataset":
                    "ISIMIP3a histsoc",
                "variable":
                    pop_var,
                "data_class":
                    "historical_population_reconstruction",
                "spatial_resolution":
                    "5 arcmin (~9 km)",
                "buffer_method":
                    "grid_cell_center_haversine_inclusion",
            })


        prov_rows.append({
            "site": site,
            "site_latitude":
                site_lat,
            "site_longitude":
                site_lon,
            "buffer_km":
                radius,
            "n_grid_cells":
                n_cells,
            "nearest_grid_latitude":
                nearest_lat,
            "nearest_grid_longitude":
                nearest_lon,
            "nearest_grid_distance_km":
                nearest_distance,
            "dataset":
                "ISIMIP3a histsoc",
            "data_class":
                "historical_population_reconstruction",
            "spatial_resolution":
                "5 arcmin",
            "buffer_method":
                "grid_cell_center_haversine_inclusion",
        })


df = pd.DataFrame(rows)


print()
print("STEP 5 — Writing annual datasets")
print("-" * 78)


long_csv = (
    EXTRACTED_OUT
    / "population_ISIMIP3a_5arcmin_buffers_1900_2018.csv"
)

df.to_csv(
    long_csv,
    index=False,
)


wide = df.pivot(
    index=[
        "site",
        "latitude",
        "longitude",
        "year",
    ],
    columns="buffer_km",
    values="population_sum",
).reset_index()


wide.columns.name = None

wide = wide.rename(
    columns={
        25: "population_25km",
        50: "population_50km",
        70: "population_70km",
    }
)


wide_csv = (
    EXTRACTED_OUT
    / "population_ISIMIP3a_5arcmin_annual_1900_2018.csv"
)

wide.to_csv(
    wide_csv,
    index=False,
)


meta_csv = (
    TABLE_OUT
    / "population_ISIMIP3a_5arcmin_file_metadata.csv"
)

pd.DataFrame(
    metadata_rows
).to_csv(
    meta_csv,
    index=False,
)


prov_csv = (
    TABLE_OUT
    / "population_ISIMIP3a_5arcmin_buffer_provenance.csv"
)

pd.DataFrame(
    prov_rows
).to_csv(
    prov_csv,
    index=False,
)


print(f"Long table: {long_csv}")
print(f"Wide table: {wide_csv}")


print()
print("STEP 6 — Historical change summary")
print("-" * 78)


summary_rows = []


for (
    site,
    radius,
), g in df.groupby(
    ["site", "buffer_km"]
):

    g = g.sort_values(
        "year"
    )


    p1900 = float(
        g.loc[
            g["year"] == 1900,
            "population_sum",
        ].iloc[0]
    )

    p2018 = float(
        g.loc[
            g["year"] == 2018,
            "population_sum",
        ].iloc[0]
    )


    early = float(
        g.loc[
            g["year"].between(
                1900,
                1919,
            ),
            "population_sum",
        ].mean()
    )


    modern = float(
        g.loc[
            g["year"].between(
                1999,
                2018,
            ),
            "population_sum",
        ].mean()
    )


    absolute = (
        p2018
        - p1900
    )


    if (
        np.isfinite(p1900)
        and p1900 != 0
    ):
        pct_change = (
            absolute
            / p1900
            * 100.0
        )

        fold_change = (
            p2018
            / p1900
        )

    else:
        pct_change = np.nan
        fold_change = np.nan


    if (
        np.isfinite(early)
        and early != 0
    ):
        period_pct = (
            (modern - early)
            / early
            * 100.0
        )

        period_fold = (
            modern
            / early
        )

    else:
        period_pct = np.nan
        period_fold = np.nan


    summary_rows.append({
        "site":
            site,
        "buffer_km":
            int(radius),
        "population_1900":
            p1900,
        "population_2018":
            p2018,
        "absolute_change_1900_2018":
            absolute,
        "percent_change_1900_2018":
            pct_change,
        "fold_change_1900_2018":
            fold_change,
        "mean_population_1900_1919":
            early,
        "mean_population_1999_2018":
            modern,
        "percent_change_20yr_period_means":
            period_pct,
        "fold_change_20yr_period_means":
            period_fold,
        "data_class":
            "historical_population_reconstruction",
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_csv = (
    TABLE_OUT
    / "population_ISIMIP3a_5arcmin_change_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False,
)


print(
    summary_df.to_string(
        index=False
    )
)


print()
print("STEP 7 — Descriptive OLS trends")
print("-" * 78)


trend_rows = []


for (
    site,
    radius,
), g in df.groupby(
    ["site", "buffer_km"]
):

    g = g.dropna(
        subset=[
            "population_sum"
        ]
    ).sort_values(
        "year"
    )


    x = g[
        "year"
    ].to_numpy(
        dtype=float
    )

    y = g[
        "population_sum"
    ].to_numpy(
        dtype=float
    )


    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )


    pred = (
        intercept
        + slope * x
    )


    ss_res = np.sum(
        (y - pred) ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )


    r2 = (
        1.0
        - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )


    trend_rows.append({
        "site":
            site,
        "buffer_km":
            int(radius),
        "n_years":
            int(len(g)),
        "ols_slope_people_per_year":
            float(slope),
        "ols_slope_people_per_decade":
            float(slope * 10.0),
        "r_squared":
            float(r2),
        "note":
            (
                "Descriptive OLS only; "
                "formal Mann-Kendall and "
                "Sen slope can be added later."
            ),
    })


trend_df = pd.DataFrame(
    trend_rows
)


trend_csv = (
    TABLE_OUT
    / "population_ISIMIP3a_5arcmin_trend_summary.csv"
)

trend_df.to_csv(
    trend_csv,
    index=False,
)


print(
    trend_df.to_string(
        index=False
    )
)


print()
print("STEP 8 — Figure")
print("-" * 78)


try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 10),
        sharex=True,
        constrained_layout=True,
    )


    for ax, site in zip(
        axes,
        SITES.keys(),
    ):

        sdf = df[
            df["site"] == site
        ]


        for radius in BUFFERS_KM:

            g = sdf[
                sdf["buffer_km"]
                == radius
            ].sort_values(
                "year"
            )


            ax.plot(
                g["year"],
                g["population_sum"],
                label=f"{radius} km",
                linewidth=1.5,
            )


        ax.set_title(
            site
        )

        ax.set_ylabel(
            "Population (people)"
        )

        ax.legend(
            frameon=False
        )

        ax.grid(
            alpha=0.2
        )


    axes[-1].set_xlabel(
        "Year"
    )


    fig.suptitle(
        "Historical population around Philippine marine sites\n"
        "ISIMIP3a histsoc, 5 arcmin, 1900-2018"
    )


    png = (
        FIG_OUT
        / "Fig_population_ISIMIP3a_5arcmin_1900_2018.png"
    )

    pdf = (
        FIG_OUT
        / "Fig_population_ISIMIP3a_5arcmin_1900_2018.pdf"
    )


    fig.savefig(
        png,
        dpi=300,
    )

    fig.savefig(
        pdf,
    )

    plt.close(
        fig
    )


    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


except Exception as exc:

    print(
        "WARNING: figure creation failed:"
    )

    print(exc)


log_file = (
    LOG_OUT
    / "population_ISIMIP3a_5arcmin_extraction_log.txt"
)


with open(
    log_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "ISIMIP3a historical population extraction\n"
    )

    f.write(
        "=" * 60
        + "\n"
    )

    f.write(
        "Data class: historical_population_reconstruction\n"
    )

    f.write(
        "Resolution: 5 arcmin\n"
    )

    f.write(
        "Period: 1900-2018\n"
    )

    f.write(
        "Buffers: 25, 50, 70 km\n"
    )

    f.write(
        "Buffer method: grid-cell-center haversine inclusion\n"
    )

    f.write(
        "Population measure: sum of gridded population values\n"
    )

    f.write(
        "Time handling: raw numeric annual offsets decoded manually "
        "from 'years since YYYY...' because xarray does not support "
        "years as a CF datetime unit.\n"
    )


print()
print("=" * 78)
print("EXTRACTION COMPLETE")
print("=" * 78)

print(
    f"Expected long rows: "
    f"{3 * 3 * 119}"
)

print(
    f"Actual long rows: "
    f"{len(df)}"
)

print()
print("Main annual dataset:")
print(f"  {wide_csv}")

print()
print("Change summary:")
print(f"  {summary_csv}")

print()
print("Trend summary:")
print(f"  {trend_csv}")

print()
print("Provenance:")
print(f"  {prov_csv}")

print()
print("Metadata:")
print(f"  {meta_csv}")

print()
print("Log:")
print(f"  {log_file}")
