"""
SCRIPT 16
ISIMIP3a HISTORICAL LAND-USE RECONSTRUCTION
1900–2018

Sites:
    Jolo
    Masinloc
    Palapag

Buffers:
    25 km
    50 km
    70 km

Purpose
-------
Extract historical land-use reconstruction around the three study sites
for comparison with historical SST, population, and modern/static
coastal habitat variables.

Data classification
-------------------
ISIMIP3a histsoc land use:
    historical_landuse_reconstruction

IMPORTANT:
These are reconstructed/harmonized historical land-use fractions.
They are NOT direct annual satellite observations.

Spatial method
--------------
The ISIMIP land-use grid is approximately 0.5 degrees.

For each site and buffer:
    1. Construct circular buffer in local UTM.
    2. Construct polygons for ISIMIP grid cells.
    3. Calculate exact cell-buffer intersections.
    4. Multiply land-use fraction by intersecting cell area.
    5. Sum category area within buffer.

The 70-km buffer is intended as the primary analysis.
25-km and 50-km results are sensitivity analyses because the
0.5-degree source grid is relatively coarse.

Project:
D:\\temporal_environmental_data
"""

from pathlib import Path
import zipfile
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from shapely.geometry import box, Point
from pyproj import CRS


warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# PROJECT DIRECTORIES
# =====================================================================

ROOT = Path(r"D:\temporal_environmental_data")

RAW = ROOT / "01_raw" / "LUH2_landuse"
EXTRACTED = ROOT / "03_extracted"
TABLES = ROOT / "04_tables"
LOGS = ROOT / "07_logs"

for directory in [RAW, EXTRACTED, TABLES, LOGS]:
    directory.mkdir(parents=True, exist_ok=True)


# =====================================================================
# STUDY SITES
# =====================================================================

SITES = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}

BUFFERS_KM = [25, 50, 70]

YEAR0 = 1900
YEAR1 = 2018


# =====================================================================
# REGIONAL DOWNLOAD BOUNDING BOX
# =====================================================================

WEST = 118.0
EAST = 126.5
SOUTH = 5.0
NORTH = 17.0


# =====================================================================
# ISIMIP REPOSITORY PATHS
# =====================================================================

REPO_BASE = "ISIMIP3a/InputData/socioeconomic/landuse/histsoc"

REPO_PATHS = [

    f"{REPO_BASE}/landuse-totals_histsoc_annual_1850_1900.nc",

    f"{REPO_BASE}/landuse-totals_histsoc_annual_1901_2021.nc",

    f"{REPO_BASE}/landuse-urbanareas_histsoc_annual_1850_1900.nc",

    f"{REPO_BASE}/landuse-urbanareas_histsoc_annual_1901_2021.nc",
]


# =====================================================================
# OUTPUT FILES
# =====================================================================

OUT_ANNUAL = (
    EXTRACTED /
    "landuse_ISIMIP3a_buffers_1900_2018.csv"
)

OUT_CHANGE = (
    TABLES /
    "landuse_ISIMIP3a_change_summary.csv"
)

OUT_PROVENANCE = (
    TABLES /
    "landuse_ISIMIP3a_buffer_provenance.csv"
)

OUT_METADATA = (
    TABLES /
    "landuse_ISIMIP3a_file_metadata.csv"
)

OUT_QC = (
    TABLES /
    "landuse_ISIMIP3a_QC.csv"
)

OUT_LOG = (
    LOGS /
    "16_ISIMIP3a_historical_landuse.log"
)


# =====================================================================
# LOGGING
# =====================================================================

log_lines = []


def log(message=""):

    print(message)

    log_lines.append(str(message))


# =====================================================================
# FILE DISCOVERY
# =====================================================================

def find_nc(pattern):

    """
    Recursively locate NetCDF files.

    IMPORTANT:
    ISIMIP bounding-box downloads append a geographic suffix.

    Example:

    landuse-totals_histsoc_annual_1901_2021_
    lon118.0to126.5lat5.0to17.0.nc

    Therefore wildcard matching is required.
    """

    return sorted(RAW.rglob(pattern))


def have_all_files():

    patterns = [

        "landuse-totals_histsoc_annual_1850_1900*.nc",

        "landuse-totals_histsoc_annual_1901_2021*.nc",

        "landuse-urbanareas_histsoc_annual_1850_1900*.nc",

        "landuse-urbanareas_histsoc_annual_1901_2021*.nc",
    ]

    return all(
        len(find_nc(pattern)) > 0
        for pattern in patterns
    )


# =====================================================================
# OPTIONAL DOWNLOAD
# =====================================================================

def download_cutout():

    log("Required NetCDF files not found locally.")

    log(
        "Requesting Philippines regional cutout "
        "from ISIMIP Files API ..."
    )

    try:

        from isimip_client.client import ISIMIPClient

    except ImportError:

        raise SystemExit(
            "\nisimip-client is not installed.\n\n"
            "Run:\n\n"
            "pip install -U isimip-client\n"
        )

    client = ISIMIPClient()

    response = client.select_bbox(
        REPO_PATHS,
        west=WEST,
        east=EAST,
        south=SOUTH,
        north=NORTH,
        poll=4
    )

    if (
        not isinstance(response, dict)
        or not response.get("file_url")
    ):

        raise RuntimeError(
            "Unexpected ISIMIP Files API response:\n"
            f"{response}"
        )

    before = set(RAW.iterdir())

    client.download(
        response["file_url"],
        path=str(RAW)
    )

    after = set(RAW.iterdir())

    new_items = list(after - before)

    log("Download completed.")

    for item in new_items:

        log(f"  {item}")

    # -------------------------------------------------------------
    # Extract downloaded ZIP archives
    # -------------------------------------------------------------

    for archive in RAW.glob("*.zip"):

        target = RAW / archive.stem

        if not target.exists():

            log(
                f"Extracting: {archive.name}"
            )

            with zipfile.ZipFile(
                archive,
                "r"
            ) as z:

                z.extractall(target)


# =====================================================================
# CHECK LOCAL DATA
# =====================================================================

log("=" * 78)

log(
    "SCRIPT 16 — ISIMIP3a HISTORICAL LAND USE"
)

log("=" * 78)


if have_all_files():

    log(
        "Required ISIMIP regional NetCDF files "
        "already exist locally."
    )

    log(
        "No download required."
    )

else:

    download_cutout()


if not have_all_files():

    raise FileNotFoundError(
        "\nThe four required NetCDF files "
        "were not found after download/extraction.\n\n"
        f"Inspect:\n{RAW}"
    )


# =====================================================================
# REPORT ACTUAL FILES FOUND
# =====================================================================

log("")

log("ISIMIP FILES FOUND")

log("-" * 78)


required_patterns = [

    "landuse-totals_histsoc_annual_1850_1900*.nc",

    "landuse-totals_histsoc_annual_1901_2021*.nc",

    "landuse-urbanareas_histsoc_annual_1850_1900*.nc",

    "landuse-urbanareas_histsoc_annual_1901_2021*.nc",
]


for pattern in required_patterns:

    matches = find_nc(pattern)

    for match in matches:

        log(
            f"  {match}"
        )


# =====================================================================
# OPEN DATASET PAIRS
# =====================================================================

def open_pair(prefix):

    """
    Open the 1850–1900 and 1901–2021 files.

    Wildcards allow use of ISIMIP bounding-box filenames.
    """

    first_matches = find_nc(
        f"{prefix}_1850_1900*.nc"
    )

    second_matches = find_nc(
        f"{prefix}_1901_2021*.nc"
    )

    if not first_matches:

        raise FileNotFoundError(
            f"No 1850–1900 file found for {prefix}"
        )

    if not second_matches:

        raise FileNotFoundError(
            f"No 1901–2021 file found for {prefix}"
        )

    f1 = first_matches[0]

    f2 = second_matches[0]

    log("")

    log(
        f"Opening: {f1.name}"
    )

    log(
        f"Opening: {f2.name}"
    )

    try:

        a = xr.open_dataset(f1)

        b = xr.open_dataset(f2)

    except Exception:

        log(
            "Standard CF time decoding failed; "
            "retrying with decode_times=False."
        )

        a = xr.open_dataset(
            f1,
            decode_times=False
        )

        b = xr.open_dataset(
            f2,
            decode_times=False
        )

    return a, b, f1, f2


tot_a, tot_b, tot_f1, tot_f2 = open_pair(
    "landuse-totals_histsoc_annual"
)

urb_a, urb_b, urb_f1, urb_f2 = open_pair(
    "landuse-urbanareas_histsoc_annual"
)

# =====================================================================
# TIME HANDLING — ROBUST ISIMIP ANNUAL TIME DECODING
# =====================================================================

def assign_year(ds, expected_start, expected_end):
    """
    Assign explicit calendar years to ISIMIP annual files.

    These ISIMIP files use a non-standard time encoding that may fail
    normal CF decoding. The filenames define the exact annual coverage,
    so use that coverage explicitly and verify that the number of time
    steps agrees with it.

    Example:
        1850-1900 -> 51 annual steps
        1901-2021 -> 121 annual steps
    """

    if "time" not in ds.dims:
        raise ValueError(
            "Dataset has no 'time' dimension."
        )

    expected_years = np.arange(
        expected_start,
        expected_end + 1,
        dtype=int
    )

    n_time = ds.sizes["time"]

    if n_time != len(expected_years):
        raise ValueError(
            "\nUnexpected number of annual time steps.\n"
            f"Expected {len(expected_years)} for "
            f"{expected_start}-{expected_end}, "
            f"but dataset contains {n_time}.\n"
            "Stopping rather than guessing the year mapping."
        )

    ds = ds.assign_coords(
        year=("time", expected_years)
    )

    ds = ds.swap_dims(
        {"time": "year"}
    )

    # Drop the original encoded time coordinate.
    # Calendar year is now the authoritative time coordinate.
    if "time" in ds.coords:
        ds = ds.drop_vars("time")

    return ds


# Explicit annual coverage from the ISIMIP filenames
tot_a = assign_year(
    tot_a,
    1850,
    1900
)

tot_b = assign_year(
    tot_b,
    1901,
    2021
)

urb_a = assign_year(
    urb_a,
    1850,
    1900
)

urb_b = assign_year(
    urb_b,
    1901,
    2021
)


# =====================================================================
# CONCATENATE HISTORICAL PERIODS
# =====================================================================

tot = xr.concat(
    [tot_a, tot_b],
    dim="year"
)

urb = xr.concat(
    [urb_a, urb_b],
    dim="year"
)


# Check full source chronology BEFORE subsetting
expected_full_years = np.arange(
    1850,
    2022,
    dtype=int
)

actual_tot_years = np.asarray(
    tot["year"].values,
    dtype=int
)

actual_urb_years = np.asarray(
    urb["year"].values,
    dtype=int
)


if not np.array_equal(
    actual_tot_years,
    expected_full_years
):
    raise ValueError(
        "\nTOTALS chronology is incorrect after concatenation.\n"
        f"Expected 1850-2021 ({len(expected_full_years)} years).\n"
        f"Found {len(actual_tot_years)} years from "
        f"{actual_tot_years.min()} to "
        f"{actual_tot_years.max()}."
    )


if not np.array_equal(
    actual_urb_years,
    expected_full_years
):
    raise ValueError(
        "\nURBAN chronology is incorrect after concatenation.\n"
        f"Expected 1850-2021 ({len(expected_full_years)} years).\n"
        f"Found {len(actual_urb_years)} years from "
        f"{actual_urb_years.min()} to "
        f"{actual_urb_years.max()}."
    )


# =====================================================================
# SELECT STUDY PERIOD
# =====================================================================

tot = tot.sel(
    year=slice(
        YEAR0,
        YEAR1
    )
)

urb = urb.sel(
    year=slice(
        YEAR0,
        YEAR1
    )
)


# =====================================================================
# FINAL STUDY-PERIOD TIME QC
# =====================================================================

expected_study_years = np.arange(
    YEAR0,
    YEAR1 + 1,
    dtype=int
)

tot_years = np.asarray(
    tot["year"].values,
    dtype=int
)

urb_years = np.asarray(
    urb["year"].values,
    dtype=int
)


if not np.array_equal(
    tot_years,
    expected_study_years
):
    raise ValueError(
        "\nTOTALS study-period chronology failed QC.\n"
        f"Expected {YEAR0}-{YEAR1} "
        f"({len(expected_study_years)} years).\n"
        f"Found: {tot_years}"
    )


if not np.array_equal(
    urb_years,
    expected_study_years
):
    raise ValueError(
        "\nURBAN study-period chronology failed QC.\n"
        f"Expected {YEAR0}-{YEAR1} "
        f"({len(expected_study_years)} years).\n"
        f"Found: {urb_years}"
    )


log("")
log("TIME COORDINATE QC")
log("-" * 78)
log(
    f"TOTALS: {tot_years[0]}-{tot_years[-1]} "
    f"({len(tot_years)} annual steps)"
)
log(
    f"URBAN:  {urb_years[0]}-{urb_years[-1]} "
    f"({len(urb_years)} annual steps)"
)
log("Expected study period: 1900-2018 = 119 annual steps")

# =====================================================================
# CONCATENATE HISTORICAL PERIODS
# =====================================================================

tot = xr.concat(
    [tot_a, tot_b],
    dim="year"
)

urb = xr.concat(
    [urb_a, urb_b],
    dim="year"
)


tot = tot.sel(
    year=slice(
        YEAR0,
        YEAR1
    )
)

urb = urb.sel(
    year=slice(
        YEAR0,
        YEAR1
    )
)


# =====================================================================
# IDENTIFY COORDINATE NAMES
# =====================================================================

def coord_name(ds, candidates):

    for candidate in candidates:

        if (
            candidate in ds.coords
            or candidate in ds.dims
        ):

            return candidate

    raise ValueError(
        f"Could not identify coordinate "
        f"among {candidates}"
    )


LAT = coord_name(
    tot,
    ["lat", "latitude"]
)

LON = coord_name(
    tot,
    ["lon", "longitude"]
)


# =====================================================================
# REGIONAL SUBSET — SAFETY CHECK
# =====================================================================

def regional(ds):

    lat = ds[LAT].values

    lon = ds[LON].values

    if lat[0] < lat[-1]:

        lat_slice = slice(
            SOUTH,
            NORTH
        )

    else:

        lat_slice = slice(
            NORTH,
            SOUTH
        )

    if lon[0] < lon[-1]:

        lon_slice = slice(
            WEST,
            EAST
        )

    else:

        lon_slice = slice(
            EAST,
            WEST
        )

    return ds.sel(
        {
            LAT: lat_slice,
            LON: lon_slice
        }
    )


tot = regional(tot)

urb = regional(urb)


# =====================================================================
# DATASET INFORMATION
# =====================================================================

log("")

log(
    "TOTALS DATASET DIMENSIONS"
)

log("-" * 78)

log(
    str(dict(tot.sizes))
)


log("")

log(
    "URBAN DATASET DIMENSIONS"
)

log("-" * 78)

log(
    str(dict(urb.sizes))
)


log("")

log(
    "TOTALS VARIABLES"
)

log("-" * 78)


for variable in tot.data_vars:

    units = tot[variable].attrs.get(
        "units",
        ""
    )

    long_name = tot[variable].attrs.get(
        "long_name",
        ""
    )

    log(
        f"  {variable}"
        f" | units={units}"
        f" | long_name={long_name}"
    )


log("")

log(
    "URBAN VARIABLES"
)

log("-" * 78)


for variable in urb.data_vars:

    units = urb[variable].attrs.get(
        "units",
        ""
    )

    long_name = urb[variable].attrs.get(
        "long_name",
        ""
    )

    log(
        f"  {variable}"
        f" | units={units}"
        f" | long_name={long_name}"
    )


# =====================================================================
# VARIABLE IDENTIFICATION
# =====================================================================

def find_var(
    ds,
    exact,
    contains=None
):

    if exact in ds.data_vars:

        return exact

    lower_lookup = {
        variable.lower(): variable
        for variable
        in ds.data_vars
    }

    if exact.lower() in lower_lookup:

        return lower_lookup[
            exact.lower()
        ]

    if contains:

        for variable in ds.data_vars:

            lower = variable.lower()

            if all(
                token.lower() in lower
                for token
                in contains
            ):

                return variable

    return None


varmap = {

    "cropland_total":

        find_var(
            tot,
            "cropland_total",
            ["cropland", "total"]
        ),

    "cropland_rainfed":

        find_var(
            tot,
            "cropland_rainfed",
            ["cropland", "rain"]
        ),

    "cropland_irrigated":

        find_var(
            tot,
            "cropland_irrigated",
            ["cropland", "irrig"]
        ),

    "pastures":

        find_var(
            tot,
            "pastures",
            ["pasture"]
        ),

    "forests_natural":

        find_var(
            tot,
            "forests_and_natural_vegetation",
            ["forest", "natural"]
        ),

    "urban":

        find_var(
            urb,
            "urbanareas",
            ["urban"]
        )
}


missing_variables = [

    key

    for key, value

    in varmap.items()

    if value is None
]


if missing_variables:

    raise ValueError(
        "\nCould not identify required "
        f"variables:\n{missing_variables}\n\n"
        f"Totals variables:\n"
        f"{list(tot.data_vars)}\n\n"
        f"Urban variables:\n"
        f"{list(urb.data_vars)}"
    )


log("")

log(
    "SELECTED LAND-USE VARIABLES"
)

log("-" * 78)


for key, value in varmap.items():

    log(
        f"  {key} <- {value}"
    )


# =====================================================================
# GRID CELL POLYGONS
# =====================================================================

lats = np.asarray(
    tot[LAT].values,
    dtype=float
)

lons = np.asarray(
    tot[LON].values,
    dtype=float
)


def edges_from_centers(values):

    values = np.asarray(
        values,
        dtype=float
    )

    order = np.argsort(values)

    sorted_values = values[order]

    mids = (
        sorted_values[:-1]
        +
        sorted_values[1:]
    ) / 2

    edges = np.empty(
        len(sorted_values) + 1
    )

    edges[1:-1] = mids

    edges[0] = (
        sorted_values[0]
        -
        (
            mids[0]
            -
            sorted_values[0]
        )
    )

    edges[-1] = (
        sorted_values[-1]
        +
        (
            sorted_values[-1]
            -
            mids[-1]
        )
    )

    return sorted_values, edges


lat_sorted, lat_edges = (
    edges_from_centers(lats)
)

lon_sorted, lon_edges = (
    edges_from_centers(lons)
)


cells = []


for i, latitude in enumerate(
    lat_sorted
):

    for j, longitude in enumerate(
        lon_sorted
    ):

        geometry = box(

            lon_edges[j],

            lat_edges[i],

            lon_edges[j + 1],

            lat_edges[i + 1]
        )

        cells.append(
            {
                "lat": latitude,
                "lon": longitude,
                "geometry": geometry
            }
        )


grid = gpd.GeoDataFrame(
    cells,
    crs="EPSG:4326"
)


# =====================================================================
# LOCAL UTM CRS
# =====================================================================

def utm_crs(
    latitude,
    longitude
):

    zone = int(
        (
            longitude + 180
        ) // 6
    ) + 1

    if latitude >= 0:

        epsg = 32600 + zone

    else:

        epsg = 32700 + zone

    return CRS.from_epsg(
        epsg
    )


# =====================================================================
# EXACT GRID-CELL / BUFFER OVERLAP
# =====================================================================

log("")

log(
    "CALCULATING EXACT BUFFER–GRID INTERSECTIONS"
)

log("-" * 78)


weight_rows = []


for site, (
    latitude,
    longitude
) in SITES.items():

    log(
        f"Processing spatial weights: {site}"
    )

    local_crs = utm_crs(
        latitude,
        longitude
    )

    projected_grid = grid.to_crs(
        local_crs
    )

    point = gpd.GeoSeries(
        [
            Point(
                longitude,
                latitude
            )
        ],
        crs="EPSG:4326"
    ).to_crs(
        local_crs
    ).iloc[0]

    for radius in BUFFERS_KM:

        buffer_geometry = point.buffer(
            radius * 1000
        )

        buffer_area_km2 = (
            buffer_geometry.area
            /
            1e6
        )

        candidates = projected_grid[
            projected_grid.intersects(
                buffer_geometry
            )
        ].copy()

        for idx, row in candidates.iterrows():

            intersection = (
                row.geometry.intersection(
                    buffer_geometry
                )
            )

            overlap_km2 = (
                intersection.area
                /
                1e6
            )

            if overlap_km2 > 0:

                weight_rows.append(
                    {
                        "site": site,

                        "buffer_km": radius,

                        "lat":
                            grid.loc[
                                idx,
                                "lat"
                            ],

                        "lon":
                            grid.loc[
                                idx,
                                "lon"
                            ],

                        "overlap_km2":
                            overlap_km2,

                        "buffer_area_km2":
                            buffer_area_km2
                    }
                )


weights = pd.DataFrame(
    weight_rows
)


log("")

log(
    f"Generated {len(weights):,} "
    "grid-cell/buffer overlap weights."
)


# =====================================================================
# EXTRACT DATA VALUE
# =====================================================================

def get_value(
    ds,
    variable,
    year,
    latitude,
    longitude
):

    value = (
        ds[variable]
        .sel(
            year=year
        )
        .sel(
            {
                LAT: latitude,
                LON: longitude
            },
            method="nearest"
        )
    )

    return float(
        np.asarray(
            value.values
        ).squeeze()
    )


# =====================================================================
# ANNUAL EXTRACTION
# =====================================================================

records = []


log("")

log(
    "EXTRACTING HISTORICAL LAND USE"
)

log("-" * 78)


for site in SITES:

    log(
        f"Extracting: {site}"
    )

    for radius in BUFFERS_KM:

        w = weights[
            (
                weights["site"]
                ==
                site
            )
            &
            (
                weights["buffer_km"]
                ==
                radius
            )
        ].copy()

        if w.empty:

            raise RuntimeError(
                f"No grid cells intersect "
                f"{site} {radius}-km buffer."
            )

        buffer_area = float(
            w[
                "buffer_area_km2"
            ].iloc[0]
        )

        for year in range(
            YEAR0,
            YEAR1 + 1
        ):

            record = {

                "site": site,

                "year": year,

                "buffer_km": radius,

                "buffer_area_km2":
                    buffer_area,

                "n_intersecting_grid_cells":
                    len(w)
            }


            # =========================================================
            # LAND-USE CATEGORIES
            # =========================================================

            for (
                output_name,
                variable
            ) in varmap.items():

                if output_name == "urban":

                    ds = urb

                else:

                    ds = tot


                values = []

                overlap_areas = []


                for row in w.itertuples():

                    value = get_value(

                        ds,

                        variable,

                        year,

                        row.lat,

                        row.lon
                    )


                    if np.isfinite(value):

                        values.append(
                            value
                        )

                        overlap_areas.append(
                            row.overlap_km2
                        )


                values = np.asarray(
                    values,
                    dtype=float
                )

                overlap_areas = np.asarray(
                    overlap_areas,
                    dtype=float
                )


                if len(values):

                    category_area = float(
                        np.sum(
                            values
                            *
                            overlap_areas
                        )
                    )

                else:

                    category_area = np.nan


                record[
                    f"{output_name}_area_km2"
                ] = category_area


                if np.isfinite(
                    category_area
                ):

                    fraction = (
                        category_area
                        /
                        buffer_area
                    )

                else:

                    fraction = np.nan


                record[
                    f"{output_name}_fraction_of_buffer"
                ] = fraction


                record[
                    f"{output_name}_percent_of_buffer"
                ] = (
                    fraction * 100
                    if np.isfinite(fraction)
                    else np.nan
                )


            # =========================================================
            # AGRICULTURAL LAND
            # =========================================================

            record[
                "agricultural_area_km2"
            ] = (

                record[
                    "cropland_total_area_km2"
                ]

                +

                record[
                    "pastures_area_km2"
                ]
            )


            record[
                "agricultural_percent_of_buffer"
            ] = (

                100

                *

                record[
                    "agricultural_area_km2"
                ]

                /

                buffer_area
            )


            records.append(
                record
            )


# =====================================================================
# BUILD ANNUAL TABLE
# =====================================================================

annual = pd.DataFrame(
    records
)


annual = annual.sort_values(
    [
        "site",
        "buffer_km",
        "year"
    ]
).reset_index(
    drop=True
)


annual[
    "data_class"
] = (
    "historical_landuse_reconstruction"
)


annual[
    "temporal_status"
] = (
    "historical_time_varying"
)


annual[
    "spatial_assumption"
] = (

    "land-use fraction assumed spatially uniform "
    "within each 0.5-degree source grid cell; "
    "fraction multiplied by exact cell-buffer "
    "intersection area"
)


annual.to_csv(
    OUT_ANNUAL,
    index=False
)


# =====================================================================
# HISTORICAL CHANGE SUMMARY
# =====================================================================

metrics = [

    "cropland_total_percent_of_buffer",

    "pastures_percent_of_buffer",

    "forests_natural_percent_of_buffer",

    "urban_percent_of_buffer",

    "agricultural_percent_of_buffer"
]


summary_records = []


for site in SITES:

    for radius in BUFFERS_KM:

        subset = annual[
            (
                annual["site"]
                ==
                site
            )
            &
            (
                annual["buffer_km"]
                ==
                radius
            )
        ]


        record = {

            "site": site,

            "buffer_km": radius
        }


        for metric in metrics:

            value_1900 = float(

                subset.loc[
                    subset["year"] == 1900,
                    metric
                ].iloc[0]
            )


            value_2018 = float(

                subset.loc[
                    subset["year"] == 2018,
                    metric
                ].iloc[0]
            )


            early_mean = float(

                subset.loc[
                    subset["year"].between(
                        1900,
                        1919
                    ),
                    metric
                ].mean()
            )


            late_mean = float(

                subset.loc[
                    subset["year"].between(
                        1999,
                        2018
                    ),
                    metric
                ].mean()
            )


            record[
                f"{metric}_1900"
            ] = value_1900


            record[
                f"{metric}_2018"
            ] = value_2018


            record[
                f"{metric}_change_1900_2018_pp"
            ] = (

                value_2018
                -
                value_1900
            )


            record[
                f"{metric}_mean_1900_1919"
            ] = early_mean


            record[
                f"{metric}_mean_1999_2018"
            ] = late_mean


            record[
                f"{metric}_change_20yr_means_pp"
            ] = (

                late_mean
                -
                early_mean
            )


        summary_records.append(
            record
        )


change_summary = pd.DataFrame(
    summary_records
)


change_summary.to_csv(
    OUT_CHANGE,
    index=False
)


# =====================================================================
# PROVENANCE TABLE
# =====================================================================

provenance_records = []


for site in SITES:

    for radius in BUFFERS_KM:

        subset = weights[
            (
                weights["site"]
                ==
                site
            )
            &
            (
                weights["buffer_km"]
                ==
                radius
            )
        ]


        provenance_records.append(
            {

                "site":
                    site,

                "buffer_km":
                    radius,

                "dataset":
                    "ISIMIP3a historical gridded land use (histsoc)",

                "dataset_version_totals":
                    "20241114 / v1.3",

                "dataset_version_urbanareas":
                    "20220607 / v1.3",

                "period_used":
                    "1900-2018",

                "nominal_resolution":
                    "0.5 degree",

                "data_class":
                    "historical_landuse_reconstruction",

                "analysis_role":
                    "historical terrestrial human-pressure context",

                "n_intersecting_grid_cells":
                    len(subset),

                "buffer_area_km2":
                    subset[
                        "buffer_area_km2"
                    ].iloc[0],

                "processing":
                    (
                        "exact UTM circular buffer-grid "
                        "intersection; source land-use "
                        "fraction multiplied by "
                        "cell-buffer overlap area"
                    ),

                "source_doi":
                    "10.48364/ISIMIP.571261.3",

                "warning":
                    (
                        "Reconstructed/harmonized historical "
                        "land use; not direct annual satellite "
                        "or census land-cover observations."
                    )
            }
        )


provenance = pd.DataFrame(
    provenance_records
)


provenance.to_csv(
    OUT_PROVENANCE,
    index=False
)


# =====================================================================
# FILE METADATA
# =====================================================================

metadata_records = []


for path, dataset in [

    (tot_f1, tot_a),

    (tot_f2, tot_b),

    (urb_f1, urb_a),

    (urb_f2, urb_b)

]:

    metadata_records.append(
        {

            "file":
                path.name,

            "path":
                str(path),

            "dimensions":
                str(
                    dict(
                        dataset.sizes
                    )
                ),

            "variables":
                ";".join(
                    dataset.data_vars
                ),

            "source":
                dataset.attrs.get(
                    "source",
                    ""
                ),

            "institution":
                dataset.attrs.get(
                    "institution",
                    ""
                ),

            "title":
                dataset.attrs.get(
                    "title",
                    ""
                )
        }
    )


metadata = pd.DataFrame(
    metadata_records
)


metadata.to_csv(
    OUT_METADATA,
    index=False
)


# =====================================================================
# QUALITY CONTROL
# =====================================================================

qc_records = []


area_columns = [

    column

    for column in annual.columns

    if column.endswith(
        "_area_km2"
    )
]


for site in SITES:

    for radius in BUFFERS_KM:

        subset = annual[
            (
                annual["site"]
                ==
                site
            )
            &
            (
                annual["buffer_km"]
                ==
                radius
            )
        ]


        years = sorted(
            subset[
                "year"
            ].unique()
        )


        negative_values = int(

            (
                subset[
                    area_columns
                ]
                <
                0
            )
            .sum()
            .sum()
        )


        qc_records.append(
            {

                "site":
                    site,

                "buffer_km":
                    radius,

                "rows":
                    len(subset),

                "n_years":
                    len(years),

                "first_year":
                    min(years),

                "last_year":
                    max(years),

                "complete_1900_2018":
                    (
                        len(years) == 119
                        and
                        min(years) == 1900
                        and
                        max(years) == 2018
                    ),

                "duplicate_years":
                    int(
                        subset[
                            "year"
                        ].duplicated().sum()
                    ),

                "negative_category_values":
                    negative_values
            }
        )


qc = pd.DataFrame(
    qc_records
)


qc.to_csv(
    OUT_QC,
    index=False
)


# =====================================================================
# NUMERIC SANITY CHECKS
# =====================================================================

log("")

log(
    "NUMERIC SANITY CHECKS"
)

log("-" * 78)


percentage_columns = [

    column

    for column in annual.columns

    if column.endswith(
        "_percent_of_buffer"
    )
]


for column in percentage_columns:

    minimum = annual[
        column
    ].min()

    maximum = annual[
        column
    ].max()

    log(
        f"{column}: "
        f"{minimum:.4f} to "
        f"{maximum:.4f}%"
    )


# =====================================================================
# 70-KM CHANGE PREVIEW
# =====================================================================

preview = change_summary[
    change_summary[
        "buffer_km"
    ]
    ==
    70
].copy()


preview_columns = [

    "site",

    "cropland_total_percent_of_buffer_change_1900_2018_pp",

    "pastures_percent_of_buffer_change_1900_2018_pp",

    "forests_natural_percent_of_buffer_change_1900_2018_pp",

    "urban_percent_of_buffer_change_1900_2018_pp"
]


log("")

log(
    "70-KM HISTORICAL CHANGE PREVIEW"
)

log(
    "Change from 1900 to 2018 "
    "(percentage points)"
)

log("-" * 78)


log(
    preview[
        preview_columns
    ].to_string(
        index=False
    )
)


# =====================================================================
# FINAL REPORT
# =====================================================================

log("")

log("=" * 78)

log(
    "SCRIPT 16 COMPLETE"
)

log("=" * 78)


log("")

log(
    "Annual historical land use:"
)

log(
    str(
        OUT_ANNUAL
    )
)


log("")

log(
    "Historical change summary:"
)

log(
    str(
        OUT_CHANGE
    )
)


log("")

log(
    "Buffer provenance:"
)

log(
    str(
        OUT_PROVENANCE
    )
)


log("")

log(
    "File metadata:"
)

log(
    str(
        OUT_METADATA
    )
)


log("")

log(
    "Quality control:"
)

log(
    str(
        OUT_QC
    )
)


log("")

log(
    "EXPECTED DIMENSIONS"
)

log("-" * 78)


log(
    "Annual table:"
)

log(
    "3 sites × 3 buffers × 119 years "
    "= 1,071 rows"
)


log(
    f"Actual annual rows: "
    f"{len(annual):,}"
)


log("")

log(
    "EVIDENCE CLASS"
)

log("-" * 78)


log(
    "ISIMIP3a land use = "
    "historical land-use reconstruction"
)


log(
    "NOT direct annual satellite observation."
)


log("")

log(
    "SPATIAL INTERPRETATION"
)

log("-" * 78)


log(
    "70 km = primary historical "
    "land-use analysis."
)


log(
    "25 km and 50 km = "
    "spatial sensitivity analyses."
)


log(
    "The source grid is approximately "
    "0.5 degrees, so small-buffer results "
    "must not be interpreted as fine-scale "
    "local land-cover measurements."
)


# =====================================================================
# WRITE LOG
# =====================================================================

OUT_LOG.write_text(
    "\n".join(
        log_lines
    ),
    encoding="utf-8"
)


log("")

log(
    f"Log saved: {OUT_LOG}"
)