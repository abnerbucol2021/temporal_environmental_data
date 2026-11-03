# =============================================================================
# SCRIPT 19 — HISTORICAL ENVIRONMENTAL-PRESSURE INTEGRATION
# =============================================================================
#
# Purpose
# -------
# Integrate the completed annual historical datasets for:
#
#   1. ERSSTv5 sea-surface temperature
#   2. ISIMIP3a historical population reconstruction
#   3. ISIMIP3a historical land-use reconstruction
#
# for Jolo, Masinloc, and Palapag over 1900–2018.
#
# IMPORTANT EVIDENCE SEPARATION
# -----------------------------
# ERSSTv5:
#   observation-based SST reconstruction
#
# ISIMIP3a population:
#   historical population reconstruction
#
# ISIMIP3a land use:
#   historical land-use reconstruction
#
# ETOPO bathymetry and Allen Coral Atlas benthic habitat are NOT included here.
# They are static/modern spatial covariates and must remain separate from the
# historical annual time series.
#
# STANDARDIZATION
# ---------------
# For each site and variable:
#
#       Z(t) = [X(t) - mean(1900–1919)] / SD(1900–1919)
#
# Thus z = 0 represents the site's early-20th-century baseline mean.
#
# Positive/negative z values MUST be interpreted according to the variable:
#
#   SST          positive = warmer
#   Population   positive = more people
#   Cropland     positive = more cropland
#   Urban        positive = more urban land
#   Forest       negative = less forest/natural vegetation
#
# No arbitrary composite pressure index is calculated.
#
# =============================================================================

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\temporal_environmental_data")

EXTRACTED = ROOT / "03_extracted"
TABLES = ROOT / "04_tables"
FIGURES = ROOT / "05_figures"
LOGS = ROOT / "07_logs"

for folder in [EXTRACTED, TABLES, FIGURES, LOGS]:
    folder.mkdir(parents=True, exist_ok=True)


SST_FILE = EXTRACTED / "ersstv5_annual_1900_2018.csv"

POP_FILE = (
    EXTRACTED /
    "population_ISIMIP3a_5arcmin_annual_1900_2018.csv"
)

LANDUSE_FILE = (
    EXTRACTED /
    "landuse_ISIMIP3a_buffers_1900_2018.csv"
)


START_YEAR = 1900
END_YEAR = 2018

BASELINE_START = 1900
BASELINE_END = 1919

PRIMARY_BUFFER_KM = 70

SITES = ["Jolo", "Masinloc", "Palapag"]


# =============================================================================
# OUTPUT FILES
# =============================================================================

OUT_MASTER = (
    EXTRACTED /
    "historical_environmental_pressure_master_1900_2018.csv"
)

OUT_STANDARDIZED = (
    EXTRACTED /
    "historical_environmental_pressure_standardized_1900_2018.csv"
)

OUT_BASELINE = (
    TABLES /
    "historical_environmental_pressure_baseline_1900_1919.csv"
)

OUT_CHANGE = (
    TABLES /
    "historical_environmental_pressure_change_summary.csv"
)

OUT_CORRELATION = (
    TABLES /
    "historical_environmental_pressure_correlations.csv"
)

OUT_QC = (
    TABLES /
    "historical_environmental_pressure_QC.csv"
)

OUT_PROVENANCE = (
    TABLES /
    "historical_environmental_pressure_provenance.csv"
)

OUT_FIG_Z = (
    FIGURES /
    "Fig_historical_environmental_pressure_standardized_1900_2018.png"
)

OUT_FIG_Z_PDF = (
    FIGURES /
    "Fig_historical_environmental_pressure_standardized_1900_2018.pdf"
)

OUT_FIG_SITE = (
    FIGURES /
    "Fig_historical_environmental_pressure_by_site_1900_2018.png"
)

OUT_FIG_SITE_PDF = (
    FIGURES /
    "Fig_historical_environmental_pressure_by_site_1900_2018.pdf"
)

OUT_LOG = (
    LOGS /
    "19_historical_environmental_pressure_integration.log"
)


# =============================================================================
# LOGGING
# =============================================================================

log_lines = []


def log(message=""):
    text = str(message)
    print(text)
    log_lines.append(text)


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


# =============================================================================
# HELPERS
# =============================================================================

def find_column(df, candidates, required=True):
    """
    Find a column using exact match first, then case-insensitive match.
    """

    columns = list(df.columns)

    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_lookup = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    if required:
        raise KeyError(
            "\nCould not find any of these columns:\n"
            f"{candidates}\n\n"
            "Available columns:\n"
            + "\n".join(map(str, columns))
        )

    return None


def standardize_site_baseline(df, value_col):
    """
    Standardize a variable independently within each site relative to
    1900–1919 mean and sample standard deviation.
    """

    output = []

    for site in SITES:

        sub = (
            df[df["site"] == site]
            .sort_values("year")
            .copy()
        )

        baseline = sub[
            sub["year"].between(
                BASELINE_START,
                BASELINE_END
            )
        ][value_col]

        baseline = pd.to_numeric(
            baseline,
            errors="coerce"
        ).dropna()

        if len(baseline) != 20:
            raise ValueError(
                f"{site} / {value_col}: "
                f"expected 20 baseline years, found {len(baseline)}."
            )

        mean = baseline.mean()
        sd = baseline.std(ddof=1)

        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"{site} / {value_col}: invalid baseline SD = {sd}"
            )

        sub[f"{value_col}_baseline_mean"] = mean
        sub[f"{value_col}_baseline_sd"] = sd

        sub[f"{value_col}_z"] = (
            sub[value_col] - mean
        ) / sd

        output.append(sub)

    return pd.concat(
        output,
        ignore_index=True
    )


def early_late_summary(df, variable):
    """
    Calculate:
      1900
      2018
      1900–1919 mean
      1999–2018 mean
      absolute change
      percent change where meaningful
    """

    records = []

    for site in SITES:

        sub = (
            df[df["site"] == site]
            .sort_values("year")
        )

        early = sub[
            sub["year"].between(1900, 1919)
        ][variable]

        late = sub[
            sub["year"].between(1999, 2018)
        ][variable]

        value_1900 = sub.loc[
            sub["year"] == 1900,
            variable
        ]

        value_2018 = sub.loc[
            sub["year"] == 2018,
            variable
        ]

        if len(value_1900) != 1 or len(value_2018) != 1:
            raise ValueError(
                f"{site}/{variable}: missing 1900 or 2018 value."
            )

        value_1900 = float(value_1900.iloc[0])
        value_2018 = float(value_2018.iloc[0])

        early_mean = float(early.mean())
        late_mean = float(late.mean())

        absolute_change = value_2018 - value_1900
        period_mean_change = late_mean - early_mean

        if np.isfinite(value_1900) and value_1900 != 0:
            percent_change = (
                absolute_change /
                abs(value_1900)
            ) * 100
        else:
            percent_change = np.nan

        records.append(
            {
                "site": site,
                "variable": variable,
                "value_1900": value_1900,
                "value_2018": value_2018,
                "absolute_change_1900_2018": absolute_change,
                "percent_change_1900_2018": percent_change,
                "mean_1900_1919": early_mean,
                "mean_1999_2018": late_mean,
                "change_late_vs_early": period_mean_change,
            }
        )

    return records


# =============================================================================
# START
# =============================================================================

section(
    "SCRIPT 19 — HISTORICAL ENVIRONMENTAL-PRESSURE INTEGRATION"
)


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

section("CHECKING INPUT FILES")

required_files = [
    SST_FILE,
    POP_FILE,
    LANDUSE_FILE,
]

for path in required_files:

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

    log(f"FOUND: {path}")


# =============================================================================
# LOAD DATA
# =============================================================================

section("LOADING HISTORICAL DATA")

sst_raw = pd.read_csv(SST_FILE)
pop_raw = pd.read_csv(POP_FILE)
land_raw = pd.read_csv(LANDUSE_FILE)

log(
    f"ERSSTv5:      "
    f"{len(sst_raw):,} rows × {len(sst_raw.columns)} columns"
)

log(
    f"Population:    "
    f"{len(pop_raw):,} rows × {len(pop_raw.columns)} columns"
)

log(
    f"Land use:      "
    f"{len(land_raw):,} rows × {len(land_raw.columns)} columns"
)


# =============================================================================
# IDENTIFY COLUMNS
# =============================================================================

section("IDENTIFYING INPUT COLUMNS")


# SST -------------------------------------------------------------------------

sst_site_col = find_column(
    sst_raw,
    ["site", "Site"]
)

sst_year_col = find_column(
    sst_raw,
    ["year", "Year"]
)

sst_value_col = find_column(
    sst_raw,
    [
        "sst_mean_degC",
        "sst_degC",
        "sst",
        "annual_sst_degC",
        "mean_sst_degC",
    ]
)

log(f"SST site column:  {sst_site_col}")
log(f"SST year column:  {sst_year_col}")
log(f"SST value column: {sst_value_col}")


# POPULATION ------------------------------------------------------------------

pop_site_col = find_column(
    pop_raw,
    ["site", "Site"]
)

pop_year_col = find_column(
    pop_raw,
    ["year", "Year"]
)

pop_buffer_col = find_column(
    pop_raw,
    [
        "buffer_km",
        "buffer",
        "radius_km",
    ],
    required=False
)

pop_value_col = find_column(
    pop_raw,
    [
        "population_70km",
        "population",
        "total_population",
        "population_count",
    ]
)

log(f"Population site column:   {pop_site_col}")
log(f"Population year column:   {pop_year_col}")
log(f"Population buffer column: {pop_buffer_col}")
log(f"Population value column:  {pop_value_col}")


# LAND USE --------------------------------------------------------------------

land_site_col = find_column(
    land_raw,
    ["site", "Site"]
)

land_year_col = find_column(
    land_raw,
    ["year", "Year"]
)

land_buffer_col = find_column(
    land_raw,
    [
        "buffer_km",
        "buffer",
        "radius_km",
    ]
)

land_cropland_col = find_column(
    land_raw,
    [
        "cropland_total_percent_of_buffer",
        "cropland_percent_of_buffer",
    ]
)

land_pasture_col = find_column(
    land_raw,
    [
        "pastures_percent_of_buffer",
        "pasture_percent_of_buffer",
    ]
)

land_forest_col = find_column(
    land_raw,
    [
        "forests_natural_percent_of_buffer",
        "forest_natural_percent_of_buffer",
        "forest_percent_of_buffer",
    ]
)

land_urban_col = find_column(
    land_raw,
    [
        "urban_percent_of_buffer",
        "urbanareas_percent_of_buffer",
    ]
)

land_agriculture_col = find_column(
    land_raw,
    [
        "agricultural_percent_of_buffer",
        "agriculture_percent_of_buffer",
    ],
    required=False
)

log(f"Land-use site column:     {land_site_col}")
log(f"Land-use year column:     {land_year_col}")
log(f"Land-use buffer column:   {land_buffer_col}")
log(f"Cropland:                 {land_cropland_col}")
log(f"Pasture:                  {land_pasture_col}")
log(f"Forest/natural:           {land_forest_col}")
log(f"Urban:                    {land_urban_col}")
log(f"Agriculture:              {land_agriculture_col}")


# =============================================================================
# PREPARE SST
# =============================================================================

section("PREPARING ERSSTv5")

sst = sst_raw[
    [
        sst_site_col,
        sst_year_col,
        sst_value_col,
    ]
].copy()

sst.columns = [
    "site",
    "year",
    "sst_degC",
]

sst["year"] = pd.to_numeric(
    sst["year"],
    errors="coerce"
)

sst["sst_degC"] = pd.to_numeric(
    sst["sst_degC"],
    errors="coerce"
)

sst = sst[
    sst["year"].between(
        START_YEAR,
        END_YEAR
    )
].copy()

sst["year"] = sst["year"].astype(int)

sst = sst[
    sst["site"].isin(SITES)
].copy()

sst = (
    sst.groupby(
        ["site", "year"],
        as_index=False
    )["sst_degC"]
    .mean()
)

log(
    f"Prepared SST rows: {len(sst):,}"
)


# =============================================================================
# PREPARE POPULATION
# =============================================================================

section("PREPARING ISIMIP3a POPULATION")

pop = pop_raw.copy()

pop["year"] = pd.to_numeric(
    pop[pop_year_col],
    errors="coerce"
)

pop["site"] = pop[pop_site_col]

pop = pop[
    pop["year"].between(
        START_YEAR,
        END_YEAR
    )
].copy()

pop["year"] = pop["year"].astype(int)

pop = pop[
    pop["site"].isin(SITES)
].copy()


# If population file contains multiple buffers, select 70 km.
if pop_buffer_col is not None:

    pop[pop_buffer_col] = pd.to_numeric(
        pop[pop_buffer_col],
        errors="coerce"
    )

    pop = pop[
        np.isclose(
            pop[pop_buffer_col],
            PRIMARY_BUFFER_KM
        )
    ].copy()


pop["population_70km"] = pd.to_numeric(
    pop[pop_value_col],
    errors="coerce"
)

pop = pop[
    [
        "site",
        "year",
        "population_70km",
    ]
].copy()

pop = (
    pop.groupby(
        ["site", "year"],
        as_index=False
    )["population_70km"]
    .mean()
)

log(
    f"Prepared population rows: {len(pop):,}"
)


# =============================================================================
# PREPARE LAND USE
# =============================================================================

section("PREPARING ISIMIP3a LAND USE")

land = land_raw.copy()

land["site"] = land[land_site_col]

land["year"] = pd.to_numeric(
    land[land_year_col],
    errors="coerce"
)

land[land_buffer_col] = pd.to_numeric(
    land[land_buffer_col],
    errors="coerce"
)

land = land[
    land["year"].between(
        START_YEAR,
        END_YEAR
    )
].copy()

land["year"] = land["year"].astype(int)

land = land[
    land["site"].isin(SITES)
].copy()

land = land[
    np.isclose(
        land[land_buffer_col],
        PRIMARY_BUFFER_KM
    )
].copy()


land["cropland_percent_70km"] = pd.to_numeric(
    land[land_cropland_col],
    errors="coerce"
)

land["pasture_percent_70km"] = pd.to_numeric(
    land[land_pasture_col],
    errors="coerce"
)

land["forest_natural_percent_70km"] = pd.to_numeric(
    land[land_forest_col],
    errors="coerce"
)

land["urban_percent_70km"] = pd.to_numeric(
    land[land_urban_col],
    errors="coerce"
)


if land_agriculture_col is not None:

    land["agricultural_percent_70km"] = pd.to_numeric(
        land[land_agriculture_col],
        errors="coerce"
    )

else:

    land["agricultural_percent_70km"] = (
        land["cropland_percent_70km"] +
        land["pasture_percent_70km"]
    )


land = land[
    [
        "site",
        "year",
        "cropland_percent_70km",
        "pasture_percent_70km",
        "forest_natural_percent_70km",
        "urban_percent_70km",
        "agricultural_percent_70km",
    ]
].copy()

land = (
    land.groupby(
        ["site", "year"],
        as_index=False
    )
    .mean(numeric_only=True)
)

log(
    f"Prepared land-use rows: {len(land):,}"
)


# =============================================================================
# CHECK EXPECTED ROW COUNTS
# =============================================================================

section("PRE-MERGE QC")

expected_rows = (
    len(SITES) *
    (END_YEAR - START_YEAR + 1)
)

log(
    f"Expected site-year rows per historical dataset: "
    f"{expected_rows:,}"
)

for name, df in [
    ("ERSSTv5", sst),
    ("Population", pop),
    ("Land use", land),
]:

    log(
        f"{name}: {len(df):,} rows"
    )

    if len(df) != expected_rows:
        log(
            f"WARNING: {name} does not contain "
            f"the expected {expected_rows:,} site-year rows."
        )


# =============================================================================
# MERGE HISTORICAL DATA
# =============================================================================

section("MERGING HISTORICAL DATASETS")

master = (
    sst
    .merge(
        pop,
        on=["site", "year"],
        how="outer",
        validate="one_to_one",
    )
    .merge(
        land,
        on=["site", "year"],
        how="outer",
        validate="one_to_one",
    )
)

master = master[
    master["site"].isin(SITES)
].copy()

master = master[
    master["year"].between(
        START_YEAR,
        END_YEAR
    )
].copy()

master["site"] = pd.Categorical(
    master["site"],
    categories=SITES,
    ordered=True
)

master = master.sort_values(
    ["site", "year"]
).reset_index(drop=True)

master["site"] = master["site"].astype(str)


log(
    f"Master rows:    {len(master):,}"
)

log(
    f"Master columns: {len(master.columns):,}"
)


# =============================================================================
# MASTER QC
# =============================================================================

section("MASTER DATASET QC")

historical_variables = [
    "sst_degC",
    "population_70km",
    "cropland_percent_70km",
    "pasture_percent_70km",
    "forest_natural_percent_70km",
    "urban_percent_70km",
    "agricultural_percent_70km",
]

qc_records = []

for site in SITES:

    sub = master[
        master["site"] == site
    ]

    years_present = sorted(
        sub["year"].dropna().astype(int).unique()
    )

    missing_years = sorted(
        set(range(START_YEAR, END_YEAR + 1)) -
        set(years_present)
    )

    log("")
    log(site)
    log("-" * 40)

    log(
        f"Rows: {len(sub)}"
    )

    log(
        f"Year range: "
        f"{sub['year'].min()}–{sub['year'].max()}"
    )

    log(
        f"Missing years: {len(missing_years)}"
    )

    if missing_years:
        log(
            "  " +
            ", ".join(map(str, missing_years))
        )

    for variable in historical_variables:

        missing_n = int(
            sub[variable].isna().sum()
        )

        minimum = sub[variable].min()
        maximum = sub[variable].max()

        log(
            f"{variable}: "
            f"missing={missing_n}, "
            f"range={minimum:.6f} to {maximum:.6f}"
        )

        qc_records.append(
            {
                "site": site,
                "variable": variable,
                "n_rows": len(sub),
                "missing_values": missing_n,
                "minimum": minimum,
                "maximum": maximum,
                "missing_year_count": len(missing_years),
            }
        )


qc = pd.DataFrame(qc_records)


# =============================================================================
# NUMERIC SANITY CHECKS
# =============================================================================

section("NUMERIC SANITY CHECKS")


if (master["population_70km"] < 0).any():
    raise ValueError(
        "Negative population values detected."
    )

log(
    "Population: no negative values detected."
)


for variable in [
    "cropland_percent_70km",
    "pasture_percent_70km",
    "forest_natural_percent_70km",
    "urban_percent_70km",
]:

    if (
        (master[variable] < 0) |
        (master[variable] > 100)
    ).any():

        raise ValueError(
            f"{variable}: values outside 0–100 detected."
        )

log(
    "Land-use percentages: all values within 0–100."
)


sst_min = master["sst_degC"].min()
sst_max = master["sst_degC"].max()

log(
    f"SST range: {sst_min:.3f} to {sst_max:.3f} °C"
)


# =============================================================================
# STANDARDIZE RELATIVE TO 1900–1919
# =============================================================================

section(
    "STANDARDIZING RELATIVE TO 1900–1919 BASELINE"
)

standardization_variables = [
    "sst_degC",
    "population_70km",
    "cropland_percent_70km",
    "pasture_percent_70km",
    "forest_natural_percent_70km",
    "urban_percent_70km",
]

standardized = master.copy()

baseline_records = []


for variable in standardization_variables:

    z_frames = []

    for site in SITES:

        mask = (
            standardized["site"] == site
        )

        baseline_mask = (
            mask &
            standardized["year"].between(
                BASELINE_START,
                BASELINE_END
            )
        )

        baseline_values = pd.to_numeric(
            standardized.loc[
                baseline_mask,
                variable
            ],
            errors="coerce"
        ).dropna()

        if len(baseline_values) != 20:

            raise ValueError(
                f"{site}/{variable}: "
                f"expected 20 baseline values, "
                f"found {len(baseline_values)}."
            )

        mean = baseline_values.mean()
        sd = baseline_values.std(ddof=1)

        if not np.isfinite(sd) or sd <= 0:

            raise ValueError(
                f"{site}/{variable}: "
                f"baseline SD invalid: {sd}"
            )

        standardized.loc[
            mask,
            f"{variable}_z"
        ] = (
            standardized.loc[
                mask,
                variable
            ] - mean
        ) / sd

        baseline_records.append(
            {
                "site": site,
                "variable": variable,
                "baseline_start": BASELINE_START,
                "baseline_end": BASELINE_END,
                "n_baseline_years": len(baseline_values),
                "baseline_mean": mean,
                "baseline_sd": sd,
            }
        )


baseline_table = pd.DataFrame(
    baseline_records
)


# =============================================================================
# FOREST LOSS DIRECTIONAL TRANSFORMATION
# =============================================================================

section("CREATING DIRECTIONALLY CONSISTENT FOREST-LOSS METRIC")

# Raw forest z:
#
# positive = MORE forest than baseline
# negative = LESS forest than baseline
#
# For pressure-oriented comparison only, create:
#
# forest_loss_pressure_z = -forest_z
#
# Thus positive values mean increasing forest-loss pressure.
#
# IMPORTANT:
# This is only a sign reversal.
# It is NOT a composite index and does not create new information.

standardized["forest_loss_pressure_z"] = (
    -standardized[
        "forest_natural_percent_70km_z"
    ]
)

log(
    "Created forest_loss_pressure_z = "
    "-forest_natural_percent_70km_z"
)


# =============================================================================
# SAVE MASTER AND STANDARDIZED DATA
# =============================================================================

master.to_csv(
    OUT_MASTER,
    index=False
)

standardized.to_csv(
    OUT_STANDARDIZED,
    index=False
)

baseline_table.to_csv(
    OUT_BASELINE,
    index=False
)

qc.to_csv(
    OUT_QC,
    index=False
)


# =============================================================================
# HISTORICAL CHANGE SUMMARY
# =============================================================================

section("CALCULATING HISTORICAL CHANGE SUMMARY")

change_records = []

for variable in historical_variables:

    change_records.extend(
        early_late_summary(
            master,
            variable
        )
    )


change_summary = pd.DataFrame(
    change_records
)

change_summary.to_csv(
    OUT_CHANGE,
    index=False
)


# =============================================================================
# CORRELATIONS — DESCRIPTIVE ONLY
# =============================================================================

section("CALCULATING DESCRIPTIVE WITHIN-SITE CORRELATIONS")

# These correlations describe covariation through time.
#
# They MUST NOT be interpreted as independent causal tests because:
#
#   - observations are temporally autocorrelated
#   - many variables have long-term monotonic trends
#   - shared trends can generate high correlations
#
# Therefore this table is explicitly descriptive/exploratory.

correlation_variables = [
    "sst_degC",
    "population_70km",
    "cropland_percent_70km",
    "pasture_percent_70km",
    "forest_natural_percent_70km",
    "urban_percent_70km",
]

corr_records = []

for site in SITES:

    sub = (
        master[
            master["site"] == site
        ][correlation_variables]
        .copy()
    )

    corr = sub.corr(
        method="spearman"
    )

    for i, var1 in enumerate(correlation_variables):

        for j, var2 in enumerate(correlation_variables):

            if j <= i:
                continue

            corr_records.append(
                {
                    "site": site,
                    "variable_1": var1,
                    "variable_2": var2,
                    "spearman_rho": corr.loc[var1, var2],
                    "interpretation": (
                        "descriptive temporal covariation only; "
                        "not a causal test"
                    ),
                }
            )


correlations = pd.DataFrame(
    corr_records
)

correlations.to_csv(
    OUT_CORRELATION,
    index=False
)


# =============================================================================
# PROVENANCE TABLE
# =============================================================================

section("CREATING VARIABLE PROVENANCE TABLE")

provenance = pd.DataFrame(
    [
        {
            "variable": "sst_degC",
            "source_dataset": "NOAA ERSSTv5",
            "data_class": "observation_based_reconstruction",
            "temporal_support": "annual, 1900-2018",
            "spatial_support": (
                "nearest valid 2-degree ERSST ocean grid cell"
            ),
            "buffer_km": np.nan,
            "historical_or_modern": "historical",
            "interpretation": (
                "regional reconstructed sea-surface temperature"
            ),
        },
        {
            "variable": "population_70km",
            "source_dataset": "ISIMIP3a histsoc population",
            "data_class": "historical_population_reconstruction",
            "temporal_support": "annual, 1900-2018",
            "spatial_support": (
                "population summed within 70-km site buffer"
            ),
            "buffer_km": 70,
            "historical_or_modern": "historical",
            "interpretation": (
                "historical human-population pressure proxy"
            ),
        },
        {
            "variable": "cropland_percent_70km",
            "source_dataset": "ISIMIP3a histsoc land use",
            "data_class": "historical_landuse_reconstruction",
            "temporal_support": "annual, 1900-2018",
            "spatial_support": (
                "0.5-degree source grid intersected with "
                "70-km circular buffer"
            ),
            "buffer_km": 70,
            "historical_or_modern": "historical",
            "interpretation": (
                "cropland percentage of full circular buffer"
            ),
        },
        {
            "variable": "pasture_percent_70km",
            "source_dataset": "ISIMIP3a histsoc land use",
            "data_class": "historical_landuse_reconstruction",
            "temporal_support": "annual, 1900-2018",
            "spatial_support": (
                "0.5-degree source grid intersected with "
                "70-km circular buffer"
            ),
            "buffer_km": 70,
            "historical_or_modern": "historical",
            "interpretation": (
                "pasture percentage of full circular buffer"
            ),
        },
        {
            "variable": "forest_natural_percent_70km",
            "source_dataset": "ISIMIP3a histsoc land use",
            "data_class": "historical_landuse_reconstruction",
            "temporal_support": "annual, 1900-2018",
            "spatial_support": (
                "0.5-degree source grid intersected with "
                "70-km circular buffer"
            ),
            "buffer_km": 70,
            "historical_or_modern": "historical",
            "interpretation": (
                "forest + natural vegetation percentage "
                "of full circular buffer"
            ),
        },
        {
            "variable": "urban_percent_70km",
            "source_dataset": "ISIMIP3a histsoc land use",
            "data_class": "historical_landuse_reconstruction",
            "temporal_support": "annual, 1900-2018",
            "spatial_support": (
                "0.5-degree source grid intersected with "
                "70-km circular buffer"
            ),
            "buffer_km": 70,
            "historical_or_modern": "historical",
            "interpretation": (
                "urban percentage of full circular buffer"
            ),
        },
        {
            "variable": "forest_loss_pressure_z",
            "source_dataset": (
                "derived from ISIMIP3a forest/natural vegetation"
            ),
            "data_class": "derived_standardized_pressure_metric",
            "temporal_support": "annual, 1900-2018",
            "spatial_support": "70-km circular buffer",
            "buffer_km": 70,
            "historical_or_modern": "historical-derived",
            "interpretation": (
                "negative of standardized forest/natural "
                "vegetation; positive = greater forest loss "
                "relative to 1900-1919 baseline"
            ),
        },
    ]
)

provenance.to_csv(
    OUT_PROVENANCE,
    index=False
)


# =============================================================================
# FIGURE 1 — STANDARDIZED HISTORICAL PRESSURES
# =============================================================================

section("CREATING STANDARDIZED HISTORICAL-PRESSURE FIGURE")

plot_variables = [
    (
        "sst_degC_z",
        "Sea-surface temperature"
    ),
    (
        "population_70km_z",
        "Human population"
    ),
    (
        "cropland_percent_70km_z",
        "Cropland"
    ),
    (
        "forest_loss_pressure_z",
        "Forest/natural vegetation loss"
    ),
    (
        "urban_percent_70km_z",
        "Urban area"
    ),
]


fig, axes = plt.subplots(
    len(plot_variables),
    1,
    figsize=(12, 15),
    sharex=True
)

for ax, (variable, title) in zip(
    axes,
    plot_variables
):

    for site in SITES:

        sub = standardized[
            standardized["site"] == site
        ]

        ax.plot(
            sub["year"],
            sub[variable],
            linewidth=2,
            label=site
        )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Standard deviations\nfrom 1900–1919 baseline"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )


axes[-1].set_xlabel(
    "Year"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.975)
)

fig.suptitle(
    "Historical environmental and anthropogenic change relative to early-20th-century baselines",
    fontsize=16,
    fontweight="bold",
    y=0.995
)

fig.text(
    0.5,
    0.015,
    (
        "Each trajectory is standardized independently by site and variable "
        "using the 1900–1919 mean and SD. "
        "Forest/natural vegetation is sign-reversed so positive values indicate "
        "increasing forest-loss pressure. No composite pressure index is shown."
    ),
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0.04, 0.045, 0.98, 0.955]
)

fig.savefig(
    OUT_FIG_Z,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUT_FIG_Z_PDF,
    bbox_inches="tight"
)

plt.close(fig)

log(f"Saved: {OUT_FIG_Z}")
log(f"Saved: {OUT_FIG_Z_PDF}")


# =============================================================================
# FIGURE 2 — PRESSURE TRAJECTORIES BY SITE
# =============================================================================

section("CREATING SITE-CENTRED STANDARDIZED FIGURE")

site_plot_variables = [
    (
        "sst_degC_z",
        "SST"
    ),
    (
        "population_70km_z",
        "Population"
    ),
    (
        "cropland_percent_70km_z",
        "Cropland"
    ),
    (
        "forest_loss_pressure_z",
        "Forest loss"
    ),
    (
        "urban_percent_70km_z",
        "Urban"
    ),
]


fig, axes = plt.subplots(
    len(SITES),
    1,
    figsize=(12, 11),
    sharex=True
)

for ax, site in zip(
    axes,
    SITES
):

    sub = standardized[
        standardized["site"] == site
    ]

    for variable, label in site_plot_variables:

        ax.plot(
            sub["year"],
            sub[variable],
            linewidth=1.8,
            label=label
        )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_title(
        site,
        loc="left",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Baseline SD"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )


axes[-1].set_xlabel(
    "Year"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.965)
)

fig.suptitle(
    "Site-specific historical environmental-pressure trajectories",
    fontsize=16,
    fontweight="bold",
    y=0.995
)

fig.text(
    0.5,
    0.015,
    (
        "Standardized relative to each site's 1900–1919 baseline. "
        "Positive forest-loss values represent declining forest/natural vegetation. "
        "High values are standardized departures, not equivalent ecological effect sizes."
    ),
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0.04, 0.045, 0.98, 0.94]
)

fig.savefig(
    OUT_FIG_SITE,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUT_FIG_SITE_PDF,
    bbox_inches="tight"
)

plt.close(fig)

log(f"Saved: {OUT_FIG_SITE}")
log(f"Saved: {OUT_FIG_SITE_PDF}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("SCRIPT 19 COMPLETE")

log("")
log("MASTER HISTORICAL DATASET")
log("-" * 78)
log(str(OUT_MASTER))

log("")
log("STANDARDIZED HISTORICAL DATASET")
log("-" * 78)
log(str(OUT_STANDARDIZED))

log("")
log("BASELINE STATISTICS")
log("-" * 78)
log(str(OUT_BASELINE))

log("")
log("HISTORICAL CHANGE SUMMARY")
log("-" * 78)
log(str(OUT_CHANGE))

log("")
log("DESCRIPTIVE CORRELATIONS")
log("-" * 78)
log(str(OUT_CORRELATION))

log("")
log("VARIABLE PROVENANCE")
log("-" * 78)
log(str(OUT_PROVENANCE))

log("")
log("QUALITY CONTROL")
log("-" * 78)
log(str(OUT_QC))

log("")
log("EXPECTED MASTER DIMENSIONS")
log("-" * 78)

log(
    f"3 sites × 119 years = {expected_rows:,} rows"
)

log(
    f"Actual master rows = {len(master):,}"
)

if len(master) == expected_rows:
    log("PASS: expected site-year dimensions obtained.")
else:
    log("WARNING: master dimensions differ from expectation.")


log("")
log("MISSING VALUES")
log("-" * 78)

missing_summary = (
    master[historical_variables]
    .isna()
    .sum()
)

for variable, n in missing_summary.items():

    log(
        f"{variable}: {int(n)}"
    )


log("")
log("IMPORTANT INTERPRETATION")
log("-" * 78)

log(
    "1. These standardized trajectories do NOT constitute a composite "
    "environmental-pressure index."
)

log(
    "2. Z-scores only express departure from each site's own "
    "1900–1919 baseline."
)

log(
    "3. A large z-score can occur when baseline variability is small; "
    "it must not automatically be interpreted as a proportionally larger "
    "ecological effect."
)

log(
    "4. Forest-loss pressure is only the sign-reversed standardized "
    "forest/natural vegetation trajectory."
)

log(
    "5. Correlations among historical variables are descriptive because "
    "long-term trends and temporal autocorrelation violate simple "
    "independence assumptions."
)

log(
    "6. ERSSTv5 is NOT a 70-km buffer measurement. It represents the "
    "nearest valid coarse regional SST reconstruction grid cell."
)

log(
    "7. ETOPO and Allen Coral Atlas metrics remain separate modern/static "
    "ecological-context variables."
)


# =============================================================================
# WRITE LOG
# =============================================================================

OUT_LOG.write_text(
    "\n".join(log_lines),
    encoding="utf-8"
)

log("")
log(f"Log saved: {OUT_LOG}")