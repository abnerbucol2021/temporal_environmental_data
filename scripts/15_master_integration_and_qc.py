"""
SCRIPT 15
MASTER INTEGRATION + QUALITY CONTROL

Project:
D:\\temporal_environmental_data

Purpose
-------
Integrate the completed historical, bathymetric, and modern benthic
datasets into publication-ready master tables while preserving their
different evidence classes.

Historical time-varying data:
    1. NOAA ERSSTv5 SST, 1900-2018
    2. ISIMIP3a historical population reconstruction, 1900-2018

Static / modern spatial context:
    3. NOAA ETOPO 2022 bathymetric habitat metrics
    4. Allen Coral Atlas benthic habitat composition
    5. Allen Coral Atlas benthic diversity
    6. Standardized sand+rubble raster patch configuration
       Primary definition:
           100 m raster
           8-neighbour connectivity
           minimum patch area = 5 ha
    7. Exploratory bathymetric habitat-capacity index

IMPORTANT
---------
Historical and modern/static variables are NOT treated as equivalent
time-series observations.

Allen Coral Atlas metrics describe modern mapped benthic habitat.
ETOPO bathymetry is static geomorphological context.
ISIMIP population is a historical reconstruction.
ERSST is an observation-based SST reconstruction.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(r"D:\temporal_environmental_data")

EXTRACTED = ROOT / "03_extracted"
TABLES = ROOT / "04_tables"
LOGS = ROOT / "07_logs"

TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


# =====================================================================
# INPUT FILES
# =====================================================================

ERSST_FILE = (
    EXTRACTED /
    "ersstv5_annual_1900_2018.csv"
)

POP_FILE = (
    EXTRACTED /
    "population_ISIMIP3a_5arcmin_annual_1900_2018.csv"
)

BATHY_FILE = (
    TABLES /
    "spawning_habitat_capacity_bathymetry_70km_summary.csv"
)

BENTHIC_FILE = (
    TABLES /
    "allen_coral_atlas_benthic_70km_summary.csv"
)

DIVERSITY_FILE = (
    TABLES /
    "benthic_diversity_70km.csv"
)

# ---------------------------------------------------------------------
# SCRIPT 14 PRIMARY standardized raster configuration
# ---------------------------------------------------------------------

PATCH_FILE = (
    TABLES /
    "sand_rubble_raster_patch_metrics_PRIMARY_100m_5ha.csv"
)

BATHY_INDEX_FILE = (
    TABLES /
    "spawning_habitat_capacity_bathymetry_exploratory_index.csv"
)


# =====================================================================
# OUTPUT FILES
# =====================================================================

MASTER_ANNUAL = (
    EXTRACTED /
    "master_historical_annual_1900_2018.csv"
)

CHANGE_SUMMARY = (
    TABLES /
    "master_historical_change_summary.csv"
)

SITE_COMPARISON = (
    TABLES /
    "master_site_comparison_historical_modern.csv"
)

PROVENANCE_FILE = (
    TABLES /
    "master_variable_provenance.csv"
)

QC_FILE = (
    TABLES /
    "master_integration_QC.csv"
)

LOG_FILE = (
    LOGS /
    "15_master_integration_and_QC.log"
)


# =====================================================================
# CONSTANTS
# =====================================================================

SITES = ["Jolo", "Masinloc", "Palapag"]

EXPECTED_YEARS = np.arange(1900, 2019)

EARLY_PERIOD = (1900, 1919)
MODERN_PERIOD = (1999, 2018)


# =====================================================================
# HELPERS
# =====================================================================

log_lines = []


def log(message=""):
    print(message)
    log_lines.append(str(message))


def load_required(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired input file not found:\n{path}"
        )

    df = pd.read_csv(path)

    log(f"Loaded: {path.name} {len(df):,}")

    return df


def clean_site(df):
    if "site" not in df.columns:
        raise ValueError(
            "Input table does not contain a 'site' column."
        )

    df = df.copy()

    df["site"] = (
        df["site"]
        .astype(str)
        .str.strip()
    )

    return df


def numeric_columns(df, exclude=None):
    if exclude is None:
        exclude = []

    return [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in exclude
    ]


def safe_mean(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.mean()


def find_first_column(df, candidates):
    """
    Return first exact or case-insensitive candidate column.
    """

    lower_lookup = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    return None


def find_column_containing(df, required_terms, excluded_terms=None):
    """
    Find first column containing all required terms.
    """

    if excluded_terms is None:
        excluded_terms = []

    for c in df.columns:

        lc = c.lower()

        if (
            all(term.lower() in lc for term in required_terms)
            and
            not any(term.lower() in lc for term in excluded_terms)
        ):
            return c

    return None


def prefix_nonkey_columns(df, prefix):
    """
    Prefix columns except site.
    Prevents accidental collisions between datasets.
    """

    rename = {}

    for c in df.columns:
        if c != "site":
            rename[c] = f"{prefix}{c}"

    return df.rename(columns=rename)


# =====================================================================
# START
# =====================================================================

log("=" * 78)
log("SCRIPT 15 — MASTER INTEGRATION + QC")
log("=" * 78)


# =====================================================================
# 1. LOAD HISTORICAL DATA
# =====================================================================

ersst = clean_site(load_required(ERSST_FILE))
population = clean_site(load_required(POP_FILE))


# =====================================================================
# 2. LOAD STATIC / MODERN HABITAT DATA
# =====================================================================

bathymetry = clean_site(load_required(BATHY_FILE))
benthic = clean_site(load_required(BENTHIC_FILE))
diversity = clean_site(load_required(DIVERSITY_FILE))

patch14 = clean_site(load_required(PATCH_FILE))

bathy_index = clean_site(load_required(BATHY_INDEX_FILE))


# =====================================================================
# 3. INSPECT SCRIPT 14 PRIMARY TABLE
# =====================================================================

log("")
log("SCRIPT 14 PRIMARY PATCH TABLE")
log("-" * 78)

log("Definition:")
log("  raster resolution = 100 m")
log("  connectivity = 8-neighbour")
log("  minimum retained patch area = 5 ha")

log("")
log("Columns:")

for c in patch14.columns:
    log(f"  {c}")


# =====================================================================
# 4. STANDARDIZE HISTORICAL YEAR FIELDS
# =====================================================================

for name, df in [
    ("ERSST", ersst),
    ("population", population)
]:

    if "year" not in df.columns:
        raise ValueError(
            f"{name} table does not contain year."
        )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    if df["year"].isna().any():
        raise ValueError(
            f"{name} contains invalid year values."
        )

    df["year"] = df["year"].astype(int)


# =====================================================================
# 5. HISTORICAL QC
# =====================================================================

qc_records = []

for name, df in [
    ("ERSSTv5", ersst),
    ("ISIMIP3a_population", population)
]:

    for site in SITES:

        sub = df[df["site"] == site]

        years = np.sort(
            sub["year"].dropna().unique()
        )

        complete = np.array_equal(
            years,
            EXPECTED_YEARS
        )

        duplicates = sub.duplicated(
            subset=["site", "year"]
        ).sum()

        qc_records.append({
            "dataset": name,
            "site": site,
            "rows": len(sub),
            "first_year":
                years.min() if len(years) else np.nan,
            "last_year":
                years.max() if len(years) else np.nan,
            "n_years": len(years),
            "complete_1900_2018": complete,
            "duplicate_site_year_rows": duplicates
        })


# =====================================================================
# 6. IDENTIFY ERSST VALUE COLUMN
# =====================================================================

sst_col = find_first_column(
    ersst,
    [
        "sst_mean",
        "mean_sst",
        "annual_mean_sst",
        "sst"
    ]
)

if sst_col is None:
    sst_col = find_column_containing(
        ersst,
        ["sst"],
        excluded_terms=[
            "max",
            "min",
            "sd",
            "std",
            "latitude",
            "longitude"
        ]
    )

if sst_col is None:
    raise ValueError(
        "Could not identify annual SST column in ERSST table."
    )

log("")
log(f"ERSST annual SST column: {sst_col}")


# =====================================================================
# 7. IDENTIFY POPULATION VALUE COLUMN
# =====================================================================

pop_col = find_first_column(
    population,
    [
        "population",
        "population_70km",
        "population_total",
        "population_sum"
    ]
)

if pop_col is None:
    pop_col = find_column_containing(
        population,
        ["population"],
        excluded_terms=[
            "latitude",
            "longitude",
            "grid",
            "distance"
        ]
    )

if pop_col is None:

    candidates = numeric_columns(
        population,
        exclude=["year"]
    )

    if len(candidates) == 1:
        pop_col = candidates[0]

if pop_col is None:
    raise ValueError(
        "Could not identify population column."
    )

log(f"Population annual column: {pop_col}")


# =====================================================================
# 8. BUILD HISTORICAL MASTER
# =====================================================================

ersst_master = ersst[
    ["site", "year", sst_col]
].copy()

ersst_master = ersst_master.rename(
    columns={
        sst_col: "sst_ersstv5_degC"
    }
)


population_master = population[
    ["site", "year", pop_col]
].copy()

population_master = population_master.rename(
    columns={
        pop_col:
        "population_ISIMIP3a_70km"
    }
)


historical = ersst_master.merge(
    population_master,
    on=["site", "year"],
    how="outer",
    validate="one_to_one"
)


historical = historical.sort_values(
    ["site", "year"]
).reset_index(drop=True)


# =====================================================================
# 9. HISTORICAL PERIOD SUMMARIES
# =====================================================================

historical_change_records = []

for site in SITES:

    sub = historical[
        historical["site"] == site
    ].copy()

    early = sub[
        sub["year"].between(
            EARLY_PERIOD[0],
            EARLY_PERIOD[1]
        )
    ]

    modern = sub[
        sub["year"].between(
            MODERN_PERIOD[0],
            MODERN_PERIOD[1]
        )
    ]

    early_sst = safe_mean(
        early["sst_ersstv5_degC"]
    )

    modern_sst = safe_mean(
        modern["sst_ersstv5_degC"]
    )

    early_pop = safe_mean(
        early["population_ISIMIP3a_70km"]
    )

    modern_pop = safe_mean(
        modern["population_ISIMIP3a_70km"]
    )

    if (
        pd.notna(early_pop)
        and early_pop != 0
        and pd.notna(modern_pop)
    ):
        pop_percent_change = (
            (modern_pop - early_pop)
            / early_pop
            * 100
        )
    else:
        pop_percent_change = np.nan

    historical_change_records.append({

        "site": site,

        "early_period":
            f"{EARLY_PERIOD[0]}-{EARLY_PERIOD[1]}",

        "modern_period":
            f"{MODERN_PERIOD[0]}-{MODERN_PERIOD[1]}",

        "sst_early_mean_degC":
            early_sst,

        "sst_modern_mean_degC":
            modern_sst,

        "sst_change_degC":
            modern_sst - early_sst,

        "population_early_mean":
            early_pop,

        "population_modern_mean":
            modern_pop,

        "population_absolute_change":
            modern_pop - early_pop,

        "population_percent_change":
            pop_percent_change
    })


historical_change = pd.DataFrame(
    historical_change_records
)


# =====================================================================
# 10. BUILD MODERN / STATIC HABITAT MASTER
# =====================================================================

# Keep dataset identity visible in column names.

bathy_mod = prefix_nonkey_columns(
    bathymetry,
    "bathy_"
)

benthic_mod = prefix_nonkey_columns(
    benthic,
    "aca_"
)

diversity_mod = prefix_nonkey_columns(
    diversity,
    "aca_div_"
)

patch_mod = prefix_nonkey_columns(
    patch14,
    "aca_patch100m5ha_"
)

index_mod = prefix_nonkey_columns(
    bathy_index,
    "bathy_index_"
)


modern_context = pd.DataFrame({
    "site": SITES
})


for df in [
    bathy_mod,
    benthic_mod,
    diversity_mod,
    patch_mod,
    index_mod
]:

    modern_context = modern_context.merge(
        df,
        on="site",
        how="left",
        validate="one_to_one"
    )


# =====================================================================
# 11. ADD HISTORICAL CHANGE TO SITE COMPARISON
# =====================================================================

site_comparison = historical_change.merge(
    modern_context,
    on="site",
    how="left",
    validate="one_to_one"
)


# =====================================================================
# 12. ADD EXPLICIT EVIDENCE LABELS
# =====================================================================

site_comparison[
    "historical_SST_evidence_class"
] = "observation_based_reconstruction"

site_comparison[
    "historical_population_evidence_class"
] = "historical_population_reconstruction"

site_comparison[
    "bathymetry_evidence_class"
] = "static_geomorphological_dataset"

site_comparison[
    "benthic_evidence_class"
] = "modern_satellite_derived_benthic_map"

site_comparison[
    "patch_configuration_evidence_class"
] = "derived_modern_spatial_metric"

site_comparison[
    "patch_configuration_primary_definition"
] = (
    "100 m raster; 8-neighbour connectivity; "
    "minimum patch area 5 ha"
)


# =====================================================================
# 13. PROVENANCE TABLE
# =====================================================================

provenance = pd.DataFrame([

    {
        "dataset": "NOAA ERSSTv5",
        "variable_group": "sea-surface temperature",
        "period_used": "1900-2018",
        "nominal_resolution":
            "2 degree",
        "data_class":
            "observation_based_reconstruction",
        "temporal_status":
            "historical_time_series",
        "processing":
            "nearest available ERSST ocean grid cell",
        "interpretation":
            "regional historical SST reconstruction"
    },

    {
        "dataset": "ISIMIP3a histsoc population",
        "variable_group": "human population",
        "period_used": "1900-2018",
        "nominal_resolution":
            "5 arcmin",
        "data_class":
            "historical_population_reconstruction",
        "temporal_status":
            "historical_time_series",
        "processing":
            "population aggregated within site buffers",
        "interpretation":
            "historical reconstructed population pressure"
    },

    {
        "dataset": "NOAA ETOPO 2022",
        "variable_group":
            "bathymetry and shallow-habitat geometry",
        "period_used":
            "static modern terrain product",
        "nominal_resolution":
            "15 arc-second",
        "data_class":
            "static_geomorphological_dataset",
        "temporal_status":
            "static_context",
        "processing":
            "0-10, 10-20 and 20-30 m bathymetric metrics "
            "within 70 km",
        "interpretation":
            "geomorphological habitat-capacity proxy"
    },

    {
        "dataset": "Allen Coral Atlas",
        "variable_group":
            "mapped benthic habitat",
        "period_used":
            "modern",
        "nominal_resolution":
            "Allen Coral Atlas mapped polygons",
        "data_class":
            "modern_satellite_derived_benthic_map",
        "temporal_status":
            "modern_context",
        "processing":
            "WFS benthic polygons clipped to 70-km site buffers",
        "interpretation":
            "modern mapped benthic habitat only; "
            "unmapped marine area is not unsuitable habitat"
    },

    {
        "dataset":
            "Allen Coral Atlas derived raster patch analysis",
        "variable_group":
            "sand+rubble habitat configuration",
        "period_used":
            "modern",
        "nominal_resolution":
            "100 m analysis raster",
        "data_class":
            "derived_modern_spatial_metric",
        "temporal_status":
            "modern_context",
        "processing":
            "sand+rubble rasterized at 100 m; "
            "8-neighbour connectivity; "
            "patches <5 ha removed",
        "interpretation":
            "standardized spatial configuration of mapped "
            "sand+rubble habitat; not historical reconstruction"
    },

    {
        "dataset":
            "Exploratory bathymetric habitat-capacity index",
        "variable_group":
            "composite bathymetric habitat context",
        "period_used":
            "static",
        "nominal_resolution":
            "derived from ETOPO 2022 metrics",
        "data_class":
            "exploratory_derived_index",
        "temporal_status":
            "static_context",
        "processing":
            "equal-weight standardized composite",
        "interpretation":
            "exploratory only; not a validated biological "
            "spawning-habitat index"
    }
])


# =====================================================================
# 14. MODERN / STATIC QC
# =====================================================================

for dataset_name, df in [

    ("ETOPO_bathymetry", bathymetry),

    ("Allen_Coral_Atlas_benthic", benthic),

    ("Allen_Coral_Atlas_diversity", diversity),

    ("Allen_Coral_Atlas_patch_100m_5ha", patch14),

    ("Bathymetric_exploratory_index", bathy_index)

]:

    observed_sites = sorted(
        df["site"]
        .dropna()
        .unique()
        .tolist()
    )

    qc_records.append({

        "dataset": dataset_name,

        "site": "ALL",

        "rows": len(df),

        "first_year": np.nan,

        "last_year": np.nan,

        "n_years": np.nan,

        "complete_1900_2018": np.nan,

        "duplicate_site_year_rows":
            df["site"].duplicated().sum(),

        "site_coverage":
            ";".join(observed_sites),

        "all_three_sites_present":
            set(SITES).issubset(
                set(observed_sites)
            )
    })


qc = pd.DataFrame(qc_records)


# =====================================================================
# 15. ADD HIGH-LEVEL INTEGRATION QC
# =====================================================================

integration_checks = []


def add_check(check, result, detail):

    integration_checks.append({
        "dataset": "MASTER_INTEGRATION",
        "site": "ALL",
        "check": check,
        "result": bool(result),
        "detail": detail
    })


add_check(
    "historical_master_expected_rows",
    len(historical) == 357,
    f"Observed {len(historical)}; expected 357"
)


add_check(
    "site_comparison_expected_rows",
    len(site_comparison) == 3,
    f"Observed {len(site_comparison)}; expected 3"
)


add_check(
    "script14_primary_table_loaded",
    len(patch14) == 3,
    f"Loaded {len(patch14)} site rows from "
    f"{PATCH_FILE.name}"
)


add_check(
    "historical_master_no_duplicate_site_year",
    historical.duplicated(
        subset=["site", "year"]
    ).sum() == 0,
    "Each site-year should occur once."
)


add_check(
    "site_comparison_no_duplicate_sites",
    site_comparison["site"].duplicated().sum() == 0,
    "Each site should occur once."
)


add_check(
    "historical_master_all_three_sites",
    set(historical["site"].unique()) == set(SITES),
    "Expected Jolo, Masinloc and Palapag."
)


add_check(
    "patch_table_all_three_sites",
    set(patch14["site"].unique()) == set(SITES),
    "Expected Jolo, Masinloc and Palapag."
)


integration_qc = pd.DataFrame(
    integration_checks
)


# Align QC columns before concatenation.

for c in integration_qc.columns:
    if c not in qc.columns:
        qc[c] = np.nan

for c in qc.columns:
    if c not in integration_qc.columns:
        integration_qc[c] = np.nan


qc = pd.concat(
    [
        qc,
        integration_qc[qc.columns]
    ],
    ignore_index=True
)


# =====================================================================
# 16. NUMERIC SANITY CHECKS
# =====================================================================

log("")
log("NUMERIC SANITY CHECKS")
log("-" * 78)


if (
    historical["population_ISIMIP3a_70km"]
    .dropna()
    .lt(0)
    .any()
):
    log("WARNING: negative population values detected.")
else:
    log("Population: no negative values detected.")


sst_values = (
    historical["sst_ersstv5_degC"]
    .dropna()
)

if len(sst_values):

    log(
        "SST range: "
        f"{sst_values.min():.3f} to "
        f"{sst_values.max():.3f} °C"
    )


# =====================================================================
# 17. SCRIPT 14 PRIMARY METRIC SUMMARY
# =====================================================================

log("")
log("STANDARDIZED SAND+RUBBLE CONFIGURATION")
log("100 m raster | 8-neighbour | minimum patch area 5 ha")
log("-" * 78)


preferred_patch_columns = [

    "site",

    "retained_patch_count",

    "patch_count",

    "largest_patch_km2",

    "largest_patch_area_km2",

    "mean_nearest_neighbor_distance_km",

    "mean_nearest_neighbour_distance_km",

    "nearest_patch_distance_km",

    "distance_site_to_nearest_patch_km",

    "site_to_nearest_patch_distance_km"
]


available = [
    c for c in preferred_patch_columns
    if c in patch14.columns
]


if len(available) > 1:

    log(
        patch14[available]
        .to_string(index=False)
    )

else:

    # If names differ from expected, print full table.
    log(
        patch14
        .to_string(index=False)
    )


# =====================================================================
# 18. WRITE OUTPUTS
# =====================================================================

historical.to_csv(
    MASTER_ANNUAL,
    index=False
)

historical_change.to_csv(
    CHANGE_SUMMARY,
    index=False
)

site_comparison.to_csv(
    SITE_COMPARISON,
    index=False
)

provenance.to_csv(
    PROVENANCE_FILE,
    index=False
)

qc.to_csv(
    QC_FILE,
    index=False
)


# =====================================================================
# 19. FINAL REPORT
# =====================================================================

log("")
log("=" * 78)
log("SCRIPT 15 COMPLETE")
log("=" * 78)

log("")
log("Historical master:")
log(str(MASTER_ANNUAL))

log("")
log("Historical change summary:")
log(str(CHANGE_SUMMARY))

log("")
log("Integrated historical + modern/static site comparison:")
log(str(SITE_COMPARISON))

log("")
log("Variable provenance:")
log(str(PROVENANCE_FILE))

log("")
log("Integration QC:")
log(str(QC_FILE))

log("")
log("MASTER DIMENSIONS")
log("-" * 78)

log(
    f"Historical annual master: "
    f"{historical.shape[0]} rows x "
    f"{historical.shape[1]} columns"
)

log(
    f"Site comparison: "
    f"{site_comparison.shape[0]} rows x "
    f"{site_comparison.shape[1]} columns"
)

log("")
log("Evidence classes remain explicitly separated:")

log(
    "  ERSSTv5      = observation-based historical reconstruction"
)

log(
    "  Population   = historical population reconstruction"
)

log(
    "  ETOPO        = static geomorphological context"
)

log(
    "  Allen Atlas  = modern satellite-derived benthic context"
)

log(
    "  Patch metrics = derived modern spatial metrics "
    "(100 m, 8-neighbour, >=5 ha)"
)

log(
    "  Bathy index  = exploratory derived index"
)

log("")
log(
    "IMPORTANT: modern/static habitat variables must not be "
    "interpreted as 1900-2018 habitat time series."
)


# =====================================================================
# 20. SAVE LOG
# =====================================================================

LOG_FILE.write_text(
    "\n".join(log_lines),
    encoding="utf-8"
)

log("")
log(f"Log saved: {LOG_FILE}")