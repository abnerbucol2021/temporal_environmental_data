# =============================================================================
# SCRIPT 20 — HISTORICAL ENVIRONMENTAL CHANGE PROFILES
# =============================================================================
#
# Purpose
# -------
# Quantify and visualize historical environmental change at Jolo, Masinloc,
# and Palapag using the integrated historical dataset produced by Script 19.
#
# Primary comparison:
#   Early historical baseline = 1900–1919
#   Contemporary endpoint      = 1999–2018
#
# Variables:
#   1. SST
#   2. Population
#   3. Cropland
#   4. Pasture
#   5. Forest + natural vegetation
#   6. Urban area
#
# IMPORTANT:
# - This script DOES NOT create a composite environmental-pressure index.
# - Variables remain separate.
# - Forest loss is expressed as a positive pressure metric:
#       early forest % - late forest %
# - Land-use percentages refer to the full 70-km circular buffer,
#   including marine area.
# - ERSSTv5 is coarse regional reconstructed SST, not a literal 70-km
#   buffer measurement.
# - Population and land use are historical reconstructions.
#
# =============================================================================

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(r"D:\temporal_environmental_data")

INPUT = (
    ROOT
    / "03_extracted"
    / "historical_environmental_pressure_master_1900_2018.csv"
)

OUT_TABLES = ROOT / "04_tables"
OUT_FIGURES = ROOT / "05_figures"
OUT_LOGS = ROOT / "07_logs"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUT_LOGS.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_LOGS / "20_historical_environmental_change_profiles.log"


# =============================================================================
# 2. SIMPLE LOGGING
# =============================================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_handle = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_handle)


print("=" * 78)
print("SCRIPT 20 — HISTORICAL ENVIRONMENTAL CHANGE PROFILES")
print("=" * 78)


# =============================================================================
# 3. SETTINGS
# =============================================================================

SITES = ["Jolo", "Masinloc", "Palapag"]

EARLY_START = 1900
EARLY_END = 1919

LATE_START = 1999
LATE_END = 2018


VARIABLES = {
    "sst_degC": {
        "label": "Sea-surface temperature",
        "short": "SST",
        "unit": "°C",
        "change_type": "difference",
        "pressure_direction": 1,
    },

    "population_70km": {
        "label": "Population",
        "short": "Population",
        "unit": "people",
        "change_type": "percent",
        "pressure_direction": 1,
    },

    "cropland_percent_70km": {
        "label": "Cropland",
        "short": "Cropland",
        "unit": "% of buffer",
        "change_type": "percentage_points",
        "pressure_direction": 1,
    },

    "pasture_percent_70km": {
        "label": "Pasture",
        "short": "Pasture",
        "unit": "% of buffer",
        "change_type": "percentage_points",
        "pressure_direction": 1,
    },

    "forest_natural_percent_70km": {
        "label": "Forest + natural vegetation",
        "short": "Forest loss",
        "unit": "% of buffer",
        "change_type": "percentage_points",
        "pressure_direction": -1,
    },

    "urban_percent_70km": {
        "label": "Urban area",
        "short": "Urban area",
        "unit": "% of buffer",
        "change_type": "percentage_points",
        "pressure_direction": 1,
    },
}


# =============================================================================
# 4. LOAD DATA
# =============================================================================

print("\nCHECKING INPUT")
print("-" * 78)

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nRequired Script 19 master file not found:\n{INPUT}"
    )

print(f"FOUND: {INPUT}")


df = pd.read_csv(INPUT)

print("\nMASTER DATASET")
print("-" * 78)
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")

required_columns = ["site", "year"] + list(VARIABLES.keys())

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise KeyError(
        "\nRequired columns missing from Script 19 master:\n"
        + "\n".join(missing_columns)
    )


# =============================================================================
# 5. BASIC QC
# =============================================================================

df["year"] = pd.to_numeric(df["year"], errors="coerce")

df = df[
    df["site"].isin(SITES)
    & df["year"].between(1900, 2018)
].copy()

df = df.sort_values(["site", "year"]).reset_index(drop=True)


print("\nINPUT QC")
print("-" * 78)

expected_rows = len(SITES) * (2018 - 1900 + 1)

print(f"Expected rows: {expected_rows}")
print(f"Actual rows:   {len(df)}")

if len(df) != expected_rows:
    print("WARNING: unexpected number of rows.")
else:
    print("PASS: expected dimensions obtained.")


for site in SITES:

    d = df[df["site"] == site]

    years = sorted(d["year"].dropna().astype(int).unique())

    missing_years = sorted(
        set(range(1900, 2019)) - set(years)
    )

    print(f"\n{site}")
    print(f"  rows: {len(d)}")

    if years:
        print(f"  years: {min(years)}–{max(years)}")

    print(f"  missing years: {len(missing_years)}")


# =============================================================================
# 6. EARLY AND LATE PERIOD MEANS
# =============================================================================

print("\n" + "=" * 78)
print("CALCULATING EARLY AND LATE PERIOD MEANS")
print("=" * 78)

early = df[
    df["year"].between(EARLY_START, EARLY_END)
].copy()

late = df[
    df["year"].between(LATE_START, LATE_END)
].copy()


print(
    f"Early period: {EARLY_START}–{EARLY_END} "
    f"({EARLY_END - EARLY_START + 1} years)"
)

print(
    f"Late period:  {LATE_START}–{LATE_END} "
    f"({LATE_END - LATE_START + 1} years)"
)


rows = []


for site in SITES:

    de = early[early["site"] == site]
    dl = late[late["site"] == site]

    if len(de) != 20:
        print(
            f"WARNING: {site} early period contains "
            f"{len(de)} observations instead of 20."
        )

    if len(dl) != 20:
        print(
            f"WARNING: {site} late period contains "
            f"{len(dl)} observations instead of 20."
        )

    for variable, info in VARIABLES.items():

        early_values = pd.to_numeric(
            de[variable],
            errors="coerce"
        )

        late_values = pd.to_numeric(
            dl[variable],
            errors="coerce"
        )

        early_mean = early_values.mean()
        late_mean = late_values.mean()

        early_sd = early_values.std(ddof=1)
        late_sd = late_values.std(ddof=1)

        raw_change = late_mean - early_mean

        if np.isfinite(early_mean) and early_mean != 0:
            relative_change_percent = (
                raw_change / early_mean
            ) * 100
        else:
            relative_change_percent = np.nan


        # -------------------------------------------------------------
        # Directionally consistent pressure change
        # -------------------------------------------------------------

        if info["pressure_direction"] == 1:
            pressure_change = raw_change
        else:
            # Forest decline becomes positive pressure
            pressure_change = -raw_change


        rows.append({
            "site": site,
            "variable": variable,
            "variable_label": info["label"],
            "display_label": info["short"],
            "unit": info["unit"],

            "early_period": f"{EARLY_START}-{EARLY_END}",
            "late_period": f"{LATE_START}-{LATE_END}",

            "early_n": early_values.notna().sum(),
            "late_n": late_values.notna().sum(),

            "early_mean": early_mean,
            "early_sd": early_sd,

            "late_mean": late_mean,
            "late_sd": late_sd,

            "raw_change": raw_change,
            "relative_change_percent": relative_change_percent,

            "pressure_direction": info["pressure_direction"],
            "pressure_change": pressure_change,
        })


change = pd.DataFrame(rows)


# =============================================================================
# 7. CREATE INTERPRETABLE CHANGE METRICS
# =============================================================================

print("\nCALCULATING INTERPRETABLE CHANGE METRICS")
print("-" * 78)


change["reported_change"] = np.nan
change["reported_change_unit"] = ""


for idx, row in change.iterrows():

    variable = row["variable"]

    if variable == "sst_degC":

        change.loc[idx, "reported_change"] = row["raw_change"]
        change.loc[idx, "reported_change_unit"] = "°C"

    elif variable == "population_70km":

        change.loc[idx, "reported_change"] = (
            row["relative_change_percent"]
        )

        change.loc[idx, "reported_change_unit"] = "%"

    elif variable == "forest_natural_percent_70km":

        # Positive number = forest/natural vegetation loss
        change.loc[idx, "reported_change"] = (
            -row["raw_change"]
        )

        change.loc[
            idx,
            "reported_change_unit"
        ] = "percentage points lost"

    else:

        change.loc[idx, "reported_change"] = row["raw_change"]

        change.loc[
            idx,
            "reported_change_unit"
        ] = "percentage points"


# =============================================================================
# 8. SITE PROFILE TABLE
# =============================================================================

print("\nCREATING SITE PROFILE TABLE")
print("-" * 78)


profile_rows = []


for site in SITES:

    ds = change[change["site"] == site]

    def get_change(variable):

        x = ds.loc[
            ds["variable"] == variable,
            "reported_change"
        ]

        if len(x) == 0:
            return np.nan

        return float(x.iloc[0])


    profile_rows.append({

        "site": site,

        "sst_change_degC":
            get_change("sst_degC"),

        "population_change_percent":
            get_change("population_70km"),

        "cropland_change_pp":
            get_change("cropland_percent_70km"),

        "pasture_change_pp":
            get_change("pasture_percent_70km"),

        "forest_natural_loss_pp":
            get_change("forest_natural_percent_70km"),

        "urban_change_pp":
            get_change("urban_percent_70km"),
    })


profile = pd.DataFrame(profile_rows)


# =============================================================================
# 9. STANDARDIZE CHANGE METRICS ACROSS SITES
# =============================================================================
#
# IMPORTANT:
# These standardized values are ONLY for graphical comparison.
#
# They are not a composite index.
#
# For each variable:
#
#     z = (site change - mean change among 3 sites) / SD
#
# With only three sites, these values are descriptive.
# =============================================================================

print("\nSTANDARDIZING CROSS-SITE CHANGE FOR VISUALIZATION")
print("-" * 78)


change_columns = [
    "sst_change_degC",
    "population_change_percent",
    "cropland_change_pp",
    "pasture_change_pp",
    "forest_natural_loss_pp",
    "urban_change_pp",
]


for col in change_columns:

    values = pd.to_numeric(
        profile[col],
        errors="coerce"
    )

    mean = values.mean()
    sd = values.std(ddof=1)

    z_col = col + "_z"

    if np.isfinite(sd) and sd > 0:
        profile[z_col] = (values - mean) / sd
    else:
        profile[z_col] = 0.0


# =============================================================================
# 10. DESCRIPTIVE RANKINGS
# =============================================================================
#
# Rank 1 = largest historical increase / pressure change.
#
# DO NOT sum these ranks into an overall site rank.
# =============================================================================

print("\nCALCULATING DESCRIPTIVE VARIABLE-SPECIFIC RANKS")
print("-" * 78)


for col in change_columns:

    rank_col = col + "_rank"

    profile[rank_col] = (
        profile[col]
        .rank(
            ascending=False,
            method="min"
        )
        .astype("Int64")
    )


# =============================================================================
# 11. SAVE TABLES
# =============================================================================

CHANGE_FILE = (
    OUT_TABLES
    / "historical_environmental_change_early_vs_late.csv"
)

PROFILE_FILE = (
    OUT_TABLES
    / "historical_environmental_change_site_profiles.csv"
)


change.to_csv(
    CHANGE_FILE,
    index=False
)

profile.to_csv(
    PROFILE_FILE,
    index=False
)


print(f"Saved: {CHANGE_FILE}")
print(f"Saved: {PROFILE_FILE}")


# =============================================================================
# 12. PRINT MAIN RESULTS
# =============================================================================

print("\n" + "=" * 78)
print("HISTORICAL CHANGE PROFILE")
print(f"{EARLY_START}-{EARLY_END} MEAN vs {LATE_START}-{LATE_END} MEAN")
print("=" * 78)


display_columns = [
    "site",
    "sst_change_degC",
    "population_change_percent",
    "cropland_change_pp",
    "pasture_change_pp",
    "forest_natural_loss_pp",
    "urban_change_pp",
]


print(
    profile[display_columns]
    .round({
        "sst_change_degC": 3,
        "population_change_percent": 1,
        "cropland_change_pp": 2,
        "pasture_change_pp": 2,
        "forest_natural_loss_pp": 2,
        "urban_change_pp": 3,
    })
    .to_string(index=False)
)


# =============================================================================
# 13. FIGURE 1 — RAW HISTORICAL CHANGE PROFILE
# =============================================================================
#
# Separate panels are necessary because variables have different units.
# =============================================================================

print("\nCREATING FIGURE 1 — RAW CHANGE PROFILE")
print("-" * 78)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)

axes = axes.flatten()


figure_variables = [
    (
        "sst_change_degC",
        "SST warming",
        "Change (°C)"
    ),

    (
        "population_change_percent",
        "Population growth",
        "Change (%)"
    ),

    (
        "cropland_change_pp",
        "Cropland expansion",
        "Change (percentage points)"
    ),

    (
        "pasture_change_pp",
        "Pasture expansion",
        "Change (percentage points)"
    ),

    (
        "forest_natural_loss_pp",
        "Forest + natural vegetation loss",
        "Loss (percentage points)"
    ),

    (
        "urban_change_pp",
        "Urban expansion",
        "Change (percentage points)"
    ),
]


for ax, (column, title, ylabel) in zip(
    axes,
    figure_variables
):

    values = profile.set_index("site").loc[
        SITES,
        column
    ]

    bars = ax.bar(
        SITES,
        values.values
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(ylabel)

    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )


    # -------------------------------------------------------------
    # Value labels
    # -------------------------------------------------------------

    max_abs = np.nanmax(
        np.abs(values.values)
    )

    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1


    for bar, value in zip(
        bars,
        values.values
    ):

        if not np.isfinite(value):
            continue

        if column == "population_change_percent":
            label = f"{value:+.0f}%"

        elif column == "sst_change_degC":
            label = f"{value:+.2f}"

        else:
            label = f"{value:+.2f}"


        offset = max_abs * 0.035

        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"


        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=9
        )


fig.suptitle(
    "Historical environmental change at Philippine study sites",
    fontsize=17,
    fontweight="bold",
    y=0.98
)


fig.text(
    0.5,
    0.015,
    (
        "Change compares 1900–1919 means with 1999–2018 means. "
        "Forest values represent loss, so positive values indicate greater "
        "forest/natural-vegetation decline. Variables are not combined "
        "into a composite pressure index."
    ),
    ha="center",
    fontsize=9
)


plt.tight_layout(
    rect=[0, 0.05, 1, 0.95]
)


FIG1_PNG = (
    OUT_FIGURES
    / "Fig_historical_environmental_change_profiles_raw.png"
)

FIG1_PDF = (
    OUT_FIGURES
    / "Fig_historical_environmental_change_profiles_raw.pdf"
)


plt.savefig(
    FIG1_PNG,
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    FIG1_PDF,
    bbox_inches="tight"
)

plt.close()


print(f"Saved: {FIG1_PNG}")
print(f"Saved: {FIG1_PDF}")


# =============================================================================
# 14. FIGURE 2 — STANDARDIZED CROSS-SITE CHANGE PROFILE
# =============================================================================
#
# This figure compares relative cross-site magnitude.
#
# IMPORTANT:
# Each column is independently standardized across only three sites.
# Therefore this is descriptive visualization only.
#
# No averaging or summation is performed.
# =============================================================================

print("\nCREATING FIGURE 2 — STANDARDIZED CROSS-SITE PROFILE")
print("-" * 78)


heatmap_columns = [
    "sst_change_degC_z",
    "population_change_percent_z",
    "cropland_change_pp_z",
    "pasture_change_pp_z",
    "forest_natural_loss_pp_z",
    "urban_change_pp_z",
]


heatmap_labels = [
    "SST\nwarming",
    "Population\ngrowth",
    "Cropland\nexpansion",
    "Pasture\nexpansion",
    "Forest/natural\nloss",
    "Urban\nexpansion",
]


heat = (
    profile
    .set_index("site")
    .loc[SITES, heatmap_columns]
    .astype(float)
)


fig, ax = plt.subplots(
    figsize=(11, 5)
)


im = ax.imshow(
    heat.values,
    aspect="auto",
    vmin=-1.5,
    vmax=1.5
)


ax.set_xticks(
    np.arange(len(heatmap_labels))
)

ax.set_xticklabels(
    heatmap_labels,
    fontsize=10
)

ax.set_yticks(
    np.arange(len(SITES))
)

ax.set_yticklabels(
    SITES,
    fontsize=11
)


for i in range(len(SITES)):

    for j in range(len(heatmap_labels)):

        value = heat.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=10
        )


cbar = fig.colorbar(
    im,
    ax=ax,
    shrink=0.85
)

cbar.set_label(
    "Standardized cross-site change (z-score)",
    rotation=270,
    labelpad=18
)


ax.set_title(
    "Relative magnitude of historical environmental change",
    fontsize=15,
    fontweight="bold",
    pad=15
)


fig.text(
    0.5,
    0.01,
    (
        "Each variable is independently standardized across the three sites. "
        "Positive values indicate relatively greater warming, population "
        "growth, land-use expansion, or forest loss. This is not a "
        "composite environmental-pressure index."
    ),
    ha="center",
    fontsize=9
)


plt.tight_layout(
    rect=[0, 0.06, 1, 1]
)


FIG2_PNG = (
    OUT_FIGURES
    / "Fig_historical_environmental_change_profiles_standardized.png"
)

FIG2_PDF = (
    OUT_FIGURES
    / "Fig_historical_environmental_change_profiles_standardized.pdf"
)


plt.savefig(
    FIG2_PNG,
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    FIG2_PDF,
    bbox_inches="tight"
)

plt.close()


print(f"Saved: {FIG2_PNG}")
print(f"Saved: {FIG2_PDF}")


# =============================================================================
# 15. FIGURE 3 — SITE PROFILE / RADAR-LIKE POLAR PLOTS
# =============================================================================
#
# IMPORTANT:
# These plots show standardized variables individually.
# The polygon area has NO quantitative ecological meaning.
# =============================================================================

print("\nCREATING FIGURE 3 — SITE-CENTRED CHANGE PROFILES")
print("-" * 78)


labels = [
    "SST",
    "Population",
    "Cropland",
    "Pasture",
    "Forest loss",
    "Urban"
]


z_columns = [
    "sst_change_degC_z",
    "population_change_percent_z",
    "cropland_change_pp_z",
    "pasture_change_pp_z",
    "forest_natural_loss_pp_z",
    "urban_change_pp_z",
]


n_vars = len(labels)

angles = np.linspace(
    0,
    2 * np.pi,
    n_vars,
    endpoint=False
).tolist()

angles += angles[:1]


fig = plt.figure(
    figsize=(15, 5)
)


for panel, site in enumerate(
    SITES,
    start=1
):

    ax = fig.add_subplot(
        1,
        3,
        panel,
        polar=True
    )

    site_row = (
        profile
        .set_index("site")
        .loc[site]
    )

    values = [
        float(site_row[col])
        for col in z_columns
    ]

    values += values[:1]


    # Shift only for plotting radius.
    # Original z-score labels remain unchanged.
    plot_values = [
        value + 1.75
        for value in values
    ]


    ax.plot(
        angles,
        plot_values,
        linewidth=2
    )

    ax.fill(
        angles,
        plot_values,
        alpha=0.12
    )


    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        labels,
        fontsize=9
    )


    ax.set_ylim(
        0,
        3.5
    )


    ax.set_yticks(
        [0.75, 1.75, 2.75]
    )

    ax.set_yticklabels(
        ["-1", "0", "+1"],
        fontsize=8
    )


    ax.set_title(
        site,
        fontsize=14,
        fontweight="bold",
        pad=20
    )


fig.suptitle(
    "Site-specific historical environmental-change profiles",
    fontsize=16,
    fontweight="bold",
    y=1.03
)


fig.text(
    0.5,
    -0.04,
    (
        "Axes show independently standardized cross-site changes. "
        "Polygon size/area is descriptive only and must not be interpreted "
        "as a composite pressure score."
    ),
    ha="center",
    fontsize=9
)


plt.tight_layout()


FIG3_PNG = (
    OUT_FIGURES
    / "Fig_historical_environmental_change_profiles_by_site.png"
)

FIG3_PDF = (
    OUT_FIGURES
    / "Fig_historical_environmental_change_profiles_by_site.pdf"
)


plt.savefig(
    FIG3_PNG,
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    FIG3_PDF,
    bbox_inches="tight"
)

plt.close()


print(f"Saved: {FIG3_PNG}")
print(f"Saved: {FIG3_PDF}")


# =============================================================================
# 16. RANKING TABLE FOR INTERPRETATION
# =============================================================================

print("\n" + "=" * 78)
print("VARIABLE-SPECIFIC CROSS-SITE RANKINGS")
print("Rank 1 = greatest historical change / pressure")
print("=" * 78)


ranking_info = [
    (
        "SST warming",
        "sst_change_degC",
        "sst_change_degC_rank"
    ),

    (
        "Population growth",
        "population_change_percent",
        "population_change_percent_rank"
    ),

    (
        "Cropland expansion",
        "cropland_change_pp",
        "cropland_change_pp_rank"
    ),

    (
        "Pasture expansion",
        "pasture_change_pp",
        "pasture_change_pp_rank"
    ),

    (
        "Forest/natural loss",
        "forest_natural_loss_pp",
        "forest_natural_loss_pp_rank"
    ),

    (
        "Urban expansion",
        "urban_change_pp",
        "urban_change_pp_rank"
    ),
]


rank_rows = []


for label, value_col, rank_col in ranking_info:

    temp = (
        profile[
            ["site", value_col, rank_col]
        ]
        .sort_values(
            rank_col
        )
    )

    for _, row in temp.iterrows():

        rank_rows.append({
            "variable": label,
            "site": row["site"],
            "change_value": row[value_col],
            "rank": row[rank_col],
        })


rank_table = pd.DataFrame(rank_rows)


RANK_FILE = (
    OUT_TABLES
    / "historical_environmental_change_variable_ranks.csv"
)


rank_table.to_csv(
    RANK_FILE,
    index=False
)


print(
    rank_table.to_string(
        index=False
    )
)

print(f"\nSaved: {RANK_FILE}")


# =============================================================================
# 17. SCIENTIFIC INTERPRETATION SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("SCIENTIFIC INTERPRETATION GUIDE")
print("=" * 78)


for site in SITES:

    row = (
        profile
        .set_index("site")
        .loc[site]
    )

    print(f"\n{site}")
    print("-" * 40)

    print(
        f"SST change: "
        f"{row['sst_change_degC']:+.3f} °C"
    )

    print(
        f"Population change: "
        f"{row['population_change_percent']:+.1f}%"
    )

    print(
        f"Cropland change: "
        f"{row['cropland_change_pp']:+.2f} percentage points"
    )

    print(
        f"Pasture change: "
        f"{row['pasture_change_pp']:+.2f} percentage points"
    )

    print(
        f"Forest/natural vegetation loss: "
        f"{row['forest_natural_loss_pp']:+.2f} percentage points"
    )

    print(
        f"Urban change: "
        f"{row['urban_change_pp']:+.3f} percentage points"
    )


# =============================================================================
# 18. FINAL QC TABLE
# =============================================================================

qc_rows = []


for site in SITES:

    d = df[df["site"] == site]

    qc_rows.append({

        "site": site,

        "total_rows": len(d),

        "year_min":
            int(d["year"].min()),

        "year_max":
            int(d["year"].max()),

        "early_period_rows":
            int(
                d["year"]
                .between(
                    EARLY_START,
                    EARLY_END
                )
                .sum()
            ),

        "late_period_rows":
            int(
                d["year"]
                .between(
                    LATE_START,
                    LATE_END
                )
                .sum()
            ),

        "missing_sst":
            int(d["sst_degC"].isna().sum()),

        "missing_population":
            int(d["population_70km"].isna().sum()),

        "missing_cropland":
            int(
                d[
                    "cropland_percent_70km"
                ]
                .isna()
                .sum()
            ),

        "missing_pasture":
            int(
                d[
                    "pasture_percent_70km"
                ]
                .isna()
                .sum()
            ),

        "missing_forest":
            int(
                d[
                    "forest_natural_percent_70km"
                ]
                .isna()
                .sum()
            ),

        "missing_urban":
            int(
                d[
                    "urban_percent_70km"
                ]
                .isna()
                .sum()
            ),
    })


qc = pd.DataFrame(qc_rows)


QC_FILE = (
    OUT_TABLES
    / "historical_environmental_change_profiles_QC.csv"
)


qc.to_csv(
    QC_FILE,
    index=False
)


# =============================================================================
# 19. COMPLETE
# =============================================================================

print("\n" + "=" * 78)
print("SCRIPT 20 COMPLETE")
print("=" * 78)


print("\nMAIN CHANGE TABLE")
print("-" * 78)
print(CHANGE_FILE)


print("\nSITE PROFILE TABLE")
print("-" * 78)
print(PROFILE_FILE)


print("\nVARIABLE-SPECIFIC RANKINGS")
print("-" * 78)
print(RANK_FILE)


print("\nQUALITY CONTROL")
print("-" * 78)
print(QC_FILE)


print("\nFIGURES")
print("-" * 78)
print(FIG1_PNG)
print(FIG1_PDF)
print(FIG2_PNG)
print(FIG2_PDF)
print(FIG3_PNG)
print(FIG3_PDF)


print("\nIMPORTANT INTERPRETATION")
print("-" * 78)

print(
    "1. Historical change is defined primarily as the difference between "
    "1900–1919 and 1999–2018 period means."
)

print(
    "2. Population change is reported as percent change because absolute "
    "population differs strongly among sites."
)

print(
    "3. Cropland, pasture, forest/natural vegetation, and urban change are "
    "reported in percentage points of the full 70-km circular buffer."
)

print(
    "4. Forest/natural vegetation decline is sign-reversed only for the "
    "forest-loss descriptor so that larger positive values indicate "
    "greater loss."
)

print(
    "5. Cross-site z-scores are descriptive standardized values calculated "
    "separately for each environmental variable."
)

print(
    "6. No standardized variables are summed or averaged. "
    "This script does NOT create a composite environmental-pressure index."
)

print(
    "7. With only three sites, variable-specific ranks and standardized "
    "profiles are descriptive and should not be treated as inferential "
    "statistics."
)

print(
    "8. ERSSTv5 represents coarse regional reconstructed SST rather than "
    "a literal 70-km-buffer SST measurement."
)

print(
    "9. ISIMIP3a population and land-use data are historical "
    "reconstructions, not direct annual observations."
)

print(
    "10. Static ETOPO bathymetry and modern Allen Coral Atlas habitat "
    "metrics are deliberately excluded from these historical-change "
    "calculations."
)


print(f"\nLog saved: {LOG_FILE}")


log_handle.close()
sys.stdout = sys.__stdout__