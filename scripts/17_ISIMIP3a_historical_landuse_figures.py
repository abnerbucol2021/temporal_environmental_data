"""
SCRIPT 17
PUBLICATION FIGURES — ISIMIP3a HISTORICAL LAND USE
1900–2018

Project:
    D:\\temporal_environmental_data

Input:
    03_extracted\\landuse_ISIMIP3a_buffers_1900_2018.csv
    04_tables\\landuse_ISIMIP3a_change_summary.csv

Outputs:
    05_figures\\Fig_ISIMIP3a_landuse_70km_1900_2018.png
    05_figures\\Fig_ISIMIP3a_landuse_70km_1900_2018.pdf

    05_figures\\Fig_ISIMIP3a_landuse_change_1900_2018.png
    05_figures\\Fig_ISIMIP3a_landuse_change_1900_2018.pdf

    05_figures\\Fig_ISIMIP3a_landuse_buffer_sensitivity.png
    05_figures\\Fig_ISIMIP3a_landuse_buffer_sensitivity.pdf

Evidence classification:
    historical_landuse_reconstruction

IMPORTANT:
    These are reconstructed/harmonized historical land-use fractions.
    They are NOT direct annual satellite observations.

Spatial interpretation:
    70 km = primary analysis.
    25 and 50 km = spatial sensitivity analyses.

Percentages refer to the FULL circular buffer, including marine area.
They are therefore coastal-landscape pressure metrics, not percentages
of terrestrial land area.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(r"D:\temporal_environmental_data")

INPUT_ANNUAL = (
    ROOT
    / "03_extracted"
    / "landuse_ISIMIP3a_buffers_1900_2018.csv"
)

INPUT_CHANGE = (
    ROOT
    / "04_tables"
    / "landuse_ISIMIP3a_change_summary.csv"
)

FIGDIR = ROOT / "05_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# OUTPUT FILES
# ======================================================================

FIG1_PNG = FIGDIR / "Fig_ISIMIP3a_landuse_70km_1900_2018.png"
FIG1_PDF = FIGDIR / "Fig_ISIMIP3a_landuse_70km_1900_2018.pdf"

FIG2_PNG = FIGDIR / "Fig_ISIMIP3a_landuse_change_1900_2018.png"
FIG2_PDF = FIGDIR / "Fig_ISIMIP3a_landuse_change_1900_2018.pdf"

FIG3_PNG = FIGDIR / "Fig_ISIMIP3a_landuse_buffer_sensitivity.png"
FIG3_PDF = FIGDIR / "Fig_ISIMIP3a_landuse_buffer_sensitivity.pdf"


# ======================================================================
# BASIC PLOT SETTINGS
# ======================================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ======================================================================
# LOAD DATA
# ======================================================================

print("=" * 78)
print("SCRIPT 17 — ISIMIP3a HISTORICAL LAND-USE FIGURES")
print("=" * 78)


if not INPUT_ANNUAL.exists():
    sys.exit(
        "\nERROR: Annual land-use file not found:\n"
        f"{INPUT_ANNUAL}\n\n"
        "Run Script 16 first."
    )


annual = pd.read_csv(INPUT_ANNUAL)

print()
print(f"Loaded: {INPUT_ANNUAL.name}")
print(f"Rows: {len(annual):,}")
print(f"Columns: {len(annual.columns):,}")


if INPUT_CHANGE.exists():
    change = pd.read_csv(INPUT_CHANGE)

    print(
        f"Loaded: {INPUT_CHANGE.name}"
    )

else:
    change = None

    print(
        "Change-summary table not found. "
        "Figure 2 will calculate changes directly."
    )


# ======================================================================
# REQUIRED COLUMNS
# ======================================================================

required = [
    "site",
    "year",
    "buffer_km",
    "cropland_total_percent_of_buffer",
    "pastures_percent_of_buffer",
    "forests_natural_percent_of_buffer",
    "urban_percent_of_buffer",
    "agricultural_percent_of_buffer",
]


missing = [
    column
    for column in required
    if column not in annual.columns
]


if missing:
    sys.exit(
        "\nERROR: Missing required columns:\n"
        + "\n".join(missing)
    )


# ======================================================================
# QC
# ======================================================================

annual["year"] = pd.to_numeric(
    annual["year"],
    errors="raise"
).astype(int)

annual["buffer_km"] = pd.to_numeric(
    annual["buffer_km"],
    errors="raise"
).astype(int)


sites = [
    site
    for site in ["Jolo", "Masinloc", "Palapag"]
    if site in annual["site"].unique()
]


if len(sites) != 3:
    sys.exit(
        "\nERROR: Expected Jolo, Masinloc, and Palapag.\n"
        f"Found: {sorted(annual['site'].unique())}"
    )


expected_years = np.arange(1900, 2019)


for site in sites:

    for buffer_km in [25, 50, 70]:

        subset = annual[
            (annual["site"] == site)
            &
            (annual["buffer_km"] == buffer_km)
        ]

        years = np.sort(
            subset["year"].unique()
        )

        if not np.array_equal(
            years,
            expected_years
        ):
            sys.exit(
                "\nERROR: Year QC failed for "
                f"{site}, {buffer_km} km."
            )


print()
print("QC PASSED")
print("-" * 78)
print("Sites: Jolo, Masinloc, Palapag")
print("Years: 1900–2018")
print("Annual steps/site/buffer: 119")
print("Buffers: 25, 50, 70 km")


# ======================================================================
# LABELS
# ======================================================================

VARIABLES = {
    "cropland_total_percent_of_buffer": "Cropland",
    "pastures_percent_of_buffer": "Pasture",
    "forests_natural_percent_of_buffer": "Forest + natural vegetation",
    "urban_percent_of_buffer": "Urban area",
}


# ======================================================================
# HELPER
# ======================================================================

def save_figure(fig, png_path, pdf_path):

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=600
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {png_path.name}")
    print(f"Saved: {pdf_path.name}")


# ======================================================================
# FIGURE 1
# PRIMARY 70-KM HISTORICAL TRAJECTORIES
# ======================================================================

print()
print("Creating Figure 1: 70-km historical trajectories...")


primary = annual[
    annual["buffer_km"] == 70
].copy()


# Determine common y-axis for the first three variables.
# Urban is much smaller and gets its own sensible range.

major_variables = [
    "cropland_total_percent_of_buffer",
    "pastures_percent_of_buffer",
    "forests_natural_percent_of_buffer",
]

major_max = primary[
    major_variables
].max().max()

major_ylim = np.ceil(
    major_max / 5
) * 5

major_ylim = max(
    major_ylim,
    10
)


urban_max = primary[
    "urban_percent_of_buffer"
].max()

urban_ylim = np.ceil(
    urban_max * 10
) / 10

urban_ylim = max(
    urban_ylim,
    1.0
)


fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(8.0, 10.0),
    sharex=True
)


for ax, (
    variable,
    label
) in zip(
    axes,
    VARIABLES.items()
):

    for site in sites:

        subset = primary[
            primary["site"] == site
        ].sort_values("year")

        ax.plot(
            subset["year"],
            subset[variable],
            linewidth=2.0,
            label=site
        )


    ax.set_ylabel(
        "% of 70-km buffer"
    )

    ax.set_title(
        label,
        loc="left",
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.6
    )


    if variable == "urban_percent_of_buffer":

        ax.set_ylim(
            0,
            urban_ylim
        )

    else:

        ax.set_ylim(
            0,
            major_ylim
        )


    ax.set_xlim(
        1900,
        2018
    )


    ax.xaxis.set_major_locator(
        MultipleLocator(20)
    )


axes[-1].set_xlabel(
    "Year"
)


# One common legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False
)


fig.suptitle(
    "Historical land-use reconstruction within 70 km of study sites",
    fontsize=13,
    fontweight="bold",
    y=1.025
)


fig.text(
    0.5,
    0.002,
    (
        "ISIMIP3a histsoc historical land-use reconstruction. "
        "Percentages are relative to the full circular buffer, "
        "including marine area."
    ),
    ha="center",
    va="bottom",
    fontsize=8
)


fig.tight_layout(
    rect=[0, 0.025, 1, 0.97]
)


save_figure(
    fig,
    FIG1_PNG,
    FIG1_PDF
)


# ======================================================================
# FIGURE 2
# 1900–2018 CHANGE AT 70 KM
# ======================================================================

print()
print("Creating Figure 2: 1900–2018 change summary...")


change_variables = [
    "cropland_total_percent_of_buffer",
    "pastures_percent_of_buffer",
    "forests_natural_percent_of_buffer",
    "urban_percent_of_buffer",
]


change_labels = [
    "Cropland",
    "Pasture",
    "Forest + natural\nvegetation",
    "Urban area",
]


change_rows = []


for site in sites:

    subset = primary[
        primary["site"] == site
    ]

    row = {
        "site": site
    }


    for variable in change_variables:

        v1900 = float(
            subset.loc[
                subset["year"] == 1900,
                variable
            ].iloc[0]
        )

        v2018 = float(
            subset.loc[
                subset["year"] == 2018,
                variable
            ].iloc[0]
        )

        row[variable] = (
            v2018 - v1900
        )


    change_rows.append(row)


change_plot = pd.DataFrame(
    change_rows
)


fig, ax = plt.subplots(
    figsize=(9.0, 5.5)
)


x = np.arange(
    len(change_variables)
)

width = 0.24


for i, site in enumerate(sites):

    values = [
        change_plot.loc[
            change_plot["site"] == site,
            variable
        ].iloc[0]
        for variable in change_variables
    ]

    offset = (
        i - 1
    ) * width

    bars = ax.bar(
        x + offset,
        values,
        width=width,
        label=site
    )


    # Add value labels
    for bar, value in zip(
        bars,
        values
    ):

        if value >= 0:

            va = "bottom"
            y = value + 0.35

        else:

            va = "top"
            y = value - 0.35


        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            y,
            f"{value:+.1f}",
            ha="center",
            va=va,
            fontsize=8
        )


ax.axhline(
    0,
    linewidth=0.8
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    change_labels
)


ax.set_ylabel(
    "Change from 1900 to 2018\n(percentage points of 70-km buffer)"
)


ax.set_title(
    "Historical land-use change within 70 km of study sites",
    fontweight="bold"
)


ax.legend(
    frameon=False,
    ncol=3
)


ax.grid(
    axis="y",
    alpha=0.25,
    linewidth=0.6
)


fig.text(
    0.5,
    0.005,
    (
        "Positive values indicate expansion; "
        "negative values indicate decline. "
        "ISIMIP3a histsoc reconstruction."
    ),
    ha="center",
    fontsize=8
)


fig.tight_layout(
    rect=[0, 0.035, 1, 1]
)


save_figure(
    fig,
    FIG2_PNG,
    FIG2_PDF
)


# ======================================================================
# FIGURE 3
# BUFFER-SENSITIVITY ANALYSIS
#
# Use cropland and forest/natural vegetation because these are the
# strongest terrestrial-transformation indicators in Script 16.
# ======================================================================

print()
print("Creating Figure 3: 25/50/70-km sensitivity...")


sensitivity_variables = {
    "cropland_total_percent_of_buffer": "Cropland",
    "forests_natural_percent_of_buffer": "Forest + natural vegetation",
}


fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(11.0, 9.0),
    sharex=True
)


for row_index, site in enumerate(sites):

    for col_index, (
        variable,
        variable_label
    ) in enumerate(
        sensitivity_variables.items()
    ):

        ax = axes[
            row_index,
            col_index
        ]


        for buffer_km in [
            25,
            50,
            70
        ]:

            subset = annual[
                (annual["site"] == site)
                &
                (
                    annual["buffer_km"]
                    ==
                    buffer_km
                )
            ].sort_values(
                "year"
            )


            ax.plot(
                subset["year"],
                subset[variable],
                linewidth=1.8,
                label=f"{buffer_km} km"
            )


        ax.grid(
            axis="y",
            alpha=0.25,
            linewidth=0.6
        )


        ax.set_xlim(
            1900,
            2018
        )


        ax.xaxis.set_major_locator(
            MultipleLocator(20)
        )


        if row_index == 0:

            ax.set_title(
                variable_label,
                fontweight="bold"
            )


        if col_index == 0:

            ax.set_ylabel(
                f"{site}\n% of buffer"
            )

        else:

            ax.set_ylabel(
                "% of buffer"
            )


        if row_index == 2:

            ax.set_xlabel(
                "Year"
            )


handles, labels = axes[
    0,
    0
].get_legend_handles_labels()


fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False
)


fig.suptitle(
    "Spatial sensitivity of reconstructed historical land-use trajectories",
    fontsize=13,
    fontweight="bold",
    y=1.025
)


fig.text(
    0.5,
    0.002,
    (
        "25- and 50-km buffers are sensitivity analyses; "
        "70 km is the primary scale. "
        "Source resolution is approximately 0.5°."
    ),
    ha="center",
    fontsize=8
)


fig.tight_layout(
    rect=[0, 0.025, 1, 0.96]
)


save_figure(
    fig,
    FIG3_PNG,
    FIG3_PDF
)


# ======================================================================
# TERMINAL SUMMARY
# ======================================================================

print()
print("=" * 78)
print("70-KM LAND-USE CHANGE — 1900 TO 2018")
print("=" * 78)


for site in sites:

    print()
    print(site)
    print("-" * 40)

    row = change_plot[
        change_plot["site"] == site
    ].iloc[0]

    print(
        "Cropland: "
        f"{row['cropland_total_percent_of_buffer']:+.3f} pp"
    )

    print(
        "Pasture: "
        f"{row['pastures_percent_of_buffer']:+.3f} pp"
    )

    print(
        "Forest/natural: "
        f"{row['forests_natural_percent_of_buffer']:+.3f} pp"
    )

    print(
        "Urban: "
        f"{row['urban_percent_of_buffer']:+.3f} pp"
    )


# ======================================================================
# ADDITIONAL EARLY/LATE PERIOD COMPARISON
# ======================================================================

print()
print("=" * 78)
print("20-YEAR PERIOD COMPARISON — 70 KM")
print("1900–1919 versus 1999–2018")
print("=" * 78)


for site in sites:

    subset = primary[
        primary["site"] == site
    ]

    early = subset[
        subset["year"].between(
            1900,
            1919
        )
    ]

    late = subset[
        subset["year"].between(
            1999,
            2018
        )
    ]

    print()
    print(site)
    print("-" * 40)

    for variable, label in VARIABLES.items():

        early_mean = early[
            variable
        ].mean()

        late_mean = late[
            variable
        ].mean()

        difference = (
            late_mean
            -
            early_mean
        )

        print(
            f"{label}: "
            f"{early_mean:.3f}% -> "
            f"{late_mean:.3f}% "
            f"({difference:+.3f} pp)"
        )


# ======================================================================
# FINAL MESSAGE
# ======================================================================

print()
print("=" * 78)
print("SCRIPT 17 COMPLETE")
print("=" * 78)

print()
print("FIGURE 1")
print("Primary 70-km historical trajectories:")
print(FIG1_PNG)
print(FIG1_PDF)

print()
print("FIGURE 2")
print("1900–2018 change summary:")
print(FIG2_PNG)
print(FIG2_PDF)

print()
print("FIGURE 3")
print("25/50/70-km spatial sensitivity:")
print(FIG3_PNG)
print(FIG3_PDF)

print()
print("INTERPRETATION RULES")
print("-" * 78)

print(
    "1. ISIMIP3a land use is a historical reconstruction, "
    "not direct annual observation."
)

print(
    "2. Percentages refer to the full circular coastal buffer, "
    "including marine area."
)

print(
    "3. Use 70 km as the primary scale."
)

print(
    "4. Treat 25 and 50 km as spatial sensitivity analyses "
    "because the source grid is approximately 0.5 degrees."
)

print(
    "5. Do not interpret these variables as direct measurements "
    "of sedimentation or coastal water quality."
)