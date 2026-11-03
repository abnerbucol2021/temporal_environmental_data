
"""
09_spawning_habitat_publication_figures.py
==========================================

Publication-ready visualization of the 70-km bathymetric spawning-habitat
capacity summary for Jolo, Masinloc, and Palapag.

INPUTS
------
D:\temporal_environmental_data\04_tables\
    spawning_habitat_capacity_bathymetry_70km_summary.csv

D:\temporal_environmental_data\04_tables\
    spawning_habitat_capacity_bathymetry_exploratory_index.csv

OUTPUTS
-------
Main multi-panel figure:
    Fig_PSHC_bathymetry_components_70km.pdf
    Fig_PSHC_bathymetry_components_70km.svg
    Fig_PSHC_bathymetry_components_70km.png

Standardized comparison heatmap:
    Fig_PSHC_bathymetry_standardized_heatmap_70km.pdf
    Fig_PSHC_bathymetry_standardized_heatmap_70km.svg
    Fig_PSHC_bathymetry_standardized_heatmap_70km.png

Exploratory index:
    Fig_PSHC_bathymetry_exploratory_index_70km.pdf
    Fig_PSHC_bathymetry_exploratory_index_70km.svg
    Fig_PSHC_bathymetry_exploratory_index_70km.png

Notes
-----
- The main figure uses raw, interpretable units and should be preferred for
  the manuscript.
- The heatmap shows z-scores only to compare relative differences among sites.
- The exploratory composite index should NOT be presented as a validated
  biological index unless the weighting/definition is justified separately.
- No explicit plotting colors are specified; Matplotlib defaults are used.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(r"D:\temporal_environmental_data")

SUMMARY_FILE = (
    ROOT / "04_tables"
    / "spawning_habitat_capacity_bathymetry_70km_summary.csv"
)

INDEX_FILE = (
    ROOT / "04_tables"
    / "spawning_habitat_capacity_bathymetry_exploratory_index.csv"
)

FIG_DIR = ROOT / "05_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SITE_ORDER = ["Jolo", "Masinloc", "Palapag"]

if not SUMMARY_FILE.exists():
    raise FileNotFoundError(
        f"Summary file not found:\n{SUMMARY_FILE}"
    )

summary = pd.read_csv(SUMMARY_FILE)

required = {
    "site",
    "shallow_area_0_10m_km2",
    "shallow_area_10_20m_km2",
    "shallow_area_20_30m_km2",
    "shallow_area_0_30m_km2",
    "shallow_0_30m_fraction_of_marine_area",
    "mean_slope_0_30m_deg",
    "shallow_patch_count_0_30m",
    "largest_shallow_patch_km2",
    "shallow_fragmentation_index",
}

missing = required - set(summary.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns:\n{sorted(missing)}"
    )

summary["site"] = pd.Categorical(
    summary["site"],
    categories=SITE_ORDER,
    ordered=True,
)

summary = (
    summary.sort_values("site")
    .reset_index(drop=True)
)

# --------------------------------------------------------------------------
# GENERAL FIGURE SETTINGS
# --------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

x = np.arange(len(summary))
site_labels = summary["site"].astype(str).tolist()


def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def annotate_bars(ax, bars, fmt="{:.0f}", offset_fraction=0.015):
    ymax = ax.get_ylim()[1]
    offset = ymax * offset_fraction

    for bar in bars:
        height = bar.get_height()

        if np.isfinite(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )


# ==========================================================================
# FIGURE 1 — MAIN FOUR-PANEL RAW-METRIC FIGURE
# ==========================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9.2, 7.2),
)

axA, axB, axC, axD = axes.ravel()

# --------------------------------------------------------------------------
# A. Shallow habitat area by depth band — stacked bars
# --------------------------------------------------------------------------

v_0_10 = summary["shallow_area_0_10m_km2"].to_numpy()
v_10_20 = summary["shallow_area_10_20m_km2"].to_numpy()
v_20_30 = summary["shallow_area_20_30m_km2"].to_numpy()

bars1 = axA.bar(
    x,
    v_0_10,
    label="0–10 m",
)

bars2 = axA.bar(
    x,
    v_10_20,
    bottom=v_0_10,
    label="10–20 m",
)

bars3 = axA.bar(
    x,
    v_20_30,
    bottom=v_0_10 + v_10_20,
    label="20–30 m",
)

axA.set_xticks(x, site_labels)
axA.set_ylabel("Shallow marine area (km²)")
axA.set_title("Shallow-water habitat by depth band")
axA.legend(
    title="Depth",
    frameon=False,
)
axA.grid(
    axis="y",
    alpha=0.20,
)

totals = (
    v_0_10
    + v_10_20
    + v_20_30
)

ylimA = max(totals) * 1.16
axA.set_ylim(0, ylimA)

for xpos, total in zip(x, totals):
    axA.text(
        xpos,
        total + ylimA * 0.018,
        f"{total:,.0f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

add_panel_label(axA, "A")


# --------------------------------------------------------------------------
# B. Proportion of marine area that is shallow 0–30 m
# --------------------------------------------------------------------------

fraction_percent = (
    summary[
        "shallow_0_30m_fraction_of_marine_area"
    ].to_numpy()
    * 100.0
)

barsB = axB.bar(
    x,
    fraction_percent,
)

axB.set_xticks(x, site_labels)
axB.set_ylabel("0–30 m habitat (% of marine area)")
axB.set_title("Relative availability of shallow habitat")
axB.grid(
    axis="y",
    alpha=0.20,
)

axB.set_ylim(
    0,
    max(fraction_percent) * 1.25,
)

annotate_bars(
    axB,
    barsB,
    fmt="{:.1f}%",
)

add_panel_label(axB, "B")


# --------------------------------------------------------------------------
# C. Largest connected shallow-water patch
# --------------------------------------------------------------------------

largest_patch = summary[
    "largest_shallow_patch_km2"
].to_numpy()

barsC = axC.bar(
    x,
    largest_patch,
)

axC.set_xticks(x, site_labels)
axC.set_ylabel("Largest 0–30 m patch (km²)")
axC.set_title("Largest connected shallow-water patch")
axC.grid(
    axis="y",
    alpha=0.20,
)

axC.set_ylim(
    0,
    max(largest_patch) * 1.20,
)

annotate_bars(
    axC,
    barsC,
    fmt="{:.0f}",
)

add_panel_label(axC, "C")


# --------------------------------------------------------------------------
# D. Number of distinct shallow-water patches
# --------------------------------------------------------------------------

patch_count = summary[
    "shallow_patch_count_0_30m"
].to_numpy()

barsD = axD.bar(
    x,
    patch_count,
)

axD.set_xticks(x, site_labels)
axD.set_ylabel("Number of 0–30 m patches")
axD.set_title("Shallow-habitat patchiness")
axD.grid(
    axis="y",
    alpha=0.20,
)

axD.set_ylim(
    0,
    max(patch_count) * 1.20,
)

annotate_bars(
    axD,
    barsD,
    fmt="{:.0f}",
)

add_panel_label(axD, "D")


fig.suptitle(
    "Bathymetric components of potential spawning-habitat capacity within 70 km",
    y=0.995,
)

fig.text(
    0.5,
    0.955,
    "NOAA ETOPO 2022, 15 arc-second bathymetry; static geomorphological proxy",
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93]
)

main_base = (
    FIG_DIR
    / "Fig_PSHC_bathymetry_components_70km"
)

fig.savefig(
    main_base.with_suffix(".pdf"),
    bbox_inches="tight",
)

fig.savefig(
    main_base.with_suffix(".svg"),
    bbox_inches="tight",
)

fig.savefig(
    main_base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ==========================================================================
# FIGURE 2 — STANDARDIZED HEATMAP
# ==========================================================================

heat_metrics = {
    "0–30 m area": "shallow_area_0_30m_km2",
    "10–20 m area": "shallow_area_10_20m_km2",
    "Shallow fraction": "shallow_0_30m_fraction_of_marine_area",
    "Largest patch": "largest_shallow_patch_km2",
    "Patch count": "shallow_patch_count_0_30m",
    "Shallow slope": "mean_slope_0_30m_deg",
}

raw = summary[
    list(heat_metrics.values())
].astype(float)

z = (
    raw
    - raw.mean(axis=0)
) / raw.std(
    axis=0,
    ddof=0,
)

z.columns = list(
    heat_metrics.keys()
)

fig, ax = plt.subplots(
    figsize=(8.7, 3.8)
)

im = ax.imshow(
    z.to_numpy(),
    aspect="auto",
)

ax.set_xticks(
    np.arange(len(z.columns)),
    labels=z.columns,
)

ax.set_yticks(
    np.arange(len(site_labels)),
    labels=site_labels,
)

plt.setp(
    ax.get_xticklabels(),
    rotation=35,
    ha="right",
    rotation_mode="anchor",
)

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        value = z.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

cbar = fig.colorbar(
    im,
    ax=ax,
    shrink=0.82,
)

cbar.set_label(
    "Standardized value (z-score)"
)

ax.set_title(
    "Standardized bathymetric spawning-habitat attributes within 70 km"
)

fig.text(
    0.5,
    0.01,
    "Positive values indicate values above the three-site mean; "
    "standardization is descriptive and does not imply biological weighting.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.055, 1, 1]
)

heat_base = (
    FIG_DIR
    / "Fig_PSHC_bathymetry_standardized_heatmap_70km"
)

fig.savefig(
    heat_base.with_suffix(".pdf"),
    bbox_inches="tight",
)

fig.savefig(
    heat_base.with_suffix(".svg"),
    bbox_inches="tight",
)

fig.savefig(
    heat_base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)

plt.close(fig)


# ==========================================================================
# FIGURE 3 — EXPLORATORY INDEX
# ==========================================================================

if INDEX_FILE.exists():

    idx = pd.read_csv(
        INDEX_FILE
    )

    if (
        "site" in idx.columns
        and
        "bathymetric_habitat_capacity_index_exploratory"
        in idx.columns
    ):

        idx["site"] = pd.Categorical(
            idx["site"],
            categories=SITE_ORDER,
            ordered=True,
        )

        idx = (
            idx.sort_values("site")
            .reset_index(drop=True)
        )

        vals = idx[
            "bathymetric_habitat_capacity_index_exploratory"
        ].to_numpy()

        fig, ax = plt.subplots(
            figsize=(6.6, 4.2)
        )

        bars = ax.bar(
            np.arange(len(idx)),
            vals,
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.set_xticks(
            np.arange(len(idx)),
            idx["site"].astype(str),
        )

        ax.set_ylabel(
            "Exploratory standardized index"
        )

        ax.set_title(
            "Exploratory bathymetric habitat-capacity index"
        )

        ax.grid(
            axis="y",
            alpha=0.20,
        )

        offset = (
            max(abs(vals))
            * 0.04
            if max(abs(vals)) > 0
            else 0.05
        )

        for bar, value in zip(
            bars,
            vals,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value
                + (
                    offset
                    if value >= 0
                    else -offset
                ),
                f"{value:+.2f}",
                ha="center",
                va=(
                    "bottom"
                    if value >= 0
                    else "top"
                ),
                fontsize=8,
            )

        fig.text(
            0.5,
            0.01,
            "Exploratory only: equal-weight standardized composite; "
            "do not interpret as a validated spawning-habitat index.",
            ha="center",
            fontsize=8,
        )

        fig.tight_layout(
            rect=[0, 0.05, 1, 1]
        )

        idx_base = (
            FIG_DIR
            / "Fig_PSHC_bathymetry_exploratory_index_70km"
        )

        fig.savefig(
            idx_base.with_suffix(".pdf"),
            bbox_inches="tight",
        )

        fig.savefig(
            idx_base.with_suffix(".svg"),
            bbox_inches="tight",
        )

        fig.savefig(
            idx_base.with_suffix(".png"),
            dpi=400,
            bbox_inches="tight",
        )

        plt.close(fig)

print("=" * 78)
print("PUBLICATION FIGURES COMPLETE")
print("=" * 78)

print("\nMain multi-panel:")
print(
    FIG_DIR
    / "Fig_PSHC_bathymetry_components_70km.pdf"
)

print("\nStandardized heatmap:")
print(
    FIG_DIR
    / "Fig_PSHC_bathymetry_standardized_heatmap_70km.pdf"
)

if INDEX_FILE.exists():
    print("\nExploratory index:")
    print(
        FIG_DIR
        / "Fig_PSHC_bathymetry_exploratory_index_70km.pdf"
    )

print(
    "\nRecommendation: use the multi-panel raw-metric figure "
    "as the manuscript figure; use the heatmap as a compact "
    "comparative or supplementary figure."
)
