
"""
12_allen_benthic_publication_figures.py
=======================================

Publication-ready figures for Allen Coral Atlas benthic habitat within 70 km
of Jolo, Masinloc, and Palapag.

INPUTS
------
D:\temporal_environmental_data\03_extracted\
    allen_coral_atlas_benthic_classes_70km.csv

D:\temporal_environmental_data\04_tables\
    allen_coral_atlas_benthic_70km_summary.csv

D:\temporal_environmental_data\01_raw\Allen_Coral_Atlas_WFS\
    Jolo_Allen_Coral_Atlas_benthic_70km.gpkg
    Masinloc_Allen_Coral_Atlas_benthic_70km.gpkg
    Palapag_Allen_Coral_Atlas_benthic_70km.gpkg

OUTPUTS
-------
1) Absolute benthic habitat area by class:
   Fig_Allen_benthic_area_70km.{png,pdf,svg}

2) Percent composition of mapped benthic habitat:
   Fig_Allen_benthic_composition_70km.{png,pdf,svg}

3) Three-panel benthic habitat maps:
   Fig_Allen_benthic_maps_70km.{png,pdf,svg}

4) Sand + rubble comparison:
   Fig_Allen_sand_rubble_70km.{png,pdf,svg}

IMPORTANT INTERPRETATION
------------------------
- Percentages are relative to Allen Coral Atlas MAPPED benthic area only.
- Unmapped marine area must not be interpreted as unsuitable habitat.
- These maps are modern satellite-derived benthic habitat snapshots.
- No explicit colors are assigned; Matplotlib defaults are used.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

ROOT = Path(r"D:\temporal_environmental_data")

LONG_FILE = (
    ROOT / "03_extracted"
    / "allen_coral_atlas_benthic_classes_70km.csv"
)

SUMMARY_FILE = (
    ROOT / "04_tables"
    / "allen_coral_atlas_benthic_70km_summary.csv"
)

GPKG_DIR = (
    ROOT / "01_raw"
    / "Allen_Coral_Atlas_WFS"
)

FIG_DIR = ROOT / "05_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SITE_ORDER = ["Jolo", "Masinloc", "Palapag"]

SITE_COORDS = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}

CLASS_ORDER = [
    "Sand",
    "Rubble",
    "Coral/Algae",
    "Seagrass",
    "Rock",
    "Microalgal Mats",
]

# ---------------------------------------------------------------------
# Load tables
# ---------------------------------------------------------------------

long_df = pd.read_csv(LONG_FILE)
summary = pd.read_csv(SUMMARY_FILE)

long_df["site"] = pd.Categorical(
    long_df["site"],
    categories=SITE_ORDER,
    ordered=True,
)

summary["site"] = pd.Categorical(
    summary["site"],
    categories=SITE_ORDER,
    ordered=True,
)

# Ensure all class combinations exist.
area_pivot = (
    long_df.pivot_table(
        index="site",
        columns="class_name",
        values="area_km2",
        aggfunc="sum",
        fill_value=0,
    )
    .reindex(index=SITE_ORDER, fill_value=0)
)

pct_pivot = (
    long_df.pivot_table(
        index="site",
        columns="class_name",
        values="percent_of_mapped_benthic_area",
        aggfunc="sum",
        fill_value=0,
    )
    .reindex(index=SITE_ORDER, fill_value=0)
)

for cls in CLASS_ORDER:
    if cls not in area_pivot.columns:
        area_pivot[cls] = 0.0
    if cls not in pct_pivot.columns:
        pct_pivot[cls] = 0.0

area_pivot = area_pivot[CLASS_ORDER]
pct_pivot = pct_pivot[CLASS_ORDER]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =====================================================================
# FIGURE 1 — ABSOLUTE BENTHIC HABITAT AREA
# =====================================================================

fig, ax = plt.subplots(figsize=(8.6, 5.4))

bottom = np.zeros(len(area_pivot))

for cls in CLASS_ORDER:
    vals = area_pivot[cls].to_numpy(dtype=float)

    ax.bar(
        SITE_ORDER,
        vals,
        bottom=bottom,
        label=cls,
    )

    bottom += vals

ax.set_ylabel("Mapped benthic habitat area (km²)")
ax.set_xlabel("Site")
ax.set_title(
    "Allen Coral Atlas benthic habitat within 70 km"
)

ax.grid(
    axis="y",
    alpha=0.20,
)

ax.legend(
    title="Benthic class",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

# Add total mapped benthic area labels.
for i, total in enumerate(bottom):
    ax.text(
        i,
        total + max(bottom) * 0.02,
        f"{total:,.0f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

ax.set_ylim(
    0,
    max(bottom) * 1.12,
)

fig.tight_layout()

base = FIG_DIR / "Fig_Allen_benthic_area_70km"

fig.savefig(
    base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".pdf"),
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# FIGURE 2 — PERCENT COMPOSITION OF MAPPED BENTHIC HABITAT
# =====================================================================

fig, ax = plt.subplots(figsize=(8.6, 5.4))

bottom = np.zeros(len(pct_pivot))

for cls in CLASS_ORDER:
    vals = pct_pivot[cls].to_numpy(dtype=float)

    ax.bar(
        SITE_ORDER,
        vals,
        bottom=bottom,
        label=cls,
    )

    bottom += vals

ax.set_ylabel("Composition of mapped benthic habitat (%)")
ax.set_xlabel("Site")
ax.set_title(
    "Relative composition of Allen Coral Atlas benthic habitat"
)

ax.set_ylim(0, 100)

ax.grid(
    axis="y",
    alpha=0.20,
)

ax.legend(
    title="Benthic class",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

fig.text(
    0.5,
    0.01,
    "Percentages are relative to Allen Coral Atlas mapped benthic area within 70 km, not total marine area.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.04, 1, 1]
)

base = FIG_DIR / "Fig_Allen_benthic_composition_70km"

fig.savefig(
    base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".pdf"),
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# FIGURE 3 — SAND + RUBBLE COMPARISON
# =====================================================================

sand_rubble = (
    summary.sort_values("site")
    [["site", "sand_km2", "rubble_km2", "sand_rubble_percent_mapped"]]
    .copy()
)

x = np.arange(len(sand_rubble))

fig, ax = plt.subplots(figsize=(7.2, 4.8))

bars1 = ax.bar(
    x,
    sand_rubble["sand_km2"],
    label="Sand",
)

bars2 = ax.bar(
    x,
    sand_rubble["rubble_km2"],
    bottom=sand_rubble["sand_km2"],
    label="Rubble",
)

totals = (
    sand_rubble["sand_km2"]
    + sand_rubble["rubble_km2"]
)

ax.set_xticks(
    x,
    sand_rubble["site"].astype(str),
)

ax.set_ylabel("Sand + rubble area (km²)")
ax.set_title(
    "Candidate sandy/rubble substrate within mapped benthic habitat"
)

ax.grid(
    axis="y",
    alpha=0.20,
)

ax.legend(
    frameon=False,
)

ax.set_ylim(
    0,
    max(totals) * 1.18,
)

for i, total in enumerate(totals):
    pct = sand_rubble.iloc[i]["sand_rubble_percent_mapped"]

    ax.text(
        i,
        total + max(totals) * 0.025,
        f"{total:.1f} km²\n({pct:.1f}% mapped)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

fig.tight_layout()

base = FIG_DIR / "Fig_Allen_sand_rubble_70km"

fig.savefig(
    base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".pdf"),
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# FIGURE 4 — THREE-PANEL SPATIAL MAP
# =====================================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(14, 5),
)

panel_labels = ["A", "B", "C"]

# One legend mapping class -> same plotting order.
legend_handles = None
legend_labels = None

for ax, site, panel in zip(
    axes,
    SITE_ORDER,
    panel_labels,
):

    gpkg = (
        GPKG_DIR
        / f"{site}_Allen_Coral_Atlas_benthic_70km.gpkg"
    )

    if not gpkg.exists():
        raise FileNotFoundError(
            f"Missing GeoPackage for {site}:\n{gpkg}"
        )

    gdf = gpd.read_file(
        gpkg,
        layer="benthic_70km",
    )

    # Dissolve by class for cleaner plotting.
    dissolved = (
        gdf[
            ["class_name", "geometry"]
        ]
        .dissolve(
            by="class_name"
        )
        .reset_index()
    )

    # Reorder classes.
    dissolved["class_name"] = pd.Categorical(
        dissolved["class_name"],
        categories=CLASS_ORDER,
        ordered=True,
    )

    dissolved = dissolved.sort_values(
        "class_name"
    )

    # Plot one class at a time so legend remains meaningful.
    handles = []
    labels = []

    for cls in CLASS_ORDER:

        subset = dissolved[
            dissolved["class_name"] == cls
        ]

        if subset.empty:
            continue

        collection = subset.plot(
            ax=ax,
            linewidth=0.15,
            edgecolor="none",
            alpha=0.85,
            label=cls,
        )

    lat, lon = SITE_COORDS[site]

    site_gdf = gpd.GeoDataFrame(
        {"site": [site]},
        geometry=[Point(lon, lat)],
        crs="EPSG:4326",
    )

    site_gdf.plot(
        ax=ax,
        marker="*",
        markersize=70,
        edgecolor="black",
        facecolor="white",
        linewidth=0.8,
        zorder=10,
    )

    ax.set_title(site)

    ax.text(
        0.02,
        0.98,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=11,
    )

    ax.set_xlabel("Longitude")

    if ax is axes[0]:
        ax.set_ylabel("Latitude")
    else:
        ax.set_ylabel("")

    ax.grid(
        alpha=0.15,
    )

# Build a figure legend from the first map's collections.
# GeoPandas does not always expose handles cleanly, so construct from patches.
from matplotlib.patches import Patch

legend_handles = [
    Patch(label=cls)
    for cls in CLASS_ORDER
]

fig.legend(
    handles=legend_handles,
    labels=CLASS_ORDER,
    title="Benthic class",
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    frameon=False,
)

fig.suptitle(
    "Allen Coral Atlas benthic habitat within 70 km of each site",
    y=1.03,
)

fig.text(
    0.5,
    0.02,
    "Modern satellite-derived benthic habitat map; unmapped areas are not equivalent to unsuitable habitat.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.05, 1, 0.91]
)

base = FIG_DIR / "Fig_Allen_benthic_maps_70km"

fig.savefig(
    base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".pdf"),
    bbox_inches="tight",
)
fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)


print("=" * 78)
print("ALLEN CORAL ATLAS PUBLICATION FIGURES COMPLETE")
print("=" * 78)

print("\nAbsolute benthic area:")
print(
    FIG_DIR / "Fig_Allen_benthic_area_70km.pdf"
)

print("\nPercent composition:")
print(
    FIG_DIR / "Fig_Allen_benthic_composition_70km.pdf"
)

print("\nSand + rubble:")
print(
    FIG_DIR / "Fig_Allen_sand_rubble_70km.pdf"
)

print("\nThree-panel benthic maps:")
print(
    FIG_DIR / "Fig_Allen_benthic_maps_70km.pdf"
)

print(
    "\nRecommendation:"
    "\n  Main quantitative figure: Fig_Allen_benthic_area_70km"
    "\n  Main spatial figure:      Fig_Allen_benthic_maps_70km"
    "\n  Supplementary figure:      Fig_Allen_benthic_composition_70km"
)
