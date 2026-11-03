
"""
13_benthic_patch_configuration_and_maps.py
==========================================

Refine Allen Coral Atlas benthic-habitat analysis for Jolo, Masinloc,
and Palapag within the previously clipped 70-km buffers.

MAIN GOALS
----------
1. Fix the three-panel habitat map so benthic classes use consistent,
   distinct colors across all sites.
2. Quantify benthic habitat configuration rather than only total area.
3. Quantify sand+rubble spatial structure as a candidate spawning-substrate
   proxy for Spratelloides gracilis.
4. Quantify benthic habitat diversity and dominance.

INPUTS
------
D:\temporal_environmental_data\01_raw\Allen_Coral_Atlas_WFS\
    Jolo_Allen_Coral_Atlas_benthic_70km.gpkg
    Masinloc_Allen_Coral_Atlas_benthic_70km.gpkg
    Palapag_Allen_Coral_Atlas_benthic_70km.gpkg

D:\temporal_environmental_data\04_tables\
    allen_coral_atlas_benthic_70km_summary.csv

OUTPUT TABLES
-------------
04_tables/benthic_patch_metrics_70km.csv
04_tables/benthic_diversity_70km.csv
04_tables/sand_rubble_configuration_70km.csv
04_tables/benthic_configuration_site_summary_70km.csv

OUTPUT FIGURES
--------------
05_figures/Fig_Allen_benthic_maps_70km_FIXED.{png,pdf,svg}
05_figures/Fig_Allen_sand_rubble_configuration_70km.{png,pdf,svg}
05_figures/Fig_Allen_benthic_diversity_70km.{png,pdf,svg}

INTERPRETATION
--------------
- Allen Coral Atlas is a modern satellite-derived benthic habitat snapshot.
- Metrics describe current mapped habitat configuration, not conditions in 1900.
- Unmapped marine area is not equivalent to unsuitable habitat.
- Sand+rubble is an exploratory spawning-substrate proxy, not a validated
  species-specific habitat-suitability model.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Point
from shapely.ops import unary_union

ROOT = Path(r"D:\temporal_environmental_data")

GPKG_DIR = ROOT / "01_raw" / "Allen_Coral_Atlas_WFS"
TABLE_DIR = ROOT / "04_tables"
FIG_DIR = ROOT / "05_figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = (
    TABLE_DIR
    / "allen_coral_atlas_benthic_70km_summary.csv"
)

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

BUFFER_KM = 70


def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def safe_cv(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    mean = values.mean()

    if mean == 0:
        return np.nan

    return float(values.std(ddof=1) / mean)


def shannon_index(proportions):
    p = np.asarray(proportions, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]

    if len(p) == 0:
        return np.nan

    return float(-np.sum(p * np.log(p)))


def simpson_diversity(proportions):
    p = np.asarray(proportions, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]

    if len(p) == 0:
        return np.nan

    return float(1.0 - np.sum(p**2))


def geometry_parts(geom):
    """
    Return polygon parts from Polygon/MultiPolygon/GeometryCollection.
    """
    if geom is None or geom.is_empty:
        return []

    gtype = geom.geom_type

    if gtype == "Polygon":
        return [geom]

    if gtype == "MultiPolygon":
        return list(geom.geoms)

    if gtype == "GeometryCollection":
        parts = []
        for g in geom.geoms:
            parts.extend(geometry_parts(g))
        return parts

    return []


def patch_metrics_from_geometry(
    union_geom,
    site_point_utm,
    total_mapped_area_km2,
):
    """
    Calculate connected-patch metrics from a dissolved habitat geometry.
    """
    parts = geometry_parts(union_geom)

    if not parts:
        return {
            "patch_count": 0,
            "total_area_km2": 0.0,
            "mean_patch_area_km2": 0.0,
            "median_patch_area_km2": 0.0,
            "largest_patch_area_km2": 0.0,
            "largest_patch_fraction": np.nan,
            "patch_area_cv": np.nan,
            "patch_density_per_100km2_mapped": np.nan,
            "nearest_patch_distance_km": np.nan,
            "area_weighted_mean_patch_distance_km": np.nan,
        }

    areas = np.array(
        [g.area / 1_000_000.0 for g in parts],
        dtype=float,
    )

    distances = np.array(
        [site_point_utm.distance(g) / 1000.0 for g in parts],
        dtype=float,
    )

    total_area = float(areas.sum())
    largest = float(areas.max())

    if total_area > 0:
        area_weighted_dist = float(
            np.average(
                distances,
                weights=areas,
            )
        )
        largest_fraction = largest / total_area
    else:
        area_weighted_dist = np.nan
        largest_fraction = np.nan

    patch_density = (
        len(parts)
        / total_mapped_area_km2
        * 100.0
        if total_mapped_area_km2 > 0
        else np.nan
    )

    return {
        "patch_count": int(len(parts)),
        "total_area_km2": total_area,
        "mean_patch_area_km2": float(areas.mean()),
        "median_patch_area_km2": float(np.median(areas)),
        "largest_patch_area_km2": largest,
        "largest_patch_fraction": largest_fraction,
        "patch_area_cv": safe_cv(areas),
        "patch_density_per_100km2_mapped": patch_density,
        "nearest_patch_distance_km": float(distances.min()),
        "area_weighted_mean_patch_distance_km":
            area_weighted_dist,
    }


print("=" * 78)
print("BENTHIC PATCH CONFIGURATION + FIXED MAP ANALYSIS")
print("=" * 78)

if not SUMMARY_FILE.exists():
    raise FileNotFoundError(
        f"Required summary file not found:\n{SUMMARY_FILE}"
    )

summary_existing = pd.read_csv(
    SUMMARY_FILE
)

patch_rows = []
diversity_rows = []
sand_rubble_rows = []
site_summary_rows = []

site_gdfs_wgs84 = {}

# -------------------------------------------------------------------------
# ANALYSIS BY SITE
# -------------------------------------------------------------------------

for site in SITE_ORDER:

    lat, lon = SITE_COORDS[site]

    gpkg = (
        GPKG_DIR
        / f"{site}_Allen_Coral_Atlas_benthic_70km.gpkg"
    )

    if not gpkg.exists():
        raise FileNotFoundError(
            f"Missing GeoPackage:\n{gpkg}"
        )

    print(f"\n{site}")

    gdf = gpd.read_file(
        gpkg,
        layer="benthic_70km",
    )

    if gdf.crs is None:
        gdf = gdf.set_crs(
            "EPSG:4326"
        )

    # Geometry repair.
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf["class_name"].notna()
    ].copy()

    site_gdfs_wgs84[site] = gdf.to_crs(
        "EPSG:4326"
    )

    epsg = utm_epsg(
        lat,
        lon,
    )

    gdf_utm = gdf.to_crs(
        epsg=epsg
    )

    site_point_utm = (
        gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326",
        )
        .to_crs(
            epsg=epsg
        )
        .iloc[0]
    )

    total_mapped_area_km2 = float(
        gdf_utm.geometry.area.sum()
        / 1_000_000.0
    )

    # ---------------------------------------------------------------------
    # Habitat composition + diversity
    # ---------------------------------------------------------------------

    class_areas = (
        gdf_utm
        .assign(
            area_km2=
                gdf_utm.geometry.area
                / 1_000_000.0
        )
        .groupby(
            "class_name"
        )[
            "area_km2"
        ]
        .sum()
    )

    class_areas = class_areas.reindex(
        CLASS_ORDER,
        fill_value=0.0,
    )

    proportions = (
        class_areas
        / class_areas.sum()
        if class_areas.sum() > 0
        else class_areas * np.nan
    )

    richness = int(
        np.sum(
            class_areas.to_numpy() > 0
        )
    )

    H = shannon_index(
        proportions.to_numpy()
    )

    evenness = (
        H / math.log(richness)
        if richness > 1
        else np.nan
    )

    simpson = simpson_diversity(
        proportions.to_numpy()
    )

    dominant_class = (
        class_areas.idxmax()
        if class_areas.sum() > 0
        else None
    )

    dominant_fraction = (
        float(
            proportions.max()
        )
        if class_areas.sum() > 0
        else np.nan
    )

    diversity_rows.append(
        {
            "site": site,
            "buffer_km": BUFFER_KM,
            "mapped_benthic_area_km2":
                total_mapped_area_km2,
            "class_richness": richness,
            "shannon_diversity_H": H,
            "shannon_evenness_J": evenness,
            "simpson_diversity_1_minus_D":
                simpson,
            "dominant_class":
                dominant_class,
            "dominant_class_fraction":
                dominant_fraction,
            "dominant_class_percent":
                dominant_fraction * 100
                if np.isfinite(dominant_fraction)
                else np.nan,
            "data_class":
                "modern_satellite_derived_benthic_habitat_map",
        }
    )

    print(
        f"  mapped area = {total_mapped_area_km2:.2f} km²"
    )

    print(
        f"  Shannon H = {H:.3f}; "
        f"evenness = {evenness:.3f}; "
        f"Simpson = {simpson:.3f}"
    )

    print(
        f"  dominant class = {dominant_class} "
        f"({dominant_fraction*100:.1f}%)"
    )

    # ---------------------------------------------------------------------
    # Patch metrics by class
    # ---------------------------------------------------------------------

    for cls in CLASS_ORDER:

        subset = gdf_utm[
            gdf_utm["class_name"] == cls
        ]

        if subset.empty:
            union_geom = None
        else:
            union_geom = unary_union(
                list(
                    subset.geometry
                )
            )

        metrics = patch_metrics_from_geometry(
            union_geom,
            site_point_utm,
            total_mapped_area_km2,
        )

        patch_rows.append(
            {
                "site": site,
                "buffer_km": BUFFER_KM,
                "class_name": cls,
                **metrics,
                "data_class":
                    "modern_satellite_derived_benthic_habitat_map",
            }
        )

    # ---------------------------------------------------------------------
    # Sand + rubble combined configuration
    # ---------------------------------------------------------------------

    sr_subset = gdf_utm[
        gdf_utm[
            "class_name"
        ].isin(
            [
                "Sand",
                "Rubble",
            ]
        )
    ]

    sr_union = (
        unary_union(
            list(
                sr_subset.geometry
            )
        )
        if not sr_subset.empty
        else None
    )

    sr_metrics = patch_metrics_from_geometry(
        sr_union,
        site_point_utm,
        total_mapped_area_km2,
    )

    sand_area = float(
        class_areas.get(
            "Sand",
            0.0,
        )
    )

    rubble_area = float(
        class_areas.get(
            "Rubble",
            0.0,
        )
    )

    sand_rubble_area = (
        sand_area
        + rubble_area
    )

    sr_percent_mapped = (
        sand_rubble_area
        / total_mapped_area_km2
        * 100.0
        if total_mapped_area_km2 > 0
        else np.nan
    )

    sand_rubble_rows.append(
        {
            "site": site,
            "buffer_km": BUFFER_KM,
            "sand_area_km2": sand_area,
            "rubble_area_km2": rubble_area,
            "sand_rubble_area_km2":
                sand_rubble_area,
            "sand_rubble_percent_mapped":
                sr_percent_mapped,
            **{
                f"sand_rubble_{k}": v
                for k, v in sr_metrics.items()
            },
            "proxy_status":
                "exploratory_spawning_substrate_proxy",
        }
    )

    print(
        f"  Sand+rubble = {sand_rubble_area:.2f} km² "
        f"({sr_percent_mapped:.1f}% mapped)"
    )

    print(
        f"  Sand+rubble connected patches = "
        f"{sr_metrics['patch_count']}"
    )

    print(
        f"  largest sand+rubble patch = "
        f"{sr_metrics['largest_patch_area_km2']:.2f} km²"
    )

    print(
        f"  nearest sand/rubble patch = "
        f"{sr_metrics['nearest_patch_distance_km']:.2f} km"
    )

    site_summary_rows.append(
        {
            "site": site,
            "mapped_benthic_area_km2":
                total_mapped_area_km2,
            "shannon_diversity_H":
                H,
            "shannon_evenness_J":
                evenness,
            "simpson_diversity":
                simpson,
            "dominant_class":
                dominant_class,
            "dominant_class_percent":
                dominant_fraction * 100,
            "sand_rubble_area_km2":
                sand_rubble_area,
            "sand_rubble_percent_mapped":
                sr_percent_mapped,
            "sand_rubble_patch_count":
                sr_metrics["patch_count"],
            "sand_rubble_largest_patch_km2":
                sr_metrics[
                    "largest_patch_area_km2"
                ],
            "sand_rubble_largest_patch_fraction":
                sr_metrics[
                    "largest_patch_fraction"
                ],
            "sand_rubble_nearest_patch_distance_km":
                sr_metrics[
                    "nearest_patch_distance_km"
                ],
            "sand_rubble_area_weighted_mean_distance_km":
                sr_metrics[
                    "area_weighted_mean_patch_distance_km"
                ],
        }
    )

# -------------------------------------------------------------------------
# WRITE TABLES
# -------------------------------------------------------------------------

patch_df = pd.DataFrame(
    patch_rows
)

diversity_df = pd.DataFrame(
    diversity_rows
)

sand_rubble_df = pd.DataFrame(
    sand_rubble_rows
)

site_summary_df = pd.DataFrame(
    site_summary_rows
)

patch_csv = (
    TABLE_DIR
    / "benthic_patch_metrics_70km.csv"
)

diversity_csv = (
    TABLE_DIR
    / "benthic_diversity_70km.csv"
)

sand_rubble_csv = (
    TABLE_DIR
    / "sand_rubble_configuration_70km.csv"
)

site_summary_csv = (
    TABLE_DIR
    / "benthic_configuration_site_summary_70km.csv"
)

patch_df.to_csv(
    patch_csv,
    index=False,
)

diversity_df.to_csv(
    diversity_csv,
    index=False,
)

sand_rubble_df.to_csv(
    sand_rubble_csv,
    index=False,
)

site_summary_df.to_csv(
    site_summary_csv,
    index=False,
)

# -------------------------------------------------------------------------
# CONSISTENT CLASS COLORS USING MATPLOTLIB DEFAULT COLOR CYCLE
# -------------------------------------------------------------------------

default_colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]

class_colors = {
    cls: default_colors[
        i % len(default_colors)
    ]
    for i, cls in enumerate(
        CLASS_ORDER
    )
}

# -------------------------------------------------------------------------
# FIGURE 1 — FIXED 3-PANEL HABITAT MAP
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(14.5, 5.2),
)

panel_labels = ["A", "B", "C"]

for ax, site, panel in zip(
    axes,
    SITE_ORDER,
    panel_labels,
):

    gdf = site_gdfs_wgs84[
        site
    ]

    for cls in CLASS_ORDER:

        subset = gdf[
            gdf["class_name"] == cls
        ]

        if subset.empty:
            continue

        subset.plot(
            ax=ax,
            facecolor=class_colors[cls],
            edgecolor="none",
            linewidth=0,
            alpha=0.85,
        )

    lat, lon = SITE_COORDS[
        site
    ]

    ax.scatter(
        [lon],
        [lat],
        marker="*",
        s=110,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=10,
    )

    ax.set_title(
        site
    )

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

    ax.set_xlabel(
        "Longitude"
    )

    if ax is axes[0]:
        ax.set_ylabel(
            "Latitude"
        )
    else:
        ax.set_ylabel(
            ""
        )

    ax.grid(
        alpha=0.15,
    )

legend_handles = [
    Patch(
        facecolor=class_colors[cls],
        edgecolor="none",
        label=cls,
    )
    for cls in CLASS_ORDER
]

fig.legend(
    handles=legend_handles,
    title="Benthic class",
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.97),
    frameon=False,
)

fig.suptitle(
    "Allen Coral Atlas benthic habitat within 70 km of each site",
    y=1.02,
)

fig.text(
    0.5,
    0.02,
    "Modern satellite-derived habitat snapshot; unmapped areas are not equivalent to unsuitable habitat.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.05, 1, 0.89]
)

map_base = (
    FIG_DIR
    / "Fig_Allen_benthic_maps_70km_FIXED"
)

fig.savefig(
    map_base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    map_base.with_suffix(".pdf"),
    bbox_inches="tight",
)

fig.savefig(
    map_base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 2 — SAND + RUBBLE CONFIGURATION
# -------------------------------------------------------------------------

plot_sr = (
    sand_rubble_df
    .set_index("site")
    .reindex(SITE_ORDER)
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 7),
)

axA, axB, axC, axD = axes.ravel()

# A total area
bars = axA.bar(
    SITE_ORDER,
    plot_sr[
        "sand_rubble_area_km2"
    ],
)

axA.set_ylabel(
    "Sand + rubble area (km²)"
)

axA.set_title(
    "Candidate substrate area"
)

axA.grid(
    axis="y",
    alpha=0.20,
)

for bar, value in zip(
    bars,
    plot_sr[
        "sand_rubble_area_km2"
    ],
):
    axA.text(
        bar.get_x()
        + bar.get_width()/2,
        value,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

axA.text(
    -0.10,
    1.05,
    "A",
    transform=axA.transAxes,
    fontweight="bold",
    fontsize=11,
)

# B patch count
bars = axB.bar(
    SITE_ORDER,
    plot_sr[
        "sand_rubble_patch_count"
    ],
)

axB.set_ylabel(
    "Connected sand+rubble patches"
)

axB.set_title(
    "Patch number"
)

axB.grid(
    axis="y",
    alpha=0.20,
)

for bar, value in zip(
    bars,
    plot_sr[
        "sand_rubble_patch_count"
    ],
):
    axB.text(
        bar.get_x()
        + bar.get_width()/2,
        value,
        f"{int(value)}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

axB.text(
    -0.10,
    1.05,
    "B",
    transform=axB.transAxes,
    fontweight="bold",
    fontsize=11,
)

# C largest patch
bars = axC.bar(
    SITE_ORDER,
    plot_sr[
        "sand_rubble_largest_patch_area_km2"
    ],
)

axC.set_ylabel(
    "Largest sand+rubble patch (km²)"
)

axC.set_title(
    "Largest connected patch"
)

axC.grid(
    axis="y",
    alpha=0.20,
)

for bar, value in zip(
    bars,
    plot_sr[
        "sand_rubble_largest_patch_area_km2"
    ],
):
    axC.text(
        bar.get_x()
        + bar.get_width()/2,
        value,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

axC.text(
    -0.10,
    1.05,
    "C",
    transform=axC.transAxes,
    fontweight="bold",
    fontsize=11,
)

# D nearest patch distance
bars = axD.bar(
    SITE_ORDER,
    plot_sr[
        "sand_rubble_nearest_patch_distance_km"
    ],
)

axD.set_ylabel(
    "Distance from site to nearest\nsand+rubble patch (km)"
)

axD.set_title(
    "Nearest candidate substrate"
)

axD.grid(
    axis="y",
    alpha=0.20,
)

for bar, value in zip(
    bars,
    plot_sr[
        "sand_rubble_nearest_patch_distance_km"
    ],
):
    axD.text(
        bar.get_x()
        + bar.get_width()/2,
        value,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

axD.text(
    -0.10,
    1.05,
    "D",
    transform=axD.transAxes,
    fontweight="bold",
    fontsize=11,
)

fig.suptitle(
    "Spatial configuration of mapped sand and rubble within 70 km",
    y=0.995,
)

fig.text(
    0.5,
    0.01,
    "Sand+rubble is treated as an exploratory spawning-substrate proxy for S. gracilis.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.04, 1, 0.95]
)

sr_base = (
    FIG_DIR
    / "Fig_Allen_sand_rubble_configuration_70km"
)

fig.savefig(
    sr_base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    sr_base.with_suffix(".pdf"),
    bbox_inches="tight",
)

fig.savefig(
    sr_base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 3 — BENTHIC HABITAT DIVERSITY
# -------------------------------------------------------------------------

plot_div = (
    diversity_df
    .set_index("site")
    .reindex(SITE_ORDER)
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.5, 4.2),
)

div_metrics = [
    (
        "shannon_diversity_H",
        "Shannon diversity (H′)",
        "A",
    ),
    (
        "shannon_evenness_J",
        "Shannon evenness (J′)",
        "B",
    ),
    (
        "simpson_diversity_1_minus_D",
        "Simpson diversity (1−D)",
        "C",
    ),
]

for ax, (
    col,
    ylabel,
    panel,
) in zip(
    axes,
    div_metrics,
):

    bars = ax.bar(
        SITE_ORDER,
        plot_div[col],
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    ax.text(
        -0.12,
        1.05,
        panel,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
    )

    for bar, value in zip(
        bars,
        plot_div[col],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width()/2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

fig.suptitle(
    "Benthic habitat diversity within mapped Allen Coral Atlas habitat",
    y=0.995,
)

fig.text(
    0.5,
    0.01,
    "Diversity indices describe proportional composition of mapped benthic classes only.",
    ha="center",
    fontsize=8,
)

fig.tight_layout(
    rect=[0, 0.05, 1, 0.94]
)

div_base = (
    FIG_DIR
    / "Fig_Allen_benthic_diversity_70km"
)

fig.savefig(
    div_base.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    div_base.with_suffix(".pdf"),
    bbox_inches="tight",
)

fig.savefig(
    div_base.with_suffix(".svg"),
    bbox_inches="tight",
)

plt.close(fig)

# -------------------------------------------------------------------------
# TERMINAL SUMMARY
# -------------------------------------------------------------------------

print()
print("=" * 78)
print("SCRIPT 13 COMPLETE")
print("=" * 78)

print("\nSand + rubble configuration:")
print(
    sand_rubble_df[
        [
            "site",
            "sand_rubble_area_km2",
            "sand_rubble_percent_mapped",
            "sand_rubble_patch_count",
            "sand_rubble_mean_patch_area_km2",
            "sand_rubble_median_patch_area_km2",
            "sand_rubble_largest_patch_area_km2",
            "sand_rubble_largest_patch_fraction",
            "sand_rubble_nearest_patch_distance_km",
            "sand_rubble_area_weighted_mean_patch_distance_km",
        ]
    ].to_string(
        index=False
    )
)

print("\nBenthic diversity:")
print(
    diversity_df[
        [
            "site",
            "class_richness",
            "shannon_diversity_H",
            "shannon_evenness_J",
            "simpson_diversity_1_minus_D",
            "dominant_class",
            "dominant_class_percent",
        ]
    ].to_string(
        index=False
    )
)

print("\nTables:")
print(f"  {patch_csv}")
print(f"  {diversity_csv}")
print(f"  {sand_rubble_csv}")
print(f"  {site_summary_csv}")

print("\nFigures:")
print(
    f"  {map_base.with_suffix('.pdf')}"
)
print(
    f"  {sr_base.with_suffix('.pdf')}"
)
print(
    f"  {div_base.with_suffix('.pdf')}"
)

print(
    "\nInterpretation reminder: these are modern spatial habitat metrics, "
    "not historical 1900 habitat reconstructions."
)
