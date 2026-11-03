
from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(r"D:\temporal_environmental_data")
GPKG_DIR = ROOT / "01_raw" / "Allen_Coral_Atlas_WFS"
TABLE_DIR = ROOT / "04_tables"
FIG_DIR = ROOT / "05_figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SITE_ORDER = ["Jolo", "Masinloc", "Palapag"]
SITE_COORDS = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}
BUFFER_KM = 70
RESOLUTIONS_M = [100, 250]
MIN_PATCH_HA = [1, 5, 10]
PRIMARY_RES_M = 100
PRIMARY_MIN_HA = 5

def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone

def patch_metrics(binary, transform, resolution_m, min_patch_ha, site_xy):
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n_raw = ndimage.label(binary.astype(np.uint8), structure=structure)

    if n_raw == 0:
        return ({
            "raw_patch_count": 0,
            "filtered_patch_count": 0,
            "total_candidate_area_km2": 0.0,
            "mean_patch_area_km2": np.nan,
            "median_patch_area_km2": np.nan,
            "largest_patch_area_km2": np.nan,
            "largest_patch_fraction": np.nan,
            "patch_area_cv": np.nan,
            "nearest_patch_distance_km": np.nan,
            "area_weighted_mean_patch_distance_km": np.nan,
            "mean_nearest_neighbor_centroid_km": np.nan,
            "median_nearest_neighbor_centroid_km": np.nan,
        }, pd.DataFrame(), np.zeros_like(binary, dtype=bool))

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    cell_area_m2 = resolution_m * resolution_m
    patch_area_ha = counts * cell_area_m2 / 10000.0
    keep_labels = np.where(patch_area_ha >= min_patch_ha)[0]
    keep_labels = keep_labels[keep_labels != 0]

    if len(keep_labels) == 0:
        return ({
            "raw_patch_count": int(n_raw),
            "filtered_patch_count": 0,
            "total_candidate_area_km2": 0.0,
            "mean_patch_area_km2": np.nan,
            "median_patch_area_km2": np.nan,
            "largest_patch_area_km2": np.nan,
            "largest_patch_fraction": np.nan,
            "patch_area_cv": np.nan,
            "nearest_patch_distance_km": np.nan,
            "area_weighted_mean_patch_distance_km": np.nan,
            "mean_nearest_neighbor_centroid_km": np.nan,
            "median_nearest_neighbor_centroid_km": np.nan,
        }, pd.DataFrame(), np.zeros_like(binary, dtype=bool))

    filtered = np.isin(labels, keep_labels)
    sx, sy = site_xy
    patch_rows = []
    centroids = []

    for lab in keep_labels:
        rr, cc = np.where(labels == lab)
        n_cells = len(rr)
        area_km2 = n_cells * cell_area_m2 / 1e6
        xs = transform.c + (cc + 0.5) * transform.a
        ys = transform.f + (rr + 0.5) * transform.e
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        centroids.append((cx, cy))
        centroid_distance_km = math.hypot(cx - sx, cy - sy) / 1000.0
        dmin_km = float(np.min(np.sqrt((xs - sx)**2 + (ys - sy)**2)) / 1000.0)

        patch_rows.append({
            "patch_id": int(lab),
            "n_cells": int(n_cells),
            "area_km2": float(area_km2),
            "centroid_x_m": cx,
            "centroid_y_m": cy,
            "centroid_distance_from_site_km": centroid_distance_km,
            "nearest_cell_distance_from_site_km": dmin_km,
        })

    patch_df = pd.DataFrame(patch_rows)
    areas = patch_df["area_km2"].to_numpy(float)
    total_area = float(areas.sum())
    largest = float(areas.max())
    mean_area = float(areas.mean())
    median_area = float(np.median(areas))
    cv = float(areas.std(ddof=1) / mean_area) if len(areas) > 1 and mean_area > 0 else np.nan
    nearest_patch_distance = float(patch_df["nearest_cell_distance_from_site_km"].min())
    weighted_mean_distance = float(np.average(
        patch_df["centroid_distance_from_site_km"], weights=areas
    ))

    centroids = np.asarray(centroids, float)
    if len(centroids) >= 2:
        tree = cKDTree(centroids)
        dists, _ = tree.query(centroids, k=2)
        nn_km = dists[:, 1] / 1000.0
        mean_nn = float(np.mean(nn_km))
        median_nn = float(np.median(nn_km))
    else:
        mean_nn = np.nan
        median_nn = np.nan

    metrics = {
        "raw_patch_count": int(n_raw),
        "filtered_patch_count": int(len(keep_labels)),
        "total_candidate_area_km2": total_area,
        "mean_patch_area_km2": mean_area,
        "median_patch_area_km2": median_area,
        "largest_patch_area_km2": largest,
        "largest_patch_fraction": largest / total_area if total_area > 0 else np.nan,
        "patch_area_cv": cv,
        "nearest_patch_distance_km": nearest_patch_distance,
        "area_weighted_mean_patch_distance_km": weighted_mean_distance,
        "mean_nearest_neighbor_centroid_km": mean_nn,
        "median_nearest_neighbor_centroid_km": median_nn,
    }
    return metrics, patch_df, filtered

def rasterize_site(site, resolution_m):
    lat, lon = SITE_COORDS[site]
    gpkg = GPKG_DIR / f"{site}_Allen_Coral_Atlas_benthic_70km.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"Missing GeoPackage:\n{gpkg}")

    gdf = gpd.read_file(gpkg, layer="benthic_70km")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    epsg = utm_epsg(lat, lon)
    gdf = gdf.to_crs(epsg=epsg)

    site_point = (
        gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        .to_crs(epsg=epsg)
        .iloc[0]
    )
    buffer_geom = site_point.buffer(BUFFER_KM * 1000)
    minx, miny, maxx, maxy = buffer_geom.bounds

    width = int(math.ceil((maxx - minx) / resolution_m))
    height = int(math.ceil((maxy - miny) / resolution_m))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)

    buffer_raster = rasterize(
        [(buffer_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    sr = gdf[gdf["class_name"].isin(["Sand", "Rubble"])].copy()

    if sr.empty:
        candidate = np.zeros((height, width), dtype=bool)
    else:
        candidate = rasterize(
            [(geom, 1) for geom in sr.geometry if geom is not None and not geom.is_empty],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)

    candidate &= buffer_raster

    return {
        "candidate": candidate,
        "buffer_mask": buffer_raster,
        "transform": transform,
        "epsg": epsg,
        "site_xy": (site_point.x, site_point.y),
    }

print("=" * 78)
print("SCRIPT 14 — RASTER-BASED SAND/RUBBLE CONNECTIVITY")
print("=" * 78)

sensitivity_rows = []
primary_patch_rows = []
primary_map_data = {}

for site in SITE_ORDER:
    print(f"\n{site}")

    for res_m in RESOLUTIONS_M:
        print(f"  Rasterizing at {res_m} m...")
        rast = rasterize_site(site, res_m)

        raw_area_km2 = float(
            rast["candidate"].sum() * res_m * res_m / 1e6
        )
        print(f"    rasterized candidate area: {raw_area_km2:.2f} km²")

        for min_ha in MIN_PATCH_HA:
            metrics, patch_df, filtered = patch_metrics(
                rast["candidate"],
                rast["transform"],
                res_m,
                min_ha,
                rast["site_xy"],
            )

            sensitivity_rows.append({
                "site": site,
                "buffer_km": BUFFER_KM,
                "resolution_m": res_m,
                "minimum_patch_area_ha": min_ha,
                "raw_raster_candidate_area_km2": raw_area_km2,
                **metrics,
                "connectivity_rule": "8-neighbour",
                "substrate_proxy": "Sand + Rubble",
                "data_source": "Allen Coral Atlas",
                "temporal_status": "modern_snapshot",
                "analysis_status": "standardized_raster_patch_analysis",
            })

            nn = metrics["mean_nearest_neighbor_centroid_km"]
            nn_txt = f"{nn:.2f}" if np.isfinite(nn) else "NA"

            print(
                f"    MMU {min_ha:>2} ha: "
                f"{metrics['filtered_patch_count']} patches, "
                f"area={metrics['total_candidate_area_km2']:.2f} km², "
                f"largest={metrics['largest_patch_area_km2']:.2f} km², "
                f"NN={nn_txt} km"
            )

            if res_m == PRIMARY_RES_M and min_ha == PRIMARY_MIN_HA:
                primary_map_data[site] = {
                    "filtered": filtered,
                    "buffer_mask": rast["buffer_mask"],
                    "transform": rast["transform"],
                    "epsg": rast["epsg"],
                    "site_xy": rast["site_xy"],
                }

                if not patch_df.empty:
                    patch_df = patch_df.copy()
                    patch_df["site"] = site
                    patch_df["resolution_m"] = res_m
                    patch_df["minimum_patch_area_ha"] = min_ha
                    primary_patch_rows.append(patch_df)

sensitivity_df = pd.DataFrame(sensitivity_rows)

primary_df = sensitivity_df[
    (sensitivity_df["resolution_m"] == PRIMARY_RES_M)
    & (sensitivity_df["minimum_patch_area_ha"] == PRIMARY_MIN_HA)
].copy()

primary_df["site"] = pd.Categorical(
    primary_df["site"],
    categories=SITE_ORDER,
    ordered=True,
)
primary_df = primary_df.sort_values("site")

primary_patch_df = (
    pd.concat(primary_patch_rows, ignore_index=True)
    if primary_patch_rows
    else pd.DataFrame()
)

sens_csv = TABLE_DIR / "sand_rubble_raster_patch_metrics_sensitivity.csv"
primary_csv = TABLE_DIR / "sand_rubble_raster_patch_metrics_PRIMARY_100m_5ha.csv"
patch_csv = TABLE_DIR / "sand_rubble_raster_patch_level_PRIMARY_100m_5ha.csv"

sensitivity_df.to_csv(sens_csv, index=False)
primary_df.to_csv(primary_csv, index=False)
primary_patch_df.to_csv(patch_csv, index=False)

# Figure 1: primary metrics
fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
metrics_to_plot = [
    ("filtered_patch_count", "Patches ≥5 ha", "A", "{:.0f}"),
    ("largest_patch_area_km2", "Largest patch (km²)", "B", "{:.2f}"),
    ("mean_nearest_neighbor_centroid_km", "Mean nearest-neighbour\ncentroid distance (km)", "C", "{:.2f}"),
    ("nearest_patch_distance_km", "Distance from site to\nnearest patch (km)", "D", "{:.2f}"),
]

for ax, (col, ylabel, panel, fmt) in zip(axes.ravel(), metrics_to_plot):
    vals = primary_df[col].to_numpy(float)
    bars = ax.bar(SITE_ORDER, vals)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.20)
    ax.text(-0.10, 1.05, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)

    for bar, value in zip(bars, vals):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                value,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )

fig.suptitle("Standardized sand+rubble patch configuration within 70 km", y=0.995)
fig.text(
    0.5, 0.01,
    "Primary analysis: 100 m raster, 8-neighbour connectivity, minimum patch area 5 ha.",
    ha="center", fontsize=8
)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])

primary_fig = FIG_DIR / "Fig_sand_rubble_raster_patch_metrics_PRIMARY"
fig.savefig(primary_fig.with_suffix(".png"), dpi=400, bbox_inches="tight")
fig.savefig(primary_fig.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(primary_fig.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

# Figure 2: sensitivity
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

for ax, res_m, panel in zip(axes, RESOLUTIONS_M, ["A", "B"]):
    sub = sensitivity_df[sensitivity_df["resolution_m"] == res_m].copy()

    for site in SITE_ORDER:
        s = sub[sub["site"] == site].sort_values("minimum_patch_area_ha")
        ax.plot(
            s["minimum_patch_area_ha"],
            s["filtered_patch_count"],
            marker="o",
            label=site,
        )

    ax.set_xlabel("Minimum patch area (ha)")
    ax.set_ylabel("Retained patch count")
    ax.set_title(f"{res_m} m raster")
    ax.grid(alpha=0.20)
    ax.text(-0.08, 1.06, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)

axes[0].legend(frameon=False)
fig.suptitle("Sensitivity of sand+rubble patch counts to mapping scale", y=1.02)
fig.tight_layout()

sens_fig = FIG_DIR / "Fig_sand_rubble_patch_sensitivity"
fig.savefig(sens_fig.with_suffix(".png"), dpi=400, bbox_inches="tight")
fig.savefig(sens_fig.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(sens_fig.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

# Figure 3: primary raster maps
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

for ax, site, panel in zip(axes, SITE_ORDER, ["A", "B", "C"]):
    data = primary_map_data[site]
    filt = data["filtered"]
    transform = data["transform"]
    h, w = filt.shape

    xmin = transform.c
    ymax = transform.f
    xmax = xmin + w * transform.a
    ymin = ymax + h * transform.e
    extent = [xmin/1000.0, xmax/1000.0, ymin/1000.0, ymax/1000.0]

    masked = np.ma.masked_where(~filt, filt.astype(float))

    ax.imshow(
        masked,
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )

    sx, sy = data["site_xy"]

    ax.scatter(
        [sx/1000.0],
        [sy/1000.0],
        marker="*",
        s=90,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )

    ax.set_title(site)
    ax.set_xlabel("Easting (km)")
    if ax is axes[0]:
        ax.set_ylabel("Northing (km)")

    ax.text(
        0.02, 0.98, panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=11,
    )

fig.suptitle("Mapped sand+rubble patches after standardized raster filtering", y=1.02)
fig.text(
    0.5, 0.01,
    "100 m raster; patches <5 ha removed. White star marks sampling locality.",
    ha="center", fontsize=8
)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])

map_fig = FIG_DIR / "Fig_sand_rubble_raster_maps_PRIMARY"
fig.savefig(map_fig.with_suffix(".png"), dpi=400, bbox_inches="tight")
fig.savefig(map_fig.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(map_fig.with_suffix(".svg"), bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 78)
print("SCRIPT 14 COMPLETE")
print("=" * 78)
print("\nPRIMARY ANALYSIS — 100 m raster, MMU 5 ha")

show_cols = [
    "site",
    "raw_patch_count",
    "filtered_patch_count",
    "total_candidate_area_km2",
    "mean_patch_area_km2",
    "median_patch_area_km2",
    "largest_patch_area_km2",
    "largest_patch_fraction",
    "nearest_patch_distance_km",
    "area_weighted_mean_patch_distance_km",
    "mean_nearest_neighbor_centroid_km",
    "median_nearest_neighbor_centroid_km",
]
print(primary_df[show_cols].to_string(index=False))

print("\nSensitivity table:")
print(
    sensitivity_df[
        [
            "site",
            "resolution_m",
            "minimum_patch_area_ha",
            "filtered_patch_count",
            "total_candidate_area_km2",
            "largest_patch_area_km2",
            "mean_nearest_neighbor_centroid_km",
        ]
    ].to_string(index=False)
)

print("\nTables:")
print(f"  {sens_csv}")
print(f"  {primary_csv}")
print(f"  {patch_csv}")

print("\nFigures:")
print(f"  {primary_fig.with_suffix('.pdf')}")
print(f"  {sens_fig.with_suffix('.pdf')}")
print(f"  {map_fig.with_suffix('.pdf')}")

print(
    "\nInterpretation reminder: these are modern standardized spatial metrics "
    "for mapped Sand + Rubble, not reconstructed 1900 habitat."
)
