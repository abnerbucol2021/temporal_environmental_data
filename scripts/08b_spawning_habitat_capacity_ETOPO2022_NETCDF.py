
"""
08b_spawning_habitat_capacity_ETOPO2022_NETCDF.py
=================================================

Derive bathymetric/geomorphological components of potential spawning habitat
capacity for Spratelloides gracilis from the NOAA ETOPO 2022 15-arcsecond
Philippines NetCDF subset downloaded through ERDDAP.

Sites:
  Jolo      6.1098, 120.9775
  Masinloc 15.55672, 119.93036
  Palapag  12.63476, 125.00978

Buffers:
  25, 50, 70 km

Outputs:
  03_extracted/spawning_habitat_capacity_bathymetry_25_50_70km.csv
  04_tables/spawning_habitat_capacity_bathymetry_70km_summary.csv
  04_tables/spawning_habitat_capacity_bathymetry_exploratory_index.csv
  05_figures/Fig_spawning_habitat_shallow_area_70km.png
  05_figures/Fig_spawning_habitat_shallow_area_70km.pdf

Primary variables:
  marine area
  0-10 m shallow area
  10-20 m shallow area
  20-30 m shallow area
  0-30 m total shallow area
  0-50 m shallow area
  fraction of marine area that is 0-30 m
  mean/median marine depth
  mean slope within 0-30 m
  0-30 m patch count
  largest shallow patch
  mean patch size
  patch density
  fragmentation index

Temporal status:
  modern/static or quasi-static geomorphological proxy
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

ROOT = Path(r"D:\temporal_environmental_data")

NC_FILE = (
    ROOT / "01_raw" / "ETOPO2022_bathymetry"
    / "ETOPO2022_15s_Philippines_118_126p5E_5_17N.nc"
)

OUT_DIR = ROOT / "03_extracted"
TABLE_DIR = ROOT / "04_tables"
FIG_DIR = ROOT / "05_figures"

for p in (OUT_DIR, TABLE_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

SITES = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}

BUFFERS = [25, 50, 70]


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


def cell_area_km2(lat_center, dlat_deg, dlon_deg):
    R = 6371.0088

    lat1 = np.radians(lat_center - dlat_deg / 2)
    lat2 = np.radians(lat_center + dlat_deg / 2)

    return (
        R**2
        * np.radians(abs(dlon_deg))
        * abs(np.sin(lat2) - np.sin(lat1))
    )


print("=" * 78)
print("POTENTIAL SPAWNING HABITAT CAPACITY — BATHYMETRIC COMPONENT")
print("=" * 78)

if not NC_FILE.exists():
    raise FileNotFoundError(
        f"ETOPO subset not found:\n{NC_FILE}\n"
        "Run 07b_bathymetry_download_ETOPO2022_ERDDAP.py first."
    )

with xr.open_dataset(NC_FILE, decode_times=False) as ds:
    lat_name = (
        "latitude" if "latitude" in ds.coords
        else "lat"
    )
    lon_name = (
        "longitude" if "longitude" in ds.coords
        else "lon"
    )

    z = ds["z"].load()
    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)

# Ensure latitude increases south -> north for gradient logic.
if lat[0] > lat[-1]:
    lat = lat[::-1]
    z = z.isel({lat_name: slice(None, None, -1)})

elev = np.asarray(z.values, dtype=float)

# Squeeze any unexpected singleton dimensions.
elev = np.squeeze(elev)

if elev.ndim != 2:
    raise RuntimeError(
        f"Expected 2-D elevation grid, got shape {elev.shape}"
    )

# Coordinates.
lon2d, lat2d = np.meshgrid(lon, lat)

dlat = float(np.median(np.diff(lat)))
dlon = float(np.median(np.diff(lon)))

print(f"Grid shape: {elev.shape}")
print(f"Resolution: {abs(dlat):.8f}° x {abs(dlon):.8f}°")
print(
    f"Elevation range: {np.nanmin(elev):.1f} to {np.nanmax(elev):.1f} m"
)

# Negative = below sea level.
marine = elev < 0
depth = np.where(marine, -elev, np.nan)

# Cell area by latitude.
area_by_row = np.array(
    [
        cell_area_km2(phi, abs(dlat), abs(dlon))
        for phi in lat
    ]
)

cell_area = np.repeat(
    area_by_row[:, None],
    len(lon),
    axis=1,
)

# Bathymetric slope.
# Fill land with nearest-ish NaN behavior only for gradient calculations;
# we later restrict results to marine shallow cells.
depth_filled = np.where(np.isfinite(depth), depth, np.nan)

dy_m = 110_574.0 * abs(dlat)
dx_m = (
    111_320.0
    * np.cos(np.radians(lat2d))
    * abs(dlon)
)

dzdy = np.gradient(depth_filled, axis=0)
dzdx = np.gradient(depth_filled, axis=1)

with np.errstate(divide="ignore", invalid="ignore"):
    gy = dzdy / dy_m
    gx = dzdx / dx_m

slope_deg = np.degrees(
    np.arctan(
        np.sqrt(gx**2 + gy**2)
    )
)

rows = []

for site, (site_lat, site_lon) in SITES.items():

    dist = haversine_km(
        site_lat,
        site_lon,
        lat2d,
        lon2d,
    )

    print(f"\n{site}")

    for radius in BUFFERS:

        inside = dist <= radius
        marine_inside = inside & marine

        marine_area = float(
            np.nansum(cell_area[marine_inside])
        )

        m0_10 = marine_inside & (depth > 0) & (depth <= 10)
        m10_20 = marine_inside & (depth > 10) & (depth <= 20)
        m20_30 = marine_inside & (depth > 20) & (depth <= 30)
        m0_30 = marine_inside & (depth > 0) & (depth <= 30)
        m0_50 = marine_inside & (depth > 0) & (depth <= 50)

        area_0_10 = float(np.nansum(cell_area[m0_10]))
        area_10_20 = float(np.nansum(cell_area[m10_20]))
        area_20_30 = float(np.nansum(cell_area[m20_30]))
        area_0_30 = float(np.nansum(cell_area[m0_30]))
        area_0_50 = float(np.nansum(cell_area[m0_50]))

        depths_here = depth[marine_inside]

        mean_depth = (
            float(np.nanmean(depths_here))
            if np.any(np.isfinite(depths_here))
            else np.nan
        )

        median_depth = (
            float(np.nanmedian(depths_here))
            if np.any(np.isfinite(depths_here))
            else np.nan
        )

        shallow_slopes = slope_deg[m0_30]

        mean_shallow_slope = (
            float(np.nanmean(shallow_slopes))
            if np.any(np.isfinite(shallow_slopes))
            else np.nan
        )

        # Connected components in 0-30 m shallow habitat.
        labels, n_patches = ndimage.label(
            m0_30,
            structure=np.ones((3, 3), dtype=int),
        )

        patch_areas = []

        for label_id in range(1, n_patches + 1):
            pmask = labels == label_id

            patch_areas.append(
                float(
                    np.nansum(
                        cell_area[pmask]
                    )
                )
            )

        largest_patch = (
            max(patch_areas)
            if patch_areas
            else 0.0
        )

        mean_patch = (
            float(np.mean(patch_areas))
            if patch_areas
            else 0.0
        )

        patch_density = (
            n_patches / area_0_30 * 100.0
            if area_0_30 > 0
            else np.nan
        )

        fragmentation = (
            1.0 - largest_patch / area_0_30
            if area_0_30 > 0
            else np.nan
        )

        shallow_fraction = (
            area_0_30 / marine_area
            if marine_area > 0
            else np.nan
        )

        print(
            f"  {radius} km: "
            f"marine={marine_area:.1f} km², "
            f"0–30m={area_0_30:.1f} km², "
            f"10–20m={area_10_20:.1f} km², "
            f"patches={n_patches}"
        )

        rows.append(
            {
                "site": site,
                "latitude": site_lat,
                "longitude": site_lon,
                "buffer_km": radius,

                "marine_area_km2": marine_area,

                "shallow_area_0_10m_km2": area_0_10,
                "shallow_area_10_20m_km2": area_10_20,
                "shallow_area_20_30m_km2": area_20_30,
                "shallow_area_0_30m_km2": area_0_30,
                "shallow_area_0_50m_km2": area_0_50,

                "shallow_0_30m_fraction_of_marine_area":
                    shallow_fraction,

                "mean_marine_depth_m": mean_depth,
                "median_marine_depth_m": median_depth,

                "mean_slope_0_30m_deg":
                    mean_shallow_slope,

                "shallow_patch_count_0_30m":
                    int(n_patches),

                "largest_shallow_patch_km2":
                    largest_patch,

                "mean_shallow_patch_km2":
                    mean_patch,

                "shallow_patch_density_per_100km2":
                    patch_density,

                "shallow_fragmentation_index":
                    fragmentation,

                "bathymetry_dataset":
                    "NOAA/NCEI ETOPO 2022 v1 via NOAA PIFSC ERDDAP",

                "bathymetry_resolution":
                    "15 arc-second",

                "bathymetry_data_class":
                    "compiled_global_relief_model",

                "derived_variable_class":
                    "derived_geomorphological_spawning_habitat_proxy",

                "temporal_status":
                    "modern_static_or_quasi_static",
            }
        )

df = pd.DataFrame(rows)

all_csv = (
    OUT_DIR
    / "spawning_habitat_capacity_bathymetry_25_50_70km.csv"
)

df.to_csv(all_csv, index=False)

summary70 = (
    df[df["buffer_km"] == 70]
    .sort_values("site")
    .copy()
)

summary_csv = (
    TABLE_DIR
    / "spawning_habitat_capacity_bathymetry_70km_summary.csv"
)

summary70.to_csv(summary_csv, index=False)

# Exploratory index: do not interpret as final PSHC yet.
components = [
    "shallow_area_0_30m_km2",
    "shallow_area_10_20m_km2",
    "shallow_0_30m_fraction_of_marine_area",
    "largest_shallow_patch_km2",
]

rank = summary70[
    ["site"] + components
].copy()

for c in components:
    vals = rank[c].astype(float)

    sd = vals.std(ddof=0)

    if sd > 0:
        rank[c + "_z"] = (
            vals - vals.mean()
        ) / sd
    else:
        rank[c + "_z"] = 0.0

zcols = [
    c + "_z"
    for c in components
]

rank[
    "bathymetric_habitat_capacity_index_exploratory"
] = rank[zcols].mean(axis=1)

rank_csv = (
    TABLE_DIR
    / "spawning_habitat_capacity_bathymetry_exploratory_index.csv"
)

rank.to_csv(rank_csv, index=False)

print("\n70-km SUMMARY")
print("-" * 78)

show_cols = [
    "site",
    "marine_area_km2",
    "shallow_area_0_10m_km2",
    "shallow_area_10_20m_km2",
    "shallow_area_20_30m_km2",
    "shallow_area_0_30m_km2",
    "shallow_0_30m_fraction_of_marine_area",
    "mean_slope_0_30m_deg",
    "shallow_patch_count_0_30m",
    "largest_shallow_patch_km2",
    "shallow_fragmentation_index",
]

print(
    summary70[show_cols]
    .to_string(index=False)
)

print("\nEXPLORATORY BATHYMETRIC CAPACITY INDEX")
print("-" * 78)

print(
    rank[
        [
            "site",
            "bathymetric_habitat_capacity_index_exploratory",
        ]
    ]
    .sort_values(
        "bathymetric_habitat_capacity_index_exploratory",
        ascending=False,
    )
    .to_string(index=False)
)

# Figure.
try:
    import matplotlib.pyplot as plt

    plot_df = (
        summary70
        .set_index("site")
        [
            [
                "shallow_area_0_10m_km2",
                "shallow_area_10_20m_km2",
                "shallow_area_20_30m_km2",
            ]
        ]
    )

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
    )

    ax.set_ylabel(
        "Shallow marine area (km²)"
    )

    ax.set_xlabel(
        "Site"
    )

    ax.set_title(
        "Potential shallow spawning habitat within 70 km"
    )

    ax.legend(
        [
            "0–10 m",
            "10–20 m",
            "20–30 m",
        ],
        title="Depth band",
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig = ax.get_figure()
    fig.tight_layout()

    png = (
        FIG_DIR
        / "Fig_spawning_habitat_shallow_area_70km.png"
    )

    pdf = (
        FIG_DIR
        / "Fig_spawning_habitat_shallow_area_70km.pdf"
    )

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nFigure PNG:\n  {png}")
    print(f"Figure PDF:\n  {pdf}")

except Exception as exc:
    print(f"\nFigure creation skipped: {exc}")

print("\nEXTRACTION COMPLETE")
print(f"All buffers:\n  {all_csv}")
print(f"70-km summary:\n  {summary_csv}")
print(f"Exploratory index:\n  {rank_csv}")
