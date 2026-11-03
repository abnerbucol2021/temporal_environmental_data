
from pathlib import Path
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from scipy import ndimage

ROOT = Path(r"D:\temporal_environmental_data")
RAW = ROOT / "01_raw" / "ETOPO2022_bathymetry"
TABLE_DIR = ROOT / "04_tables"
OUT_DIR = ROOT / "03_extracted"
FIG_DIR = ROOT / "05_figures"

for p in (TABLE_DIR, OUT_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

SITES = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}
BUFFERS = [25, 50, 70]

TILES = sorted(RAW.glob("ETOPO_2022_v1_15s_*_bed.tif"))
if len(TILES) < 4:
    raise RuntimeError(f"Expected 4 ETOPO tiles in {RAW}, found {len(TILES)}.")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2-lat1; dlon = lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def cell_area_km2(lat_center, dlat_deg, dlon_deg):
    R = 6371.0088
    lat1 = np.radians(lat_center-dlat_deg/2)
    lat2 = np.radians(lat_center+dlat_deg/2)
    return R**2*np.radians(abs(dlon_deg))*abs(np.sin(lat2)-np.sin(lat1))

print("="*78)
print("POTENTIAL SPAWNING HABITAT CAPACITY — BATHYMETRIC COMPONENT")
print("="*78)

srcs = [rasterio.open(p) for p in TILES]
mosaic, transform = merge(srcs)
for s in srcs: s.close()

elev = mosaic[0].astype(float)
height, width = elev.shape

xs = transform.c + (np.arange(width)+0.5)*transform.a
ys = transform.f + (np.arange(height)+0.5)*transform.e
lon2d, lat2d = np.meshgrid(xs, ys)

dx_deg, dy_deg = abs(transform.a), abs(transform.e)
marine = elev < 0
depth = np.where(marine, -elev, np.nan)

area_by_row = np.array([cell_area_km2(lat, dy_deg, dx_deg) for lat in ys])
cell_area = np.repeat(area_by_row[:,None], width, axis=1)

# slope
dy_m = 110574.0*dy_deg
filled = np.where(np.isfinite(depth), depth, np.nan)
dzdy = np.gradient(filled, axis=0)
dzdx = np.gradient(filled, axis=1)
dx_m = 111320.0*np.cos(np.radians(lat2d))*dx_deg
with np.errstate(divide="ignore", invalid="ignore"):
    gy = dzdy/dy_m
    gx = dzdx/dx_m
slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))

rows=[]

for site,(slat,slon) in SITES.items():
    dist = haversine_km(slat, slon, lat2d, lon2d)

    for radius in BUFFERS:
        inside = dist <= radius
        mi = inside & marine
        marine_area = float(np.nansum(cell_area[mi]))

        bands = {
            "shallow_area_0_10m_km2": mi & (depth>0) & (depth<=10),
            "shallow_area_10_20m_km2": mi & (depth>10) & (depth<=20),
            "shallow_area_20_30m_km2": mi & (depth>20) & (depth<=30),
        }

        areas = {k: float(np.nansum(cell_area[m])) for k,m in bands.items()}
        shallow30 = mi & (depth>0) & (depth<=30)
        shallow50 = mi & (depth>0) & (depth<=50)

        area30 = float(np.nansum(cell_area[shallow30]))
        area50 = float(np.nansum(cell_area[shallow50]))

        dvals = depth[mi]
        mean_depth = float(np.nanmean(dvals))
        median_depth = float(np.nanmedian(dvals))

        svals = slope_deg[shallow30]
        mean_slope = float(np.nanmean(svals)) if np.any(np.isfinite(svals)) else np.nan

        labels,npatch = ndimage.label(shallow30, structure=np.ones((3,3),dtype=int))
        patch_areas=[]
        for lid in range(1,npatch+1):
            m = labels==lid
            patch_areas.append(float(np.nansum(cell_area[m])))

        largest_patch = max(patch_areas) if patch_areas else 0.0
        mean_patch = float(np.mean(patch_areas)) if patch_areas else 0.0
        fragmentation = 1-largest_patch/area30 if area30>0 else np.nan
        patch_density = npatch/area30*100 if area30>0 else np.nan

        rows.append({
            "site":site,
            "latitude":slat,
            "longitude":slon,
            "buffer_km":radius,
            "marine_area_km2":marine_area,
            **areas,
            "shallow_area_0_30m_km2":area30,
            "shallow_area_0_50m_km2":area50,
            "shallow_0_30m_fraction_of_marine_area":area30/marine_area if marine_area>0 else np.nan,
            "mean_marine_depth_m":mean_depth,
            "median_marine_depth_m":median_depth,
            "mean_slope_0_30m_deg":mean_slope,
            "shallow_patch_count_0_30m":int(npatch),
            "largest_shallow_patch_km2":largest_patch,
            "mean_shallow_patch_km2":mean_patch,
            "shallow_patch_density_per_100km2":patch_density,
            "shallow_fragmentation_index":fragmentation,
            "bathymetry_dataset":"NOAA ETOPO 2022 v1",
            "bathymetry_resolution":"15 arc-second",
            "bathymetry_data_class":"compiled_global_relief_model",
            "derived_variable_class":"derived_geomorphological_spawning_habitat_proxy",
            "temporal_status":"modern_static_or_quasi_static",
        })

df = pd.DataFrame(rows)

out_csv = OUT_DIR / "spawning_habitat_capacity_bathymetry_25_50_70km.csv"
df.to_csv(out_csv,index=False)

summary70 = df[df.buffer_km==70].sort_values("site").copy()
summary_csv = TABLE_DIR / "spawning_habitat_capacity_bathymetry_70km_summary.csv"
summary70.to_csv(summary_csv,index=False)

components = [
    "shallow_area_0_30m_km2",
    "shallow_0_30m_fraction_of_marine_area",
    "largest_shallow_patch_km2",
]
rank = summary70[["site"]+components].copy()
for c in components:
    v=rank[c].astype(float)
    rank[c+"_z"]=(v-v.mean())/v.std(ddof=0) if v.std(ddof=0)>0 else 0.0
zcols=[c+"_z" for c in components]
rank["bathymetric_habitat_capacity_index_exploratory"]=rank[zcols].mean(axis=1)
rank_csv = TABLE_DIR / "spawning_habitat_capacity_bathymetry_exploratory_index.csv"
rank.to_csv(rank_csv,index=False)

print("\n70-km SUMMARY")
print(summary70[[
    "site","marine_area_km2",
    "shallow_area_0_10m_km2","shallow_area_10_20m_km2",
    "shallow_area_20_30m_km2","shallow_area_0_30m_km2",
    "shallow_0_30m_fraction_of_marine_area",
    "shallow_patch_count_0_30m","largest_shallow_patch_km2",
    "mean_slope_0_30m_deg"
]].to_string(index=False))

print("\nEXPLORATORY BATHYMETRIC CAPACITY INDEX")
print(rank[["site","bathymetric_habitat_capacity_index_exploratory"]]
      .sort_values("bathymetric_habitat_capacity_index_exploratory",ascending=False)
      .to_string(index=False))

# figure
try:
    import matplotlib.pyplot as plt
    p = summary70.set_index("site")[[
        "shallow_area_0_10m_km2",
        "shallow_area_10_20m_km2",
        "shallow_area_20_30m_km2",
    ]]
    ax=p.plot(kind="bar",stacked=True,figsize=(8,5))
    ax.set_ylabel("Shallow marine area (km²)")
    ax.set_xlabel("Site")
    ax.set_title("Potential shallow spawning habitat within 70 km")
    ax.legend(["0–10 m","10–20 m","20–30 m"],title="Depth band",frameon=False)
    ax.grid(axis="y",alpha=0.2)
    fig=ax.get_figure()
    fig.tight_layout()
    png=FIG_DIR/"Fig_spawning_habitat_shallow_area_70km.png"
    pdf=FIG_DIR/"Fig_spawning_habitat_shallow_area_70km.pdf"
    fig.savefig(png,dpi=400,bbox_inches="tight")
    fig.savefig(pdf,bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure PNG: {png}")
    print(f"Figure PDF: {pdf}")
except Exception as e:
    print(f"\nFigure skipped: {e}")

print("\nEXTRACTION COMPLETE")
print(out_csv)
print(summary_csv)
print(rank_csv)
