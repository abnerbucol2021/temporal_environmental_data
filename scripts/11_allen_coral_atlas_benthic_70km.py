
"""
11_allen_coral_atlas_benthic_70km.py
====================================

Full Allen Coral Atlas benthic-habitat extraction for:
  Jolo
  Masinloc
  Palapag

Buffer:
  70 km

WFS layer:
  coral-atlas:benthic_data_verbose

Classes observed in diagnostic test:
  Coral/Algae
  Microalgal Mats
  Rock
  Rubble
  Sand
  Seagrass

Method:
1. Query WFS separately for each site using a bounding box around the 70-km buffer.
2. Paginate with count=5000 and startIndex so no features are truncated.
3. Read all returned polygons with GeoPandas.
4. Project each site to an appropriate local UTM CRS.
5. Construct an exact 70-km circular buffer.
6. Clip habitat polygons to the buffer.
7. Calculate clipped area (km²) by class.
8. Calculate percent of mapped benthic habitat represented by each class.
9. Calculate candidate S. gracilis substrate metrics:
      Sand
      Rubble
      Coral/Algae
      Seagrass
   individually, plus exploratory combinations.

IMPORTANT:
- "Mapped benthic habitat" is not the same as total marine area.
- Unmapped cells must NOT be interpreted as unsuitable habitat.
- Allen Coral Atlas benthic mapping is a modern satellite-derived snapshot.
- Composite spawning-substrate metrics are exploratory only.

Requires:
    conda install -c conda-forge geopandas pyogrio shapely pandas requests -y
"""

from pathlib import Path
import json
import math
import time
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point

ROOT = Path(r"D:\temporal_environmental_data")

RAW_DIR = ROOT / "01_raw" / "Allen_Coral_Atlas_WFS"
OUT_DIR = ROOT / "03_extracted"
TABLE_DIR = ROOT / "04_tables"
FIG_DIR = ROOT / "05_figures"

for p in (RAW_DIR, OUT_DIR, TABLE_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://allencoralatlas.org/geoserver/ows"
LAYER = "coral-atlas:benthic_data_verbose"

SITES = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}

BUFFER_KM = 70
PAGE_SIZE = 5000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(academic environmental analysis; "
        "Allen Coral Atlas WFS)"
    )
}


def utm_epsg(lat, lon):
    """
    Return a WGS84 UTM EPSG code appropriate for a point.
    """
    zone = int((lon + 180) // 6) + 1

    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def km_to_lat_deg(km):
    return km / 110.574


def km_to_lon_deg(km, lat):
    return km / (111.320 * math.cos(math.radians(lat)))


def request_page(bbox, start_index, attempts=4):
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": bbox,
        "count": PAGE_SIZE,
        "startIndex": start_index,
    }

    last = None

    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=240,
                allow_redirects=True,
            )

            r.raise_for_status()
            return r.json()

        except Exception as exc:
            last = exc
            print(
                f"    page startIndex={start_index}, "
                f"attempt {attempt}/{attempts} failed: {exc}"
            )

            if attempt < attempts:
                time.sleep(3 * attempt)

    raise RuntimeError(
        f"WFS request failed after {attempts} attempts: {last}"
    )


def fetch_all_features(site, lat, lon):
    """
    Fetch all benthic polygons intersecting a generous lon/lat bbox
    around the 70-km site buffer.
    """

    # Slightly larger than 70 km to ensure full coverage before clipping.
    margin_km = BUFFER_KM + 5

    dlat = km_to_lat_deg(margin_km)
    dlon = km_to_lon_deg(margin_km, lat)

    west = lon - dlon
    east = lon + dlon
    south = lat - dlat
    north = lat + dlat

    bbox = (
        f"{west:.6f},{south:.6f},"
        f"{east:.6f},{north:.6f},EPSG:4326"
    )

    print(f"\n{site}")
    print(f"  WFS bbox: {bbox}")

    all_features = []
    start_index = 0
    page_num = 1

    while True:
        print(
            f"  Page {page_num}: "
            f"startIndex={start_index}"
        )

        payload = request_page(
            bbox,
            start_index,
        )

        features = payload.get(
            "features",
            [],
        )

        print(
            f"    returned {len(features)} features"
        )

        all_features.extend(
            features
        )

        if len(features) < PAGE_SIZE:
            break

        start_index += PAGE_SIZE
        page_num += 1

        # Hard safety ceiling.
        if page_num > 100:
            raise RuntimeError(
                f"More than 500,000 features requested for {site}; "
                "aborting as a safety limit."
            )

    print(
        f"  Total WFS features retrieved: "
        f"{len(all_features)}"
    )

    raw_geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    raw_path = (
        RAW_DIR
        / f"{site}_Allen_Coral_Atlas_benthic_bbox.geojson"
    )

    raw_path.write_text(
        json.dumps(
            raw_geojson,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return all_features, raw_path


def features_to_gdf(features):
    records = []

    for f in features:
        geom = f.get("geometry")
        props = f.get(
            "properties",
            {},
        )

        if geom is None:
            continue

        class_name = props.get(
            "class_name"
        )

        records.append(
            {
                "class_name": class_name,
                "source_area_sqkm": props.get(
                    "area_sqkm"
                ),
                "geometry": shape(geom),
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "class_name",
                "source_area_sqkm",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )


summary_rows = []
long_rows = []
coverage_rows = []

print("=" * 78)
print("ALLEN CORAL ATLAS — 70-km BENTHIC HABITAT EXTRACTION")
print("=" * 78)

for site, (lat, lon) in SITES.items():

    features, raw_path = fetch_all_features(
        site,
        lat,
        lon,
    )

    gdf = features_to_gdf(
        features
    )

    if gdf.empty:
        print(
            f"  WARNING: no habitat polygons returned for {site}"
        )
        continue

    print(
        f"  Geometry records after conversion: {len(gdf)}"
    )

    # Drop missing/invalid class labels.
    gdf = gdf[
        gdf["class_name"].notna()
    ].copy()

    # Repair minor geometry issues.
    gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[
        ~gdf.geometry.is_empty
        & gdf.geometry.notna()
    ].copy()

    epsg = utm_epsg(
        lat,
        lon,
    )

    print(
        f"  Local projected CRS: EPSG:{epsg}"
    )

    gdf_utm = gdf.to_crs(
        epsg=epsg
    )

    site_point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs="EPSG:4326",
    ).to_crs(
        epsg=epsg
    ).iloc[0]

    buffer_geom = site_point.buffer(
        BUFFER_KM * 1000
    )

    # Keep only polygons intersecting exact 70-km buffer.
    intersects = gdf_utm.intersects(
        buffer_geom
    )

    clipped = gdf_utm[
        intersects
    ].copy()

    if clipped.empty:
        print(
            f"  WARNING: no Atlas polygons intersect exact "
            f"{BUFFER_KM}-km buffer."
        )
        continue

    clipped["geometry"] = clipped.geometry.intersection(
        buffer_geom
    )

    clipped = clipped[
        ~clipped.geometry.is_empty
    ].copy()

    clipped["clipped_area_km2"] = (
        clipped.geometry.area
        / 1_000_000.0
    )

    # Remove zero-area slivers.
    clipped = clipped[
        clipped["clipped_area_km2"] > 0
    ].copy()

    class_area = (
        clipped
        .groupby(
            "class_name",
            as_index=False,
        )["clipped_area_km2"]
        .sum()
        .sort_values(
            "clipped_area_km2",
            ascending=False,
        )
    )

    total_mapped = float(
        class_area[
            "clipped_area_km2"
        ].sum()
    )

    class_area[
        "percent_of_mapped_benthic_area"
    ] = (
        class_area[
            "clipped_area_km2"
        ]
        / total_mapped
        * 100.0
        if total_mapped > 0
        else float("nan")
    )

    print(
        f"  Total mapped benthic area inside "
        f"{BUFFER_KM} km: {total_mapped:.2f} km²"
    )

    print(
        "  Habitat classes:"
    )

    for row in class_area.itertuples(
        index=False
    ):
        print(
            f"    {row.class_name}: "
            f"{row.clipped_area_km2:.2f} km² "
            f"({row.percent_of_mapped_benthic_area:.1f}%)"
        )

    for row in class_area.itertuples(
        index=False
    ):
        long_rows.append(
            {
                "site": site,
                "latitude": lat,
                "longitude": lon,
                "buffer_km": BUFFER_KM,
                "class_name": row.class_name,
                "area_km2": row.clipped_area_km2,
                "percent_of_mapped_benthic_area":
                    row.percent_of_mapped_benthic_area,
                "dataset":
                    "Allen Coral Atlas benthic_data_verbose",
                "data_class":
                    "modern_satellite_derived_benthic_habitat_map",
                "temporal_status":
                    "modern_snapshot",
            }
        )

    area_lookup = dict(
        zip(
            class_area["class_name"],
            class_area["clipped_area_km2"],
        )
    )

    def A(name):
        return float(
            area_lookup.get(
                name,
                0.0,
            )
        )

    # Keep each biologically relevant class separately.
    sand = A("Sand")
    rubble = A("Rubble")
    coral_algae = A("Coral/Algae")
    seagrass = A("Seagrass")
    rock = A("Rock")
    microalgal = A("Microalgal Mats")

    # Exploratory composite combinations only.
    sand_rubble = (
        sand
        + rubble
    )

    sand_rubble_algae = (
        sand
        + rubble
        + coral_algae
    )

    broad_candidate = (
        sand
        + rubble
        + coral_algae
        + seagrass
    )

    summary_rows.append(
        {
            "site": site,
            "latitude": lat,
            "longitude": lon,
            "buffer_km": BUFFER_KM,

            "mapped_benthic_area_km2":
                total_mapped,

            "sand_km2":
                sand,

            "rubble_km2":
                rubble,

            "coral_algae_km2":
                coral_algae,

            "seagrass_km2":
                seagrass,

            "rock_km2":
                rock,

            "microalgal_mats_km2":
                microalgal,

            "sand_plus_rubble_km2":
                sand_rubble,

            "sand_plus_rubble_plus_coral_algae_km2":
                sand_rubble_algae,

            "broad_candidate_spawning_substrate_km2":
                broad_candidate,

            "sand_percent_mapped":
                sand / total_mapped * 100
                if total_mapped > 0
                else float("nan"),

            "rubble_percent_mapped":
                rubble / total_mapped * 100
                if total_mapped > 0
                else float("nan"),

            "coral_algae_percent_mapped":
                coral_algae / total_mapped * 100
                if total_mapped > 0
                else float("nan"),

            "seagrass_percent_mapped":
                seagrass / total_mapped * 100
                if total_mapped > 0
                else float("nan"),

            "sand_rubble_percent_mapped":
                sand_rubble / total_mapped * 100
                if total_mapped > 0
                else float("nan"),

            "broad_candidate_percent_mapped":
                broad_candidate / total_mapped * 100
                if total_mapped > 0
                else float("nan"),

            "dataset":
                "Allen Coral Atlas benthic_data_verbose",

            "data_class":
                "modern_satellite_derived_benthic_habitat_map",

            "temporal_status":
                "modern_snapshot",

            "composite_status":
                "exploratory_not_validated",
        }
    )

    coverage_rows.append(
        {
            "site": site,
            "buffer_km": BUFFER_KM,
            "wfs_features_retrieved":
                len(features),
            "features_intersecting_buffer":
                len(clipped),
            "mapped_benthic_area_km2":
                total_mapped,
            "utm_epsg":
                epsg,
            "raw_geojson":
                str(raw_path),
        }
    )

    # Save clipped site polygons as GeoPackage.
    gpkg = (
        RAW_DIR
        / f"{site}_Allen_Coral_Atlas_benthic_70km.gpkg"
    )

    clipped.to_crs(
        "EPSG:4326"
    ).to_file(
        gpkg,
        layer="benthic_70km",
        driver="GPKG",
    )

    print(
        f"  Saved clipped habitat polygons:\n    {gpkg}"
    )


long_df = pd.DataFrame(
    long_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

coverage_df = pd.DataFrame(
    coverage_rows
)

long_csv = (
    OUT_DIR
    / "allen_coral_atlas_benthic_classes_70km.csv"
)

summary_csv = (
    TABLE_DIR
    / "allen_coral_atlas_benthic_70km_summary.csv"
)

coverage_csv = (
    TABLE_DIR
    / "allen_coral_atlas_benthic_WFS_coverage_70km.csv"
)

long_df.to_csv(
    long_csv,
    index=False,
)

summary_df.to_csv(
    summary_csv,
    index=False,
)

coverage_df.to_csv(
    coverage_csv,
    index=False,
)


print()
print("=" * 78)
print("BENTHIC EXTRACTION COMPLETE")
print("=" * 78)

print(
    f"Long habitat table:\n  {long_csv}"
)

print(
    f"Site summary:\n  {summary_csv}"
)

print(
    f"WFS coverage/provenance:\n  {coverage_csv}"
)

print()
print("SUMMARY")
print("-" * 78)

if not summary_df.empty:
    show_cols = [
        "site",
        "mapped_benthic_area_km2",
        "sand_km2",
        "rubble_km2",
        "coral_algae_km2",
        "seagrass_km2",
        "sand_plus_rubble_km2",
        "broad_candidate_spawning_substrate_km2",
        "broad_candidate_percent_mapped",
    ]

    print(
        summary_df[
            show_cols
        ].to_string(
            index=False
        )
    )

print()
print(
    "Interpretation warning:"
)

print(
    "  Percentages are relative to Allen Coral Atlas MAPPED benthic area "
    "within 70 km, not to all marine area."
)

print(
    "  Unmapped marine areas must not be counted as unsuitable habitat."
)
