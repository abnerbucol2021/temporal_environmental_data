
"""
11b_allen_coral_atlas_benthic_70km_TILED.py
===========================================

Robust Allen Coral Atlas benthic-habitat extraction for:
  Jolo
  Masinloc
  Palapag

Buffer:
  70 km

WFS layer:
  coral-atlas:benthic_data_verbose

WHY THIS VERSION
----------------
Allen Coral Atlas WFS accepted GetFeature with count=5000 in the diagnostic
test, but rejected requests containing startIndex. Therefore this script does
NOT use pagination.

Instead it uses adaptive spatial tiling:
1. Divide each site's 70-km search area into geographic tiles.
2. Query each tile independently with count=5000.
3. If a tile returns exactly 5000 features, subdivide that tile recursively.
4. Deduplicate overlapping WFS features by feature ID.
5. Clip all polygons to an exact projected 70-km circular buffer.
6. Calculate area by benthic class.

Classes observed in the diagnostic test:
  Coral/Algae
  Microalgal Mats
  Rock
  Rubble
  Sand
  Seagrass

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
from shapely.geometry import shape, Point, box

ROOT = Path(r"D:\temporal_environmental_data")

RAW_DIR = ROOT / "01_raw" / "Allen_Coral_Atlas_WFS"
OUT_DIR = ROOT / "03_extracted"
TABLE_DIR = ROOT / "04_tables"

for p in (RAW_DIR, OUT_DIR, TABLE_DIR):
    p.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://allencoralatlas.org/geoserver/ows"
LAYER = "coral-atlas:benthic_data_verbose"

SITES = {
    "Jolo": (6.1098, 120.9775),
    "Masinloc": (15.55672, 119.93036),
    "Palapag": (12.63476, 125.00978),
}

BUFFER_KM = 70

# Match the successful diagnostic request.
COUNT_LIMIT = 5000

# Initial tile width/height in degrees.
# ~22 km in latitude; small enough to keep feature counts manageable.
INITIAL_TILE_DEG = 0.20

# Recursive subdivision safety.
MAX_SUBDIVISION_DEPTH = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(academic environmental analysis; Allen Coral Atlas WFS)"
    )
}


def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def km_to_lat_deg(km):
    return km / 110.574


def km_to_lon_deg(km, lat):
    return km / (111.320 * math.cos(math.radians(lat)))


def wfs_get_bbox(west, south, east, north, attempts=4):
    """
    Request one bbox using the exact pattern proven to work in the diagnostic.
    No startIndex.
    """
    bbox_text = (
        f"{west:.7f},{south:.7f},"
        f"{east:.7f},{north:.7f},EPSG:4326"
    )

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": bbox_text,
        "count": COUNT_LIMIT,
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

            if r.status_code != 200:
                # Save a short server message for diagnostics.
                snippet = r.text[:1000]
                raise RuntimeError(
                    f"HTTP {r.status_code}: {snippet}"
                )

            payload = r.json()
            return payload

        except Exception as exc:
            last = exc
            print(
                f"      attempt {attempt}/{attempts} failed: {exc}"
            )

            if attempt < attempts:
                time.sleep(3 * attempt)

    raise RuntimeError(
        f"WFS request failed after {attempts} attempts: {last}"
    )


def feature_key(feature):
    """
    Prefer stable WFS feature id. Fallback to geometry+properties signature.
    """
    fid = feature.get("id")

    if fid:
        return ("id", fid)

    geom = feature.get("geometry")
    props = feature.get("properties", {})

    return (
        "fallback",
        json.dumps(
            {"geometry": geom, "properties": props},
            sort_keys=True,
            default=str,
        ),
    )


def subdivide_bbox(west, south, east, north):
    midx = (west + east) / 2.0
    midy = (south + north) / 2.0

    return [
        (west, south, midx, midy),
        (midx, south, east, midy),
        (west, midy, midx, north),
        (midx, midy, east, north),
    ]


def fetch_tile_recursive(
    west,
    south,
    east,
    north,
    feature_dict,
    depth=0,
):
    """
    Request a tile. If result hits COUNT_LIMIT, recursively split it into 4.
    """
    payload = wfs_get_bbox(
        west,
        south,
        east,
        north,
    )

    features = payload.get(
        "features",
        [],
    )

    print(
        f"    depth={depth} "
        f"bbox=({west:.4f},{south:.4f},{east:.4f},{north:.4f}) "
        f"-> {len(features)} features"
    )

    # Hitting exactly the cap means the response may be truncated.
    if (
        len(features) >= COUNT_LIMIT
        and depth < MAX_SUBDIVISION_DEPTH
    ):
        print(
            "      feature cap reached; subdividing tile"
        )

        for child in subdivide_bbox(
            west,
            south,
            east,
            north,
        ):
            fetch_tile_recursive(
                *child,
                feature_dict=feature_dict,
                depth=depth + 1,
            )

        return

    if (
        len(features) >= COUNT_LIMIT
        and depth >= MAX_SUBDIVISION_DEPTH
    ):
        raise RuntimeError(
            "A smallest allowed tile still returned 5000 features. "
            "Increase MAX_SUBDIVISION_DEPTH or reduce INITIAL_TILE_DEG."
        )

    for feature in features:
        feature_dict[
            feature_key(feature)
        ] = feature


def build_initial_tiles(
    west,
    south,
    east,
    north,
    tile_deg=INITIAL_TILE_DEG,
):
    tiles = []

    y = south

    while y < north:
        y2 = min(
            y + tile_deg,
            north,
        )

        x = west

        while x < east:
            x2 = min(
                x + tile_deg,
                east,
            )

            tiles.append(
                (
                    x,
                    y,
                    x2,
                    y2,
                )
            )

            x = x2

        y = y2

    return tiles


def fetch_site_features(site, lat, lon):
    margin_km = BUFFER_KM + 5

    dlat = km_to_lat_deg(
        margin_km
    )

    dlon = km_to_lon_deg(
        margin_km,
        lat,
    )

    west = lon - dlon
    east = lon + dlon
    south = lat - dlat
    north = lat + dlat

    print(f"\n{site}")
    print(
        f"  search bbox: "
        f"{west:.6f},{south:.6f},"
        f"{east:.6f},{north:.6f}"
    )

    tiles = build_initial_tiles(
        west,
        south,
        east,
        north,
    )

    print(
        f"  initial tiles: {len(tiles)} "
        f"({INITIAL_TILE_DEG}° grid)"
    )

    feature_dict = {}

    for i, tile in enumerate(
        tiles,
        start=1,
    ):
        print(
            f"  Tile {i}/{len(tiles)}"
        )

        fetch_tile_recursive(
            *tile,
            feature_dict=feature_dict,
            depth=0,
        )

    features = list(
        feature_dict.values()
    )

    print(
        f"  Unique features after deduplication: "
        f"{len(features)}"
    )

    raw_geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    raw_path = (
        RAW_DIR
        / f"{site}_Allen_Coral_Atlas_benthic_tiled.geojson"
    )

    raw_path.write_text(
        json.dumps(
            raw_geojson,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return features, raw_path


def features_to_gdf(features):
    records = []

    for f in features:
        geom = f.get(
            "geometry"
        )

        props = f.get(
            "properties",
            {},
        )

        if geom is None:
            continue

        records.append(
            {
                "feature_id":
                    f.get("id"),

                "class_name":
                    props.get("class_name"),

                "source_area_sqkm":
                    props.get("area_sqkm"),

                "geometry":
                    shape(geom),
            }
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
print("ALLEN CORAL ATLAS — TILED 70-km BENTHIC EXTRACTION")
print("=" * 78)


for site, (
    lat,
    lon,
) in SITES.items():

    features, raw_path = fetch_site_features(
        site,
        lat,
        lon,
    )

    gdf = features_to_gdf(
        features
    )

    if gdf.empty:
        print(
            f"  WARNING: no features returned for {site}"
        )
        continue

    print(
        f"  Features converted to geometry: {len(gdf)}"
    )

    gdf = gdf[
        gdf["class_name"].notna()
    ].copy()

    # Repair invalid geometry robustly.
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    epsg = utm_epsg(
        lat,
        lon,
    )

    print(
        f"  projected CRS: EPSG:{epsg}"
    )

    gdf_utm = gdf.to_crs(
        epsg=epsg
    )

    site_point = (
        gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326",
        )
        .to_crs(
            epsg=epsg
        )
        .iloc[0]
    )

    buffer_geom = site_point.buffer(
        BUFFER_KM * 1000
    )

    # Bounding spatial filter first.
    minx, miny, maxx, maxy = (
        buffer_geom.bounds
    )

    candidates = gdf_utm.cx[
        minx:maxx,
        miny:maxy,
    ].copy()

    candidates = candidates[
        candidates.intersects(
            buffer_geom
        )
    ].copy()

    print(
        f"  polygons intersecting exact 70-km buffer: "
        f"{len(candidates)}"
    )

    if candidates.empty:
        print(
            "  WARNING: no mapped benthic polygons intersect buffer."
        )
        continue

    candidates[
        "geometry"
    ] = candidates.geometry.intersection(
        buffer_geom
    )

    candidates = candidates[
        ~candidates.geometry.is_empty
    ].copy()

    candidates[
        "clipped_area_km2"
    ] = (
        candidates.geometry.area
        / 1_000_000.0
    )

    candidates = candidates[
        candidates[
            "clipped_area_km2"
        ] > 0
    ].copy()

    class_area = (
        candidates
        .groupby(
            "class_name",
            as_index=False,
        )[
            "clipped_area_km2"
        ]
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
    )

    print(
        f"  Total mapped benthic area inside 70 km: "
        f"{total_mapped:.2f} km²"
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
                "class_name":
                    row.class_name,
                "area_km2":
                    row.clipped_area_km2,
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

    lookup = dict(
        zip(
            class_area[
                "class_name"
            ],
            class_area[
                "clipped_area_km2"
            ],
        )
    )

    def A(name):
        return float(
            lookup.get(
                name,
                0.0,
            )
        )

    sand = A(
        "Sand"
    )

    rubble = A(
        "Rubble"
    )

    coral_algae = A(
        "Coral/Algae"
    )

    seagrass = A(
        "Seagrass"
    )

    rock = A(
        "Rock"
    )

    microalgal = A(
        "Microalgal Mats"
    )

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
            "site":
                site,

            "latitude":
                lat,

            "longitude":
                lon,

            "buffer_km":
                BUFFER_KM,

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
            "site":
                site,

            "buffer_km":
                BUFFER_KM,

            "unique_wfs_features_retrieved":
                len(features),

            "features_intersecting_buffer":
                len(candidates),

            "mapped_benthic_area_km2":
                total_mapped,

            "utm_epsg":
                epsg,

            "raw_geojson":
                str(raw_path),

            "retrieval_method":
                "adaptive_spatial_tiling_no_startIndex",
        }
    )

    # Save clipped polygons.
    gpkg = (
        RAW_DIR
        / f"{site}_Allen_Coral_Atlas_benthic_70km.gpkg"
    )

    candidates.to_crs(
        "EPSG:4326"
    ).to_file(
        gpkg,
        layer="benthic_70km",
        driver="GPKG",
    )

    print(
        f"  Saved clipped polygons:\n    {gpkg}"
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
    "  Percentages are relative to Allen Coral Atlas MAPPED benthic "
    "area inside 70 km, not to total marine area."
)

print(
    "  Unmapped/deeper/turbid marine areas must not be interpreted "
    "as unsuitable habitat."
)
