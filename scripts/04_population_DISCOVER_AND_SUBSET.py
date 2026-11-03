"""
Phase 3A — Robust ISIMIP historical population discovery + Philippines cutout
=============================================================================

This script DOES NOT hard-code the population repository directory.

It:
  1. searches the live ISIMIP Repository API for historical population files,
  2. prints the actual paths returned by the repository,
  3. automatically identifies the 30-arcmin annual historical population files
     covering 1900-2018,
  4. submits a spatial cutout around the three Philippine study sites,
  5. downloads and validates the resulting NetCDF subset.

Requires:
    pip install isimip-client

Output:
    D:\temporal_environmental_data\01_raw\HYDE_population
"""

from pathlib import Path
import json
import shutil
import sys
import zipfile

ROOT = Path(r"D:\temporal_environmental_data")
OUT_DIR = ROOT / "01_raw" / "HYDE_population"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEST, EAST, SOUTH, NORTH = 118.0, 126.5, 5.0, 17.0

try:
    from isimip_client.client import ISIMIPClient
except ImportError:
    print("ERROR: isimip-client is not installed.")
    print("Run: pip install isimip-client")
    sys.exit(1)


def unpack_results(response):
    """Handle both dict-style and list-style API responses."""
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        if "results" in response and isinstance(response["results"], list):
            return response["results"]
        return [response]
    return []


def extract_path(item):
    """Try common path fields used by repository responses."""
    if not isinstance(item, dict):
        return None

    for key in ("path", "file_path", "filepath"):
        val = item.get(key)
        if isinstance(val, str) and val:
            return val

    # Some responses can have directory + filename split.
    name = item.get("name") or item.get("filename")
    directory = item.get("directory")
    if isinstance(name, str) and isinstance(directory, str):
        return directory.rstrip("/") + "/" + name

    # Sometimes nested file metadata.
    f = item.get("file")
    if isinstance(f, dict):
        return extract_path(f)

    return None


client = ISIMIPClient()

print("=" * 76)
print("ISIMIP3a historical population — LIVE REPOSITORY DISCOVERY")
print("=" * 76)
print()

queries = [
    "population histsoc 30arcmin annual",
    "population_histsoc",
    "historical population 30arcmin",
    "population 30arcmin",
]

all_items = []
seen_serialized = set()

for q in queries:
    print(f"Searching repository: {q!r}")
    try:
        response = client.files(query=q, page_size=200)
    except TypeError:
        # Some client/API combinations may not accept page_size.
        response = client.files(query=q)

    items = unpack_results(response)
    print(f"  returned {len(items)} item(s)")

    for item in items:
        try:
            sig = json.dumps(item, sort_keys=True, default=str)
        except Exception:
            sig = repr(item)
        if sig not in seen_serialized:
            seen_serialized.add(sig)
            all_items.append(item)

print()
print(f"Unique repository records collected: {len(all_items)}")

paths = []
for item in all_items:
    p = extract_path(item)
    if p and p not in paths:
        paths.append(p)

# If records are dataset-like and contain a files list, collect nested file paths too.
for item in all_items:
    if isinstance(item, dict):
        nested = item.get("files")
        if isinstance(nested, list):
            for f in nested:
                p = extract_path(f)
                if p and p not in paths:
                    paths.append(p)

print()
print("Population-related paths returned by the live repository:")
population_paths = [
    p for p in paths
    if "pop" in p.lower() or "population" in p.lower()
]

for p in population_paths:
    print(" ", p)

# Score candidate files instead of assuming exact filenames.
def score_path(p):
    s = p.lower()
    score = 0
    if "population" in s:
        score += 10
    if "histsoc" in s:
        score += 10
    if "30arcmin" in s or "30_arcmin" in s or "30min" in s:
        score += 10
    if "annual" in s:
        score += 5
    if s.endswith(".nc"):
        score += 5
    return score


candidates = [
    p for p in population_paths
    if score_path(p) >= 25 and p.lower().endswith(".nc")
]

print()
print("Candidate historical 30-arcmin NetCDF files:")
for p in sorted(candidates):
    print(" ", p)

if not candidates:
    print()
    print("ERROR: No suitable NetCDF candidates were identified.")
    print("The repository search results above are the important diagnostic.")
    print("Please paste this output back into ChatGPT.")
    sys.exit(2)

# Select files whose names/ranges together cover 1900-2018.
# We prefer a small number of broad historical files.
selected = []

for p in candidates:
    s = p.lower()

    # Likely early historical file containing 1900.
    if any(x in s for x in (
        "1850_1900", "1850-1900",
        "1860_1900", "1860-1900",
        "1861_1900", "1861-1900"
    )):
        selected.append(p)

    # Likely later historical file containing 1901-2018.
    if any(x in s for x in (
        "1901_2020", "1901-2020",
        "1901_2021", "1901-2021",
        "1901_2022", "1901-2022",
        "1901_2100", "1901-2100"
    )):
        selected.append(p)

# Fallback: if the repository now uses one single broad file.
if not selected:
    for p in candidates:
        s = p.lower()
        if (
            ("1850" in s or "1860" in s or "1861" in s)
            and ("2020" in s or "2021" in s or "2022" in s)
        ):
            selected.append(p)

selected = list(dict.fromkeys(selected))

# If automatic date parsing is inconclusive, use all strong population candidates.
# This remains safe because the Files API will subset spatially.
if not selected:
    print()
    print("Could not infer date ranges from filenames; using all strong candidates.")
    selected = candidates

print()
print("Repository path(s) selected for the cutout:")
for p in selected:
    print(" ", p)

print()
print(
    f"Submitting bbox cutout: west={WEST}, east={EAST}, "
    f"south={SOUTH}, north={NORTH}"
)
print("Waiting for ISIMIP server job...")

response = client.cutout_bbox(
    selected,
    west=WEST,
    east=EAST,
    south=SOUTH,
    north=NORTH,
    poll=4,
)

if not isinstance(response, dict):
    print("Cutout response:", response)
    raise RuntimeError("ISIMIP did not return a completed cutout job.")

if response.get("status") not in (None, "finished"):
    print(json.dumps(response, indent=2, default=str))
    raise RuntimeError(
        f"ISIMIP cutout status was {response.get('status')!r}"
    )

file_url = response.get("file_url")
if not file_url:
    print(json.dumps(response, indent=2, default=str))
    raise RuntimeError("Cutout response contains no file_url.")

print()
print("Cutout complete.")
print("Generated archive:", response.get("file_name", "(server-generated ZIP)"))

before = {p.resolve() for p in OUT_DIR.glob("*.zip")}
client.download(file_url, path=OUT_DIR)
after = {p.resolve() for p in OUT_DIR.glob("*.zip")}

new_zips = [Path(p) for p in (after - before)]

if new_zips:
    zip_path = max(new_zips, key=lambda p: p.stat().st_mtime)
else:
    # If client overwrote/reused a name, take newest ZIP.
    zips = list(OUT_DIR.glob("*.zip"))
    if not zips:
        raise RuntimeError("Download completed but no ZIP was found.")
    zip_path = max(zips, key=lambda p: p.stat().st_mtime)

final_zip = OUT_DIR / "population_ISIMIP3a_Philippines_subset.zip"
if zip_path.resolve() != final_zip.resolve():
    final_zip.unlink(missing_ok=True)
    zip_path.replace(final_zip)

print(f"Downloaded: {final_zip}")
print(f"Size: {final_zip.stat().st_size / 1024 / 1024:.2f} MB")

print()
print("Validating ZIP...")
with zipfile.ZipFile(final_zip, "r") as z:
    bad = z.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupted ZIP member: {bad}")

    print("ZIP members:")
    for n in z.namelist():
        print(" ", n)

    extract_dir = OUT_DIR / "philippines_subset"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    z.extractall(extract_dir)

nc_files = sorted(extract_dir.rglob("*.nc"))

if not nc_files:
    raise RuntimeError("No NetCDF files found after extraction.")

print()
print("Validating NetCDF files...")

import xarray as xr

for nc in nc_files:
    print()
    print(nc.name)

    with xr.open_dataset(nc, decode_cf=True) as ds:
        print("  dimensions:", dict(ds.sizes))
        print("  variables:", list(ds.data_vars))
        print("  coordinates:", list(ds.coords))

        # Support either lat/lon or latitude/longitude.
        lat_name = "lat" if "lat" in ds.coords else (
            "latitude" if "latitude" in ds.coords else None
        )
        lon_name = "lon" if "lon" in ds.coords else (
            "longitude" if "longitude" in ds.coords else None
        )

        if lat_name:
            print(
                f"  latitude: {float(ds[lat_name].min()):.3f} "
                f"to {float(ds[lat_name].max()):.3f}"
            )
        if lon_name:
            print(
                f"  longitude: {float(ds[lon_name].min()):.3f} "
                f"to {float(ds[lon_name].max()):.3f}"
            )
        if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
            print(
                "  time:",
                str(ds.time.values[0])[:10],
                "to",
                str(ds.time.values[-1])[:10],
            )

print()
print("=" * 76)
print("DONE")
print("=" * 76)
print("Subset archive:", final_zip)
print("Extracted NetCDFs:", extract_dir)
print("Study period to use: 1900-2018")
print("data_class = historical_population_reconstruction")
