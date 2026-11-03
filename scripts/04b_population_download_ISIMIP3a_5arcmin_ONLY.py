"""
04b_population_download_ISIMIP3a_5arcmin_ONLY.py
================================================

Download a clean Philippines spatial subset using ONLY the two intended
ISIMIP3a historical-society (histsoc) 5-arcmin population files.

This avoids mixing:
  - ISIMIP2a
  - ISIMIP3b
  - 15/30 arcmin products
  - 1901soc / 2015soc alternatives

Requires:
    conda activate temporal_env
    pip/conda package: isimip-client

Outputs:
    D:\temporal_environmental_data\01_raw\HYDE_population\
        population_ISIMIP3a_histsoc_5arcmin_Philippines.zip
        ISIMIP3a_histsoc_5arcmin_clean\
"""

from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(r"D:\temporal_environmental_data")
OUT_DIR = ROOT / "01_raw" / "HYDE_population"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_ZIP = OUT_DIR / "population_ISIMIP3a_histsoc_5arcmin_Philippines.zip"
CLEAN_DIR = OUT_DIR / "ISIMIP3a_histsoc_5arcmin_clean"

PATHS = [
    (
        "ISIMIP3a/InputData/socioeconomic/pop/histsoc/"
        "population_histsoc_5arcmin_annual_1850_1900.nc"
    ),
    (
        "ISIMIP3a/InputData/socioeconomic/pop/histsoc/"
        "population_histsoc_5arcmin_annual_1901_2021.nc"
    ),
]

WEST, EAST, SOUTH, NORTH = 118.0, 126.5, 5.0, 17.0

try:
    from isimip_client.client import ISIMIPClient
except ImportError:
    print("ERROR: isimip-client is not installed.")
    sys.exit(1)

print("=" * 78)
print("ISIMIP3a histsoc population — CLEAN 5-ARCMIN CUTOUT")
print("=" * 78)
print()
print("Exact repository paths:")
for p in PATHS:
    print(" ", p)

print()
print(
    f"Bounding box: west={WEST}, east={EAST}, "
    f"south={SOUTH}, north={NORTH}"
)

client = ISIMIPClient()

print()
print("Submitting clean cutout job...")
response = client.cutout_bbox(
    PATHS,
    west=WEST,
    east=EAST,
    south=SOUTH,
    north=NORTH,
    poll=4,
)

if not isinstance(response, dict):
    raise RuntimeError(
        f"Unexpected cutout response: {response!r}"
    )

file_url = response.get("file_url")
if not file_url:
    raise RuntimeError(
        f"No file_url returned by ISIMIP. Response: {response}"
    )

print()
print("Cutout completed.")
print("Server archive:", response.get("file_name", "(unnamed)"))

# Track ZIPs before/after download.
before = {
    p.resolve(): p.stat().st_mtime
    for p in OUT_DIR.glob("*.zip")
}

client.download(file_url, path=OUT_DIR)

after = list(OUT_DIR.glob("*.zip"))
if not after:
    raise RuntimeError("Download completed but no ZIP was found.")

# Prefer newly created/updated ZIP.
candidates = []
for p in after:
    rp = p.resolve()
    old_mtime = before.get(rp)
    if old_mtime is None or p.stat().st_mtime > old_mtime:
        candidates.append(p)

if not candidates:
    candidates = after

downloaded_zip = max(candidates, key=lambda p: p.stat().st_mtime)

if downloaded_zip.resolve() != FINAL_ZIP.resolve():
    FINAL_ZIP.unlink(missing_ok=True)
    downloaded_zip.replace(FINAL_ZIP)

print(f"Downloaded clean archive: {FINAL_ZIP}")
print(f"Size: {FINAL_ZIP.stat().st_size / 1024 / 1024:.2f} MB")

print()
print("Validating archive...")
with zipfile.ZipFile(FINAL_ZIP, "r") as z:
    bad = z.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupted ZIP member: {bad}")

    members = [
        info for info in z.infolist()
        if info.filename.lower().endswith(".nc")
    ]

    print("NetCDF members:")
    for info in members:
        print(
            f"  {info.filename} "
            f"({info.file_size / 1024 / 1024:.2f} MB, CRC={info.CRC:08x})"
        )

    if len(members) != 2:
        raise RuntimeError(
            f"Expected exactly 2 NetCDF files, found {len(members)}."
        )

    expected_tokens = ["1850_1900", "1901_2021"]

    for token in expected_tokens:
        matches = [
            m for m in members
            if token in Path(m.filename).name
            and "population_histsoc_5arcmin_annual" in Path(m.filename).name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one clean {token} file; found {len(matches)}."
            )

    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for info in members:
        dest = CLEAN_DIR / Path(info.filename).name
        with z.open(info, "r") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"Extracted: {dest}")

print()
print("Validating NetCDF structure with xarray...")

import xarray as xr

for nc in sorted(CLEAN_DIR.glob("*.nc")):
    print()
    print(nc.name)

    with xr.open_dataset(nc, decode_cf=True) as ds:
        print("  dimensions:", dict(ds.sizes))
        print("  variables:", list(ds.data_vars))
        print("  coordinates:", list(ds.coords))

        lat_name = "lat" if "lat" in ds.coords else (
            "latitude" if "latitude" in ds.coords else None
        )
        lon_name = "lon" if "lon" in ds.coords else (
            "longitude" if "longitude" in ds.coords else None
        )

        if lat_name:
            print(
                f"  latitude: {float(ds[lat_name].min()):.4f} "
                f"to {float(ds[lat_name].max()):.4f}"
            )

        if lon_name:
            print(
                f"  longitude: {float(ds[lon_name].min()):.4f} "
                f"to {float(ds[lon_name].max()):.4f}"
            )

        if "time" in ds.coords and ds.sizes.get("time", 0):
            try:
                years = ds["time"].dt.year.values
                print(f"  years: {int(years.min())} to {int(years.max())}")
            except Exception:
                print(
                    "  time:",
                    str(ds["time"].values[0]),
                    "to",
                    str(ds["time"].values[-1]),
                )

print()
print("=" * 78)
print("CLEAN DOWNLOAD COMPLETE")
print("=" * 78)
print(f"Archive: {FINAL_ZIP}")
print(f"Clean folder: {CLEAN_DIR}")
print()
print("Next run:")
print(
    r"python D:\temporal_environmental_data\scripts"
    r"\05_population_extract_5arcmin_CLEAN.py"
)
