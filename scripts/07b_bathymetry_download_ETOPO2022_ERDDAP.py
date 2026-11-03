
"""
07b_bathymetry_download_ETOPO2022_ERDDAP.py
===========================================

Download a Philippines spatial subset of NOAA/NCEI ETOPO 2022
15 arc-second relief through NOAA PIFSC ERDDAP.

Bounding box:
  latitude  5.0 to 17.0 N
  longitude 118.0 to 126.5 E

This covers all 70-km buffers around:
  Jolo
  Masinloc
  Palapag

Output:
  D:\temporal_environmental_data\01_raw\ETOPO2022_bathymetry\
  ETOPO2022_15s_Philippines_118_126p5E_5_17N.nc

Note:
The ERDDAP dataset is the ETOPO 2022 "Ice Surface" version. In the Philippines
there are no ice sheets, so it is equivalent to bed elevation for this study area.
"""

from pathlib import Path
import time
import requests
import xarray as xr

ROOT = Path(r"D:\temporal_environmental_data")
OUT_DIR = ROOT / "01_raw" / "ETOPO2022_bathymetry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "ETOPO2022_15s_Philippines_118_126p5E_5_17N.nc"
PART_FILE = OUT_FILE.with_suffix(".nc.part")

BASE = (
    "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/"
    "ETOPO_2022_v1_15s.nc"
)

# ERDDAP query order for z is [latitude][longitude].
# Longitude is 0..360; our study area is already positive east longitude.
QUERY = "z[(5.0):(17.0)][(118.0):(126.5)]"

URL = f"{BASE}?{QUERY}"

def mb(n):
    return n / 1024 / 1024

def download(attempts=5):
    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 1_000_000:
        print(
            f"Using existing file: {OUT_FILE} "
            f"({mb(OUT_FILE.stat().st_size):.2f} MB)"
        )
        return

    for attempt in range(1, attempts + 1):
        try:
            PART_FILE.unlink(missing_ok=True)

            print(f"Downloading ETOPO subset (attempt {attempt}/{attempts})")
            print(URL)

            with requests.get(
                URL,
                stream=True,
                timeout=(30, 300),
                headers={"User-Agent": "Mozilla/5.0"},
            ) as r:
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0))
                print("Content type:", r.headers.get("Content-Type", "unknown"))

                if total:
                    print(f"Expected size: {mb(total):.2f} MB")

                got = 0
                with open(PART_FILE, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)

                            if total:
                                print(
                                    f"\rDownloaded {mb(got):.2f} MB "
                                    f"({100 * got / total:.1f}%)",
                                    end="",
                                    flush=True,
                                )

            print()

            # Validate NetCDF before promoting.
            with xr.open_dataset(PART_FILE, decode_times=False) as ds:
                if "z" not in ds.data_vars:
                    raise RuntimeError(
                        f"Expected variable 'z' not found. "
                        f"Variables: {list(ds.data_vars)}"
                    )

                print("\nSubset validation:")
                print(" dimensions:", dict(ds.sizes))
                print(" variables:", list(ds.data_vars))
                print(" coordinates:", list(ds.coords))

                lat_name = (
                    "latitude" if "latitude" in ds.coords
                    else "lat" if "lat" in ds.coords
                    else None
                )

                lon_name = (
                    "longitude" if "longitude" in ds.coords
                    else "lon" if "lon" in ds.coords
                    else None
                )

                if lat_name is None or lon_name is None:
                    raise RuntimeError(
                        "Could not identify latitude/longitude coordinates."
                    )

                print(
                    f" latitude: "
                    f"{float(ds[lat_name].min()):.6f} to "
                    f"{float(ds[lat_name].max()):.6f}"
                )

                print(
                    f" longitude: "
                    f"{float(ds[lon_name].min()):.6f} to "
                    f"{float(ds[lon_name].max()):.6f}"
                )

                print(
                    " z units:",
                    ds["z"].attrs.get("units", ""),
                )

                print(
                    " z range:",
                    float(ds["z"].min()),
                    "to",
                    float(ds["z"].max()),
                )

            PART_FILE.replace(OUT_FILE)

            print(f"\nSaved validated subset:\n  {OUT_FILE}")
            print(f"Size: {mb(OUT_FILE.stat().st_size):.2f} MB")
            return

        except Exception as exc:
            print(f"Attempt {attempt} failed: {exc}")
            PART_FILE.unlink(missing_ok=True)

            if attempt == attempts:
                raise

            time.sleep(3 * attempt)

print("=" * 78)
print("NOAA ETOPO 2022 — 15 ARC-SECOND ERDDAP PHILIPPINES SUBSET")
print("=" * 78)

download()

print("\nDOWNLOAD COMPLETE")
print(OUT_FILE)
