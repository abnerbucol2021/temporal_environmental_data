
from pathlib import Path
import time
import requests

ROOT = Path(r"D:\temporal_environmental_data")
OUT = ROOT / "01_raw" / "ETOPO2022_bathymetry"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/15s_bed_elev_gtif"

TILES = [
    "ETOPO_2022_v1_15s_N00E105_bed.tif",
    "ETOPO_2022_v1_15s_N00E120_bed.tif",
    "ETOPO_2022_v1_15s_N15E105_bed.tif",
    "ETOPO_2022_v1_15s_N15E120_bed.tif",
]

def mb(n):
    return n / 1024 / 1024

def download(url, dest, attempts=5):
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already present: {dest.name} ({mb(dest.stat().st_size):.2f} MB)")
        return
    for attempt in range(1, attempts + 1):
        try:
            part.unlink(missing_ok=True)
            print(f"\nDownloading {dest.name} (attempt {attempt}/{attempts})")
            with requests.get(url, stream=True, timeout=(30,300),
                              headers={"User-Agent":"Mozilla/5.0"}) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                if total:
                    print(f"Expected size: {mb(total):.2f} MB")
                got = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)
                            if total:
                                print(f"\rDownloaded {mb(got):.2f} MB ({100*got/total:.1f}%)",
                                      end="", flush=True)
            print()
            with open(part, "rb") as f:
                header = f.read(4)
            if header not in (b"II*\x00", b"MM\x00*"):
                raise IOError(f"File is not a TIFF: {header!r}")
            part.replace(dest)
            print(f"Saved: {dest}")
            return
        except Exception as exc:
            print(f"Attempt {attempt} failed: {exc}")
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(3*attempt)

print("="*78)
print("NOAA ETOPO 2022 — 15 ARC-SECOND BATHYMETRY DOWNLOAD")
print("="*78)

for tile in TILES:
    download(f"{BASE}/{tile}", OUT / tile)

print("\nValidating rasters...")
import rasterio
for tile in TILES:
    p = OUT / tile
    with rasterio.open(p) as src:
        print(f"\n{tile}")
        print(" CRS:", src.crs)
        print(" size:", src.width, "x", src.height)
        print(" bounds:", src.bounds)
        print(" resolution:", src.res)

print("\nDOWNLOAD COMPLETE")
print(OUT)
