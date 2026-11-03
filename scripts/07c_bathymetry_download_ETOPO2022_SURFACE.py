
from pathlib import Path
import time
import requests
import rasterio

ROOT = Path(r"D:\temporal_environmental_data")
OUT = ROOT / "01_raw" / "ETOPO2022_bathymetry"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/"
    "data/15s/15s_surface_elev_gtif"
)

TILES = [
    "ETOPO_2022_v1_15s_N15E105_surface.tif",
    "ETOPO_2022_v1_15s_N15E120_surface.tif",
    "ETOPO_2022_v1_15s_N30E105_surface.tif",
    "ETOPO_2022_v1_15s_N30E120_surface.tif",
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
            print(url)

            with requests.get(
                url,
                stream=True,
                timeout=(30,300),
                headers={"User-Agent":"Mozilla/5.0"},
                allow_redirects=True,
            ) as r:
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0))
                print("Content type:", r.headers.get("Content-Type","unknown"))

                if total:
                    print(f"Expected size: {mb(total):.2f} MB")

                got = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)
                            if total:
                                print(
                                    f"\rDownloaded {mb(got):.2f} MB "
                                    f"({100*got/total:.1f}%)",
                                    end="",
                                    flush=True
                                )
            print()

            with open(part, "rb") as f:
                header = f.read(4)

            if header not in (b"II*\x00", b"MM\x00*"):
                raise IOError(f"Not a TIFF: {header!r}")

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
print("NOAA ETOPO 2022 — VERIFIED 15 ARC-SECOND SURFACE TILES")
print("="*78)

for t in TILES:
    download(f"{BASE}/{t}", OUT/t)

print("\nValidating footprints...")

for t in TILES:
    p = OUT/t
    with rasterio.open(p) as src:
        print(f"\n{t}")
        print("  CRS:", src.crs)
        print("  bounds:", src.bounds)
        print("  resolution:", src.res)
        print("  size:", src.width, "x", src.height)

print("\nDOWNLOAD COMPLETE")
print(OUT)
