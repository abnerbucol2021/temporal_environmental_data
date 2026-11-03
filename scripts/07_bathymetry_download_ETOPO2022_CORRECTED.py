
from pathlib import Path
import time
import requests

ROOT = Path(r"D:\temporal_environmental_data")
OUT = ROOT / "01_raw" / "ETOPO2022_bathymetry"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/"
    "data/15s/15s_bed_elev_gtif"
)

# Correct ETOPO naming:
# latitude code = NORTH/TOP edge of each 15° tile.
#
# N15E105 covers  0–15°N, 105–120°E
# N15E120 covers  0–15°N, 120–135°E
# N30E105 covers 15–30°N, 105–120°E
# N30E120 covers 15–30°N, 120–135°E
#
# These four tiles fully cover 70-km buffers around:
# Jolo, Masinloc, and Palapag.
TILES = [
    "ETOPO_2022_v1_15s_N15E105_bed.tif",
    "ETOPO_2022_v1_15s_N15E120_bed.tif",
    "ETOPO_2022_v1_15s_N30E105_bed.tif",
    "ETOPO_2022_v1_15s_N30E120_bed.tif",
]


def mb(n):
    return n / 1024 / 1024


def download(url, dest, attempts=5):
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(
            f"Already present: {dest.name} "
            f"({mb(dest.stat().st_size):.2f} MB)"
        )
        return

    for attempt in range(1, attempts + 1):
        try:
            part.unlink(missing_ok=True)

            print(
                f"\nDownloading {dest.name} "
                f"(attempt {attempt}/{attempts})"
            )

            with requests.get(
                url,
                stream=True,
                timeout=(30, 300),
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            ) as r:

                r.raise_for_status()

                total = int(
                    r.headers.get("Content-Length", 0)
                )

                ctype = r.headers.get(
                    "Content-Type",
                    "unknown",
                )

                print("Content type:", ctype)

                if total:
                    print(
                        f"Expected size: "
                        f"{mb(total):.2f} MB"
                    )

                got = 0

                with open(part, "wb") as f:
                    for chunk in r.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)

                            if total:
                                print(
                                    f"\rDownloaded "
                                    f"{mb(got):.2f} MB "
                                    f"({100*got/total:.1f}%)",
                                    end="",
                                    flush=True,
                                )

            print()

            if total and part.stat().st_size != total:
                raise IOError(
                    "Incomplete download: "
                    f"{part.stat().st_size} "
                    f"of {total} bytes"
                )

            with open(part, "rb") as f:
                header = f.read(4)

            if header not in (
                b"II*\x00",
                b"MM\x00*",
            ):
                raise IOError(
                    "Downloaded file does not "
                    f"look like GeoTIFF: {header!r}"
                )

            part.replace(dest)

            print(f"Saved: {dest}")

            return

        except Exception as exc:

            print(
                f"Attempt {attempt} failed: "
                f"{exc}"
            )

            part.unlink(
                missing_ok=True
            )

            if attempt == attempts:
                raise

            time.sleep(
                3 * attempt
            )


print("=" * 78)
print(
    "NOAA ETOPO 2022 — "
    "CORRECTED 15 ARC-SECOND TILE DOWNLOAD"
)
print("=" * 78)

print("\nRequired tiles:")
for t in TILES:
    print(" ", t)

for tile in TILES:

    download(
        f"{BASE}/{tile}",
        OUT / tile,
    )


print("\nValidating raster footprints...")

import rasterio

expected_bounds = {
    "ETOPO_2022_v1_15s_N15E105_bed.tif":
        (105, 0, 120, 15),

    "ETOPO_2022_v1_15s_N15E120_bed.tif":
        (120, 0, 135, 15),

    "ETOPO_2022_v1_15s_N30E105_bed.tif":
        (105, 15, 120, 30),

    "ETOPO_2022_v1_15s_N30E120_bed.tif":
        (120, 15, 135, 30),
}


for tile in TILES:

    path = OUT / tile

    with rasterio.open(
        path
    ) as src:

        print(f"\n{tile}")

        print(
            "  CRS:",
            src.crs,
        )

        print(
            "  size:",
            src.width,
            "x",
            src.height,
        )

        print(
            "  bounds:",
            src.bounds,
        )

        print(
            "  resolution:",
            src.res,
        )

        exp = expected_bounds[
            tile
        ]

        b = src.bounds

        tol = 0.01

        if not (
            abs(b.left - exp[0]) < tol
            and abs(b.bottom - exp[1]) < tol
            and abs(b.right - exp[2]) < tol
            and abs(b.top - exp[3]) < tol
        ):
            raise RuntimeError(
                "Unexpected tile footprint "
                f"for {tile}: {b}"
            )


print("\n" + "=" * 78)
print("DOWNLOAD COMPLETE")
print("=" * 78)

print(
    "Bathymetry directory:"
)

print(
    f"  {OUT}"
)

print()
print(
    "Next run:"
)

print(
    r"python D:\temporal_environmental_data\scripts"
    r"\08_spawning_habitat_capacity_ETOPO2022.py"
)
