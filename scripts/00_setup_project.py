from pathlib import Path

ROOT = Path(r"D:\temporal_environmental_data")

folders = [
    "00_metadata",
    r"01_raw\ERSST",
    r"01_raw\EN4",
    r"01_raw\HYDE_population",
    r"01_raw\LUH2_landuse",
    r"01_raw\Copernicus_SPM",
    r"01_raw\CMIP6",
    r"02_processed\ERSST",
    r"02_processed\EN4",
    r"02_processed\population",
    r"02_processed\landuse",
    r"02_processed\sediment_proxy",
    "03_extracted",
    "04_tables",
    "05_figures",
    "06_statistics",
    "07_logs",
    "scripts",
]

for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

sites_csv = ROOT / "00_metadata" / "sites.csv"
if not sites_csv.exists():
    sites_csv.write_text(
        "site,latitude,longitude\n"
        "Jolo,6.1098,120.9775\n"
        "Masinloc,15.55672,119.93036\n"
        "Palapag,12.63476,125.00978\n",
        encoding="utf-8",
    )

print(f"Project created at: {ROOT}")
print(f"Site file: {sites_csv}")
