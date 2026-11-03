from pathlib import Path
import csv, json, sys, time, requests, xml.etree.ElementTree as ET

ROOT = Path(r"D:\temporal_environmental_data")
OUT = ROOT / "01_raw" / "Allen_Coral_Atlas_WFS"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://allencoralatlas.org/geoserver/ows"
CAP = OUT / "WFS_GetCapabilities.xml"
FTCSV = OUT / "WFS_feature_types.csv"
GJ = OUT / "Masinloc_WFS_test.geojson"

WEST, SOUTH, EAST, NORTH = 119.60, 15.20, 120.30, 15.90
HEADERS = {"User-Agent":"Mozilla/5.0"}

def get(params, attempts=4, timeout=180):
    last = None
    for i in range(1, attempts+1):
        try:
            r = requests.get(BASE, params=params, headers=HEADERS,
                             timeout=timeout, allow_redirects=True)
            print(f"HTTP {r.status_code} | {r.headers.get('Content-Type','unknown')}")
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            print(f"Attempt {i}/{attempts} failed: {e}")
            if i < attempts:
                time.sleep(3*i)
    raise RuntimeError(last)

print("="*78)
print("ALLEN CORAL ATLAS — WFS DIAGNOSTIC TEST")
print("="*78)

print("\nSTEP 1 — GetCapabilities")
r = get({"service":"WFS","version":"2.0.0","request":"GetCapabilities"}, timeout=120)
CAP.write_bytes(r.content)
print("Saved:", CAP)

print("\nSTEP 2 — Parse feature types")
root = ET.fromstring(r.content)
feature_types = []
for ft in root.iter():
    if ft.tag.endswith("FeatureType"):
        name = title = abstract = ""
        for ch in list(ft):
            if ch.tag.endswith("Name") and ch.text:
                name = ch.text.strip()
            elif ch.tag.endswith("Title") and ch.text:
                title = ch.text.strip()
            elif ch.tag.endswith("Abstract") and ch.text:
                abstract = ch.text.strip()
        if name:
            feature_types.append({"name":name,"title":title,"abstract":abstract})

# deduplicate
seen = {}
for x in feature_types:
    seen[x["name"]] = x
feature_types = list(seen.values())

with open(FTCSV,"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["name","title","abstract"])
    w.writeheader(); w.writerows(feature_types)

print("Published feature types:", len(feature_types))
print("Saved:", FTCSV)

cands = []
for x in feature_types:
    txt = (x["name"]+" "+x["title"]+" "+x["abstract"]).lower()
    score = 0
    if "benthic" in txt: score += 20
    if "geomorphic" in txt: score += 20
    if "habitat" in txt: score += 5
    if "reef" in txt: score += 2
    if score:
        y = dict(x); y["score"] = score; cands.append(y)

cands.sort(key=lambda z:(-z["score"], z["name"]))

print("\nSTEP 3 — Candidate habitat layers")
if not cands:
    for x in feature_types:
        print(f"  {x['name']} | {x['title']}")
    sys.exit("No benthic/geomorphic/habitat candidates found.")

for i,x in enumerate(cands,1):
    print(f"[{i}] score={x['score']} | {x['name']} | {x['title']}")

benthic = [x for x in cands if "benthic" in (x["name"]+" "+x["title"]+" "+x["abstract"]).lower()]
geom = [x for x in cands if "geomorphic" in (x["name"]+" "+x["title"]+" "+x["abstract"]).lower()]
chosen = benthic[0] if benthic else (geom[0] if geom else cands[0])

print("\nSelected diagnostic layer:")
print(" ", chosen["name"])
print(" ", chosen["title"])

print("\nSTEP 4 — Masinloc GetFeature GeoJSON test")
bbox = f"{WEST},{SOUTH},{EAST},{NORTH},EPSG:4326"
params = {
    "service":"WFS",
    "version":"2.0.0",
    "request":"GetFeature",
    "typeNames":chosen["name"],
    "outputFormat":"application/json",
    "srsName":"EPSG:4326",
    "bbox":bbox,
    "count":5000,
}
r2 = get(params)
print("Response size:", round(len(r2.content)/1024,1), "KB")

try:
    payload = r2.json()
except Exception:
    bad = OUT / "Masinloc_WFS_test_nonjson_response.txt"
    bad.write_bytes(r2.content)
    raise RuntimeError(f"Non-JSON response saved to {bad}")

GJ.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
features = payload.get("features", [])
print("Features returned:", len(features))
print("Saved:", GJ)

if not features:
    print("\nWFS worked but this layer returned no features in the test box.")
    print("Paste the candidate layer list above back into ChatGPT.")
    sys.exit(3)

keys = sorted({k for feat in features[:200]
               for k in feat.get("properties",{}).keys()})
print("\nSTEP 5 — Property fields")
for k in keys:
    print(" ",k)

likely = [k for k in keys if any(t in k.lower() for t in
          ("class","habitat","benthic","geomorphic","zone","label","name","type"))]

if likely:
    print("\nLikely classification fields:")
    for field in likely:
        vals = sorted({str(f.get("properties",{}).get(field))
                       for f in features
                       if f.get("properties",{}).get(field) is not None})
        print(f"  {field}: {len(vals)} unique")
        for v in vals[:30]:
            print("    ",v)

print("\nFirst 3 feature properties:")
for f in features[:3]:
    print(json.dumps(f.get("properties",{}), indent=2, ensure_ascii=False))

print("\nSTEP 6 — Optional GeoPandas validation")
try:
    import geopandas as gpd
    gdf = gpd.read_file(GJ)
    print("GeoPandas OK:", len(gdf), "features")
    print("CRS:", gdf.crs)
    print("Geometry:", gdf.geometry.geom_type.value_counts().to_dict())
    print("Bounds:", gdf.total_bounds)
except ImportError:
    print("GeoPandas not installed; raw GeoJSON test is still valid.")
    print("Install later with: conda install -c conda-forge geopandas pyogrio shapely -y")
except Exception as e:
    print("GeoPandas validation warning:", e)

print("\n"+"="*78)
print("WFS TEST COMPLETE")
print("="*78)
print("Capabilities:", CAP)
print("Feature types:", FTCSV)
print("Masinloc GeoJSON:", GJ)
