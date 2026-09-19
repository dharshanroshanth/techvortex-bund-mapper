"""Georeference SAM 3's per-region pixel polygons into one WGS84 layer.

Reads:  outputs/sam3layer/sam3_pixel_polys.json  (from the Kaggle kernel)
        sam3_regions/meta.json                    (per-region affine transform)
Writes: outputs/sam3/sam3_parcels.geojson         (the "SAM 3" map layer)
"""
import json, os
import geopandas as gpd
from shapely.geometry import Polygon
from affine import Affine

polys = json.load(open("outputs/sam3layer/sam3_pixel_polys.json"))
meta = json.load(open("sam3_regions/meta.json"))
crs, regions = meta["crs"], meta["regions"]

records = []
for name, plist in polys.items():
    m = regions.get(name)
    if not m:
        continue
    T = Affine(*m["t"])                       # PNG pixel -> CRS (EPSG:32643)
    for pp in plist:
        geom = Polygon([T * (float(x), float(y)) for x, y in pp]).buffer(0)
        if not geom.is_empty and geom.area > 0:
            records.append({"geometry": geom, "source": name})

gdf = gpd.GeoDataFrame(records, crs=crs)
# de-duplicate parcels straddling adjacent regions (>50% overlap)
gdf["_a"] = gdf.geometry.area
gdf = gdf.sort_values("_a", ascending=False).reset_index(drop=True)
kept, keep_idx = [], []
for i, g in enumerate(gdf.geometry.values):
    if any(g.intersects(k) and g.intersection(k).area / min(g.area, k.area) > 0.5 for k in kept):
        continue
    kept.append(g); keep_idx.append(i)
gdf = gdf.iloc[keep_idx].reset_index(drop=True).drop(columns="_a")

gdf["area_ha"] = (gdf.geometry.area / 10_000).round(3)
gdf["area_acre"] = (gdf["area_ha"] * 2.47105).round(3)
cen = gdf.geometry.centroid.to_crs(4326)
gdf["lon"], gdf["lat"] = cen.x.round(6), cen.y.round(6)
gdf["parcel_id"] = range(1, len(gdf) + 1)

os.makedirs("outputs/sam3", exist_ok=True)
gdf.to_crs(4326).to_file("outputs/sam3/sam3_parcels.geojson", driver="GeoJSON")
print("SAM3 layer parcels:", len(gdf), "| total ha:", round(float(gdf["area_ha"].sum()), 1))
