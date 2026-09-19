"""FastAPI backend for the Bund -> Parcel Mapper GIS web app.

Serves the React frontend and exposes:
  GET  /api/catalog          available layers + sample tiles
  GET  /api/geojson/{key}    a precomputed parcel layer (drone / region)
  POST /api/detect           run Delineate Anything on a sample tile (live)
  /outputs/*                 overlays and result files (static)
"""
import os
import sys
import json
import glob
from datetime import datetime

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import pipeline, large_infer  # noqa: E402
import geopandas as gpd
import pandas as pd
import rasterio

FRONT = os.path.join(ROOT, "frontend")
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data", "ai4b")
REGISTRY_PATH = os.path.join(OUT, "master_registry.geojson")

app = FastAPI(title="Dossier")

# Drone-only: the full survey + the SAM 3 benchmark layer.
LAYERS = {
    "drone": {"label": "Drone survey — Delineate Anything v2", "region": "India (drone)",
              "fields": 1854, "area_ha": 364.3, "dynamic": False,
              "path": os.path.join(OUT, "df1_full", "DF1, DF2_parcels.geojson"),
              "overlay": "/outputs/df1_full/DF1, DF2_overlay.png"},
    "sam3": {"label": "SAM 3 (Meta) — benchmark", "region": "India (drone)",
             "fields": 0, "area_ha": 0.0, "dynamic": True,
             "path": os.path.join(OUT, "sam3", "sam3_parcels.geojson"),
             "overlay": None},
}

DRONE_PATH = r"C:\Users\dhars\Downloads\DF1, DF2.tif"
_CELL = 24576  # native px per region (~1 km at 4 cm GSD)


def drone_regions():
    """Split the drone orthomosaic into a grid of detectable regions."""
    if not os.path.exists(DRONE_PATH):
        return {}
    with rasterio.open(DRONE_PATH) as s:
        W, H = s.width, s.height
    regs, n = {}, 1
    for ry in range(0, H, _CELL):
        for rx in range(0, W, _CELL):
            regs[f"R{n}"] = (rx, ry, min(_CELL, W - rx), min(_CELL, H - ry))
            n += 1
    return regs


REGIONS = drone_regions()


def sample_tiles():
    return list(REGIONS.keys())


@app.get("/api/catalog")
def catalog():
    layers = []
    for k, v in LAYERS.items():
        avail = os.path.exists(v["path"])
        fields, area = v["fields"], v["area_ha"]
        if v.get("dynamic") and avail:
            try:
                g = gpd.read_file(v["path"])
                fields = len(g)
                area = round(float(g["area_ha"].sum()), 1) if "area_ha" in g.columns else 0.0
            except Exception:
                pass
        layers.append({"key": k, "label": v["label"], "region": v["region"],
                       "fields": fields, "area_ha": area, "overlay": v["overlay"],
                       "available": avail})
               
    registry_available = os.path.exists(REGISTRY_PATH)
    reg_fields = 0
    reg_area = 0.0
    if registry_available:
        try:
            reg_gdf = gpd.read_file(REGISTRY_PATH)
            reg_fields = len(reg_gdf)
            if "area_ha" in reg_gdf.columns:
                reg_area = round(float(reg_gdf["area_ha"].sum()), 2)
        except Exception:
            pass
            
    layers.insert(0, {
        "key": "registry", "label": "Master Land Registry", "region": "Global (Aggregated)",
        "fields": reg_fields, "area_ha": reg_area, "overlay": None,
        "available": registry_available
    })
    
    return {"layers": layers, "tiles": sample_tiles()}


@app.get("/api/geojson/{key}")
def geojson(key: str):
    if key == "registry" and os.path.exists(REGISTRY_PATH):
        return FileResponse(REGISTRY_PATH, media_type="application/geo+json")
    lyr = LAYERS.get(key)
    if lyr and os.path.exists(lyr["path"]):
        return FileResponse(lyr["path"], media_type="application/geo+json")
    return JSONResponse({"type": "FeatureCollection", "features": []})


def update_registry(new_gj_path: str):
    if not os.path.exists(new_gj_path):
        return
    new_gdf = gpd.read_file(new_gj_path)
    if new_gdf.empty:
        return
        
    if not os.path.exists(REGISTRY_PATH):
        new_gdf.to_file(REGISTRY_PATH, driver="GeoJSON")
        return
        
    reg_gdf = gpd.read_file(REGISTRY_PATH)
    if reg_gdf.empty:
        new_gdf.to_file(REGISTRY_PATH, driver="GeoJSON")
        return
        
    # Spatial deduplication: skip new polygons that heavily intersect existing ones
    new_polys_to_keep = []
    # Create spatial index for fast intersection
    sindex = reg_gdf.sindex
    
    for _, row in new_gdf.iterrows():
        geom = row.geometry
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = reg_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(geom)]
        
        is_duplicate = False
        for _, reg_row in precise_matches.iterrows():
            inter_area = geom.intersection(reg_row.geometry).area
            if inter_area / geom.area > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            new_polys_to_keep.append(row)
            
    if new_polys_to_keep:
        to_append = gpd.GeoDataFrame(new_polys_to_keep, crs=new_gdf.crs)
        updated_reg = gpd.GeoDataFrame(pd.concat([reg_gdf, to_append], ignore_index=True), crs=reg_gdf.crs)
        updated_reg = updated_reg.drop(columns=["parcel_id"], errors="ignore")
        updated_reg["parcel_id"] = range(1, len(updated_reg) + 1)
        updated_reg.to_file(REGISTRY_PATH, driver="GeoJSON")


@app.post("/api/detect")
def detect(body: dict = Body(...)):
    region = body.get("tile", "") or body.get("region", "")
    register_flag = body.get("register", False)

    win = REGIONS.get(region)
    if win is None:
        return JSONResponse({"error": "region not found"}, status_code=404)

    summary = large_infer.run_large(DRONE_PATH, os.path.join(OUT, "api"),
                                    target_gsd_m=1.0, region=win)
    gj = {"type": "FeatureCollection", "features": []}
    gp = summary["paths"].get("geojson")
    
    if gp and os.path.exists(gp):
        if register_flag:
            update_registry(gp)
        with open(gp, "r", encoding="utf-8") as f:
            gj = json.load(f)
            
    ov = summary["paths"].get("overlay")
    return {"stats": {"fields": summary["parcels_detected"], "area_ha": summary.get("total_area_ha")},
            "geojson": gj,
            "overlay": ("/outputs/api/" + os.path.basename(ov)) if ov else None}


app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONT, "index.html"))
