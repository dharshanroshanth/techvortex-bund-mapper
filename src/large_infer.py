"""Tiled field detection for LARGE orthomosaics (drone/satellite GeoTIFFs).

A full orthomosaic is far too big to feed to the model at once, so we:
  1. read it in windows (memory-safe),
  2. resample each window toward a target ground-sample-distance (~1 m) so the
     model sees field-scale structure, not individual plants,
  3. run Delineate Anything per tile and map polygons back to real coordinates,
  4. drop duplicates where a field straddles two tiles,
  5. measure area (reprojecting to metres when needed) and export.

Usage:
    python -m src.large_infer --image path/to/DF1.tif --out outputs/df1
"""
from __future__ import annotations
import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import geopandas as gpd  # noqa: E402
from src import delineate, pipeline  # noqa: E402


def run_large(img_path, out_dir, target_gsd_m=1.0, tile_px=1024, conf=0.2,
              overlap=0.0, overview_max=2400, region=None, preview_px=0):
    """Detect field parcels across a large georeferenced orthomosaic.

    region=(x,y,w,h) restricts the scan to a native-pixel window.
    preview_px>0 auto-selects a centred square of that many native pixels
    (a fast preview before committing to the full file).
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(img_path) as src:
        W, H = src.width, src.height
        crs = src.crs
        transform = src.transform
        nbands = src.count
        gsd = abs(transform.a)                                   # CRS units / pixel
        if crs is not None and crs.is_projected and gsd > 0:
            dec = max(1, int(round(target_gsd_m / gsd)))         # decimate to ~target GSD
        else:
            dec = 1
        step = tile_px * dec                                     # native px per tile
        stride = max(1, int(step * (1.0 - overlap)))
        bands = [1, 2, 3] if nbands >= 3 else [1, 1, 1]

        if preview_px and not region:
            s = min(preview_px, W, H)
            region = ((W - s) // 2, (H - s) // 2, s, s)
        if region:
            rx, ry, rw, rh = region
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(rw, W - rx), min(rh, H - ry)
        else:
            rx, ry, rw, rh = 0, 0, W, H

        print(f"{stem}: {W}x{H}px, gsd~{gsd:.3f}, decimate x{dec}, "
              f"tiles of {step}px (stride {stride}), scan window {rw}x{rh} at ({rx},{ry})")

        polys = []
        n_tiles = 0
        y = ry
        while y < ry + rh:
            wh = min(step, ry + rh - y)
            x = rx
            while x < rx + rw:
                ww = min(step, rx + rw - x)
                oh, ow = max(1, wh // dec), max(1, ww // dec)
                arr = src.read(bands, window=Window(x, y, ww, wh),
                               out_shape=(3, oh, ow), resampling=Resampling.average)
                rgb = np.transpose(arr, (1, 2, 0)).astype("uint8")
                n_tiles += 1
                if rgb.mean() < 3 or rgb.std() < 2:               # skip blank/nodata tiles
                    x += stride
                    continue
                for poly in delineate.predict_polygons(rgb, conf=conf):
                    if len(poly) < 3:
                        continue
                    pts = [transform * (x + px * dec, y + py * dec) for px, py in poly]
                    g = Polygon(pts).buffer(0)
                    if (not g.is_empty) and g.area > 0:
                        polys.append(g)
                x += stride
            y += stride
        print(f"  scanned {n_tiles} tiles, {len(polys)} raw detections")

    # de-duplicate fields that appear in overlapping / adjacent tiles
    polys.sort(key=lambda g: g.area, reverse=True)
    kept = []
    for g in polys:
        if any(g.intersects(k) and g.intersection(k).area / min(g.area, k.area) > 0.5
               for k in kept):
            continue
        kept.append(g)

    gdf = gpd.GeoDataFrame(geometry=kept, crs=crs) if kept else pipeline._empty_gdf(crs)
    if len(gdf):
        gdf["geometry"] = gdf.geometry.simplify(max(1.0, gsd * dec))
        if crs is not None:
            area_crs = crs if crs.is_projected else gdf.estimate_utm_crs()
            ag = gdf.geometry.to_crs(area_crs)
            gdf["area_ha"] = (ag.area / 10_000.0).round(3)
            gdf["area_acre"] = (gdf["area_ha"] * pipeline.ACRES_PER_HA).round(3)
            cen = gdf.geometry.centroid.to_crs(4326)
            gdf["lon"] = cen.x.round(6); gdf["lat"] = cen.y.round(6)
        else:
            gdf["area_ha"] = np.nan; gdf["area_acre"] = np.nan
            gdf["lon"] = np.nan; gdf["lat"] = np.nan
        gdf["_a"] = gdf.geometry.area
        gdf = gdf.sort_values("_a", ascending=False).drop(columns="_a").reset_index(drop=True)
        gdf["parcel_id"] = range(1, len(gdf) + 1)

    paths = pipeline.export(gdf, out_dir, stem) if len(gdf) else {}

    # downsampled overview overlay (of the scanned window)
    if len(gdf):
        png = os.path.join(out_dir, f"{stem}_overlay.png")
        _overlay_large(img_path, gdf, png, overview_max, region=(rx, ry, rw, rh))
        paths["overlay"] = png

    total_ha = round(float(gdf["area_ha"].sum()), 2) if (len(gdf) and crs is not None) else None
    return {"stem": stem, "crs": str(crs), "georeferenced": crs is not None,
            "boundary_source": "Delineate Anything v2 (tiled)",
            "parcels_detected": len(gdf), "total_area_ha": total_ha, "paths": paths}


def _overlay_large(img_path, gdf, out_png, overview_max, region=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio.plot as rplot

    with rasterio.open(img_path) as src:
        if region:
            rx, ry, rw, rh = region
        else:
            rx, ry, rw, rh = 0, 0, src.width, src.height
        odec = max(1, int(max(rw, rh) / overview_max))
        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        win = Window(rx, ry, rw, rh)
        ov = src.read(bands, window=win, out_shape=(3, max(1, rh // odec), max(1, rw // odec)),
                      resampling=Resampling.average)
        ov_transform = src.window_transform(win) * rasterio.Affine.scale(odec)

    fig, ax = plt.subplots(figsize=(13, 13))
    rplot.show(ov, transform=ov_transform, ax=ax)
    gdf.boundary.plot(ax=ax, color="#ff2d55", linewidth=1.0)
    ax.set_title(f"{os.path.basename(img_path)}  -  {len(gdf)} fields, "
                 f"{gdf['area_ha'].sum():.1f} ha total", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/large")
    ap.add_argument("--gsd", type=float, default=1.0, help="target ground sample distance (m)")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--preview", type=int, default=0,
                    help="fast preview: centred square of this many native px")
    ap.add_argument("--window", default="", help="scan only x,y,w,h (native px)")
    a = ap.parse_args()
    region = tuple(int(v) for v in a.window.split(",")) if a.window else None
    print(json.dumps(run_large(a.image, a.out, target_gsd_m=a.gsd, tile_px=a.tile,
                               conf=a.conf, region=region, preview_px=a.preview), indent=2))
