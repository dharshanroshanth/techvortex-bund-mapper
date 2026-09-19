"""
Bund -> Parcel pipeline (TechVortex).

Turns a georeferenced aerial tile + a boundary/extent mask into individual
field parcels with real-world GPS geometry and area. This is the core of the
problem statement (steps 3-7):

    detect bunds (boundary)  ->  find fields (watershed)  ->  polygons (GPS)
    ->  area (ha / acre)     ->  overlay + GeoJSON/Shapefile/CSV export.

For the vertical slice the "detected boundary" is taken from the AI4Boundaries
label mask (band 2). Later, a trained segmentation model produces that same
boundary map and the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import os
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.features import shapes as rio_shapes
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.segmentation import watershed
from shapely.geometry import shape as shapely_shape
import geopandas as gpd

# AI4Boundaries mask band order (1-indexed in file, 0-indexed here)
B_EXTENT, B_BOUNDARY, B_DISTANCE, B_ENUM = 0, 1, 2, 3

ACRES_PER_HA = 2.47105


def _empty_gdf(crs):
    """An empty GeoDataFrame with the expected columns (for 'no detections')."""
    gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
    for c in ["parcel_id", "area_ha", "area_acre", "area_px", "lon", "lat"]:
        gdf[c] = []
    return gdf


def read_image(img_path: str):
    """Return (rgb HxWx3 uint8, rasterio dataset profile, transform, crs)."""
    with rasterio.open(img_path) as src:
        rgb = reshape_as_image(src.read()[:3])
        return rgb, src.profile, src.transform, src.crs


def read_mask(mask_path: str):
    """Return the 4-band AI4Boundaries mask as float32 (4, H, W)."""
    with rasterio.open(mask_path) as src:
        return src.read().astype("float32")


def parcels_from_boundary(extent: np.ndarray, boundary: np.ndarray,
                          min_area_px: int = 50) -> np.ndarray:
    """Grow closed field parcels from a (possibly broken) boundary map.

    This is the post-processing that turns thin, imperfect bund lines into
    closed labelled parcels:
      1. interior  = inside a field AND not on a bund
      2. seeds     = connected blobs of interior
      3. watershed = flood each seed across boundary gaps, clipped to extent

    Returns an int32 label image (0 = background, 1..N = parcels).
    """
    field = extent > 0.5
    bund = boundary > 0.5
    interior = field & ~bund

    # remove salt noise so we don't seed spurious parcels
    interior = ndi.binary_opening(interior, iterations=1)

    distance = ndi.distance_transform_edt(interior)
    markers = label(interior)
    labels = watershed(-distance, markers, mask=field)

    # drop tiny fragments
    for lab in np.unique(labels):
        if lab == 0:
            continue
        if (labels == lab).sum() < min_area_px:
            labels[labels == lab] = 0
    return labels.astype("int32")


def vectorize(labels: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """Convert a label image into a GeoDataFrame of georeferenced polygons.

    Area is computed in the source CRS. EPSG:3035 (used by AI4Boundaries) is
    an equal-area projection, so square metres are accurate as-is.
    """
    records = []
    for geom, val in rio_shapes(labels, mask=labels > 0, transform=transform):
        if val == 0:
            continue
        poly = shapely_shape(geom).buffer(0)          # fix any self-touch
        if poly.is_empty:
            continue
        records.append({"parcel_id": int(val), "geometry": poly})

    if not records:
        return _empty_gdf(crs)
    gdf = gpd.GeoDataFrame(records, crs=crs)
    # merge multipart pieces that share a parcel id, keep the largest ring
    gdf = gdf.dissolve(by="parcel_id", as_index=False)
    gdf["geometry"] = gdf.geometry.simplify(1.0)      # ~1 unit tolerance

    georef = crs is not None
    if georef:
        # equal-area / projected CRS in metres -> real areas + GPS centroid
        gdf["area_ha"] = gdf.geometry.area / 10_000.0
        gdf["area_acre"] = gdf["area_ha"] * ACRES_PER_HA
        cen = gdf.geometry.centroid.to_crs(4326)
        gdf["lon"] = cen.x.round(6)
        gdf["lat"] = cen.y.round(6)
    else:
        # plain image (no georeferencing) -> area is relative (pixels)
        gdf["area_ha"] = np.nan
        gdf["area_acre"] = np.nan
        gdf["lon"] = np.nan
        gdf["lat"] = np.nan
    gdf["area_px"] = gdf.geometry.area.round(0)

    gdf["_a"] = gdf.geometry.area
    gdf = gdf.sort_values("_a", ascending=False).drop(columns="_a").reset_index(drop=True)
    gdf["parcel_id"] = range(1, len(gdf) + 1)
    return gdf


def export(gdf: gpd.GeoDataFrame, out_dir: str, stem: str) -> dict:
    """Write a CSV always; GeoJSON (WGS84) + GeoPackage only when georeferenced."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    if gdf.crs is not None:
        geojson = os.path.join(out_dir, f"{stem}_parcels.geojson")
        gdf.to_crs(4326).to_file(geojson, driver="GeoJSON")
        paths["geojson"] = geojson

        gpkg = os.path.join(out_dir, f"{stem}_parcels.gpkg")
        gdf.to_file(gpkg, driver="GPKG")
        paths["gpkg"] = gpkg

    csv = os.path.join(out_dir, f"{stem}_areas.csv")
    gdf.drop(columns="geometry").to_csv(csv, index=False)
    paths["csv"] = csv
    return paths


def overlay(img_path: str, gdf: gpd.GeoDataFrame, out_png: str):
    """Draw parcel boundaries + area labels over the original tile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(img_path) as src:
        fig, ax = plt.subplots(figsize=(11, 11))
        rasterio.plot.show(src, ax=ax)
        gdf.boundary.plot(ax=ax, color="#ff2d55", linewidth=1.6)
        for _, row in gdf.iterrows():
            c = row.geometry.centroid
            ax.annotate(f"{row.parcel_id}\n{row.area_ha:.2f} ha",
                        (c.x, c.y), color="white", fontsize=7,
                        ha="center", va="center", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="#00000088", ec="none"))
        ax.set_title(f"{os.path.basename(img_path)}  -  "
                     f"{len(gdf)} parcels, {gdf.area_ha.sum():.1f} ha total",
                     fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)


def run(img_path: str, mask_path: str, out_dir: str) -> dict:
    """Full pipeline on one tile. Returns a small summary dict."""
    stem = os.path.splitext(os.path.basename(img_path))[0].replace("_ortho_1m_512", "")
    _, _, transform, crs = read_image(img_path)
    mask = read_mask(mask_path)

    labels = parcels_from_boundary(mask[B_EXTENT], mask[B_BOUNDARY])
    gdf = vectorize(labels, transform, crs)
    paths = export(gdf, out_dir, stem)
    png = os.path.join(out_dir, f"{stem}_overlay.png")
    overlay(img_path, gdf, png)
    paths["overlay"] = png

    # ground-truth parcel count from the enumeration band, for a sanity metric
    enum = mask[B_ENUM]
    gt_ids = np.unique(enum[enum > 0])
    return {
        "stem": stem, "crs": str(crs),
        "parcels_detected": len(gdf),
        "parcels_ground_truth": int(len(gt_ids)),
        "total_area_ha": round(float(gdf.area_ha.sum()), 2),
        "paths": paths,
    }


def read_image_any(path: str):
    """Return (rgb HxWx3 uint8, transform, crs).

    Uses rasterio when the file is a georeferenced raster (real GPS + area).
    Falls back to reading it as an ordinary photo (identity transform, crs
    None) so the model still runs on non-GeoTIFF drone images.
    """
    try:
        with rasterio.open(path) as src:
            rgb = reshape_as_image(src.read()[:3])
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            return rgb.astype("uint8"), src.transform, src.crs
    except Exception:
        pass
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), rasterio.Affine.identity(), None


def overlay_px(rgb: np.ndarray, gdf: gpd.GeoDataFrame, out_png: str):
    """Overlay parcel outlines on a plain (non-georeferenced) image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgb)
    for _, row in gdf.iterrows():
        polys = [row.geometry] if row.geometry.geom_type == "Polygon" else list(row.geometry.geoms)
        for poly in polys:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="#ff2d55", linewidth=1.6)
        c = row.geometry.centroid
        ax.annotate(str(row.parcel_id), (c.x, c.y), color="white", fontsize=7,
                    ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#00000088", ec="none"))
    ax.set_title(f"{len(gdf)} parcels detected (model)", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_model(img_path: str, out_dir: str, weights: str = None) -> dict:
    """Full pipeline using the TRAINED model to detect bunds from RGB.

    Works on any image: a georeferenced GeoTIFF gives real GPS + area; a plain
    photo gives parcels with relative (pixel) areas.
    """
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from src import detect

    stem = os.path.splitext(os.path.basename(img_path))[0].replace("_ortho_1m_512", "")
    rgb, transform, crs = read_image_any(img_path)

    kind, wp = detect.best_available()
    if kind == "mt":
        extent_prob, boundary = detect.predict_extent_boundary(rgb, weights or wp)
        extent = (extent_prob > 0.5).astype("float32")          # predicted field mask
        model_name = "multi-task U-Net (extent + boundary)"
    else:
        boundary = detect.predict_boundary(rgb, weights or wp)
        extent = np.ones(boundary.shape, dtype="float32")       # whole tile
        model_name = "U-Net (boundary)"
    labels = parcels_from_boundary(extent, boundary, min_area_px=120)
    gdf = vectorize(labels, transform, crs)
    paths = export(gdf, out_dir, stem)
    png = os.path.join(out_dir, f"{stem}_overlay.png")
    if crs is not None:
        overlay(img_path, gdf, png)
    else:
        overlay_px(rgb, gdf, png)
    paths["overlay"] = png
    return {
        "stem": stem, "crs": str(crs), "georeferenced": crs is not None,
        "boundary_source": model_name,
        "parcels_detected": len(gdf),
        "total_area_ha": (round(float(gdf.area_ha.sum()), 2) if crs is not None else None),
        "paths": paths,
    }


def run_delineate(img_path: str, out_dir: str, conf: float = 0.2) -> dict:
    """Full pipeline using Delineate Anything v2 (SOTA field-boundary model).

    The model returns one polygon per field, so parcels come straight out —
    we just georeference them, measure area, and export.
    """
    import sys
    from shapely.geometry import Polygon
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from src import delineate

    stem = os.path.splitext(os.path.basename(img_path))[0].replace("_ortho_1m_512", "")
    rgb, transform, crs = read_image_any(img_path)
    polys_px = delineate.predict_polygons(rgb, conf=conf)

    records = []
    for poly in polys_px:
        if len(poly) < 3:
            continue
        pts = [transform * (float(x), float(y)) for x, y in poly]   # pixel -> CRS (or identity)
        geom = Polygon(pts).buffer(0)
        if geom.is_empty or geom.area <= 0:
            continue
        records.append({"geometry": geom})

    gdf = gpd.GeoDataFrame(records, crs=crs) if records else _empty_gdf(crs)
    if len(gdf):
        gdf["geometry"] = gdf.geometry.simplify(1.0)
        georef = crs is not None
        if georef:
            gdf["area_ha"] = gdf.geometry.area / 10_000.0
            gdf["area_acre"] = gdf["area_ha"] * ACRES_PER_HA
            cen = gdf.geometry.centroid.to_crs(4326)
            gdf["lon"] = cen.x.round(6); gdf["lat"] = cen.y.round(6)
        else:
            gdf["area_ha"] = np.nan; gdf["area_acre"] = np.nan
            gdf["lon"] = np.nan; gdf["lat"] = np.nan
        gdf["area_px"] = gdf.geometry.area.round(0)
        gdf["_a"] = gdf.geometry.area
        gdf = gdf.sort_values("_a", ascending=False).drop(columns="_a").reset_index(drop=True)
        gdf["parcel_id"] = range(1, len(gdf) + 1)

    paths = export(gdf, out_dir, stem) if len(gdf) else {}
    png = os.path.join(out_dir, f"{stem}_overlay.png")
    if len(gdf):
        if crs is not None:
            overlay(img_path, gdf, png)
        else:
            overlay_px(rgb, gdf, png)
        paths["overlay"] = png
    return {
        "stem": stem, "crs": str(crs), "georeferenced": crs is not None,
        "boundary_source": "Delineate Anything v2",
        "parcels_detected": len(gdf),
        "total_area_ha": (round(float(gdf.area_ha.sum()), 2) if (len(gdf) and crs is not None) else None),
        "paths": paths,
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", help="4-band AI4Boundaries mask (label mode)")
    ap.add_argument("--model", action="store_true", help="use our trained U-Net")
    ap.add_argument("--delineate", action="store_true", help="use Delineate Anything v2 (SOTA)")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    if a.delineate:
        print(json.dumps(run_delineate(a.image, a.out), indent=2))
    elif a.model or not a.mask:
        print(json.dumps(run_model(a.image, a.out), indent=2))
    else:
        print(json.dumps(run(a.image, a.mask, a.out), indent=2))
