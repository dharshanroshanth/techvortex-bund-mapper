"""TechVortex web demo — Bund -> Parcel Mapper.

Pick a sample tile, upload an image, or point at a large drone/aerial GeoTIFF;
choose a detector; and get individual parcels with GPS geometry + area, an
overlay, and GIS exports.
"""
import os
import sys
import glob

import streamlit as st
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import pipeline, detect, delineate, large_infer  # noqa: E402

DATA = os.path.join(ROOT, "data", "ai4b")
OUT = os.path.join(ROOT, "outputs", "app")

st.set_page_config(page_title="Bund -> Parcel Mapper", layout="wide", page_icon="\U0001F33E")

st.title("\U0001F33E  Bund \u2192 Parcel Mapper")
st.caption(
    "Detect field **bunds** from aerial/drone imagery \u2192 group into individual "
    "**parcels** \u2192 georeferenced GPS polygons \u2192 **area** (ha/acre) \u2192 GIS export. "
    "\u2014 TechVortex\u201926"
)

HAS_DA = delineate.available()
HAS_UNET = detect.any_model_available()

DA = "Delineate Anything v2 (SOTA)"
UNET = "Our trained U-Net"
GT = "Ground-truth labels"
SRC_SAMPLE = "Sample tile"
SRC_UPLOAD = "Upload image"
SRC_LARGE = "Large drone / aerial file (path)"


def samples():
    out = []
    for img in sorted(glob.glob(os.path.join(DATA, "images", "*.tif"))):
        m = img.replace("images", "masks").replace(
            "_ortho_1m_512.tif", "_ortholabel_1m_512.tif")
        fid = os.path.basename(img).replace("_ortho_1m_512.tif", "")
        out.append((fid, img, m if os.path.exists(m) else None))
    return out


with st.sidebar:
    st.header("Input")
    if HAS_DA:
        st.success("Delineate Anything v2 loaded \u2713")
    if HAS_UNET:
        st.success("Our U-Net loaded \u2713")

    source = st.radio("Source", [SRC_SAMPLE, SRC_UPLOAD, SRC_LARGE])

    img_path = mask_path = None
    large_mode = source == SRC_LARGE
    preview_px = 0

    if source == SRC_SAMPLE:
        detector = st.radio("Detector", [DA, UNET, GT])
        pairs = samples()
        fid = st.selectbox("Tile", [p[0] for p in pairs], index=min(4, len(pairs) - 1))
        _, img_path, mask_path = next(p for p in pairs if p[0] == fid)
    elif source == SRC_UPLOAD:
        detector = st.radio("Detector", [DA, UNET])
        up = st.file_uploader("Image (GeoTIFF for real GPS, or any JPG/PNG/TIF)",
                              type=["tif", "tiff", "jpg", "jpeg", "png"])
        if up is not None:
            os.makedirs(OUT, exist_ok=True)
            img_path = os.path.join(OUT, "upload_" + up.name)
            with open(img_path, "wb") as f:
                f.write(up.getbuffer())
    else:  # large file by path
        detector = DA
        st.caption("Detector: **Delineate Anything v2** (best for large orthomosaics).")
        large_path = st.text_input("GeoTIFF path on this machine",
                                   value=r"C:\Users\dhars\Downloads\DF1, DF2.tif")
        full = st.checkbox("Full scan (whole file \u2014 slower)", value=False)
        preview_px = 0 if full else int(st.number_input(
            "Preview region size (native px)", value=24576, min_value=2048, step=4096))
        img_path = large_path if (large_path and os.path.exists(large_path)) else None
        if large_path and not img_path:
            st.warning("File not found at that path.")

    # readiness
    if large_mode:
        ready = bool(img_path) and HAS_DA
    elif detector == DA:
        ready = bool(img_path) and HAS_DA
    elif detector == UNET:
        ready = bool(img_path) and HAS_UNET
    else:
        ready = bool(mask_path)

    run_btn = st.button("Detect parcels", type="primary", disabled=not ready)
    if large_mode and img_path:
        st.caption("Large files stream from disk \u2014 a full scan may take minutes.")

if run_btn and img_path:
    try:
        with st.spinner(f"Running {detector}\u2026 (large files can take a while)"):
            if large_mode:
                summary = large_infer.run_large(img_path, OUT, preview_px=preview_px)
            elif detector == DA:
                summary = pipeline.run_delineate(img_path, OUT)
            elif detector == UNET:
                summary = pipeline.run_model(img_path, OUT)
            else:
                summary = pipeline.run(img_path, mask_path, OUT)
    except Exception as e:
        st.error(f"Could not process this image: {e}")
        st.info("For real GPS + area, use a **georeferenced GeoTIFF**.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Parcels detected", summary["parcels_detected"])
    c2.metric("Detector", detector.split(" (")[0])
    c3.metric("Total field area",
              f"{summary['total_area_ha']} ha" if summary.get("total_area_ha") else "\u2014 (no georef)")

    left, right = st.columns([3, 2])
    with left:
        if summary["paths"].get("overlay"):
            st.image(summary["paths"]["overlay"], use_container_width=True,
                     caption="Detected parcels overlaid on the imagery")
        else:
            st.warning("No parcels detected on this image.")
    with right:
        if summary["paths"].get("csv"):
            df = pd.read_csv(summary["paths"]["csv"])
            show = [c for c in ["parcel_id", "area_ha", "area_acre", "area_px", "lat", "lon"]
                    if c in df.columns and df[c].notna().any()]
            st.dataframe(df[show], use_container_width=True, height=430)
        for label, key, mime in [("\u2b07 GeoJSON (WGS84)", "geojson", "application/geo+json"),
                                 ("\u2b07 GeoPackage", "gpkg", "application/geopackage+sqlite3"),
                                 ("\u2b07 Area CSV", "csv", "text/csv")]:
            p = summary["paths"].get(key)
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    st.download_button(label, f.read(), file_name=os.path.basename(p), mime=mime)
        if summary.get("total_area_ha"):
            st.caption(f"CRS: {summary['crs']} (projected) \u2014 areas are accurate.")
else:
    st.info("Pick a source + detector in the sidebar, then click **Detect parcels**. "
            "Try **Large drone / aerial file** to run on a full orthomosaic.")
