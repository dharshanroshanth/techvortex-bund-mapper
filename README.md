# Dossier — AI Agricultural Land Parcel & Bund Detection

**TechVortex '26 · Problem Statement PS06**

Automatically detect field **bunds** in drone/aerial imagery, delineate individual
**parcels**, convert them to **GIS-ready polygons** with real GPS, compute each
parcel's **area** (ha/acre), and export for any GIS — replacing manual land surveying.

Demonstrated on real high-resolution drone orthomosaics of farmland in **Tamil Nadu, India**:
**10,935 field parcels across ~1,926 hectares** mapped automatically.

## Pipeline
```
drone/aerial GeoTIFF
   → preprocess (tile large orthomosaics, resample to field scale)
   → bund/field detection (model)
   → parcel extraction (instances, or watershed on boundaries)
   → vectorize + georeference (pixels → GPS polygons)
   → area (exact, equal-area / UTM)
   → overlay + export (GeoJSON / GeoPackage / CSV / PNG)
```

## Detectors (three, one pipeline)
1. **Delineate Anything v2** *(default — SOTA)* — a YOLOv11 field-boundary
   foundation model (73M fields, 61 countries). Outputs parcels directly.
2. **Our multi-task U-Net** *(trained by us)* — ResNet-34 encoder, predicts field
   extent + boundary; trained on AI4Boundaries. Watershed closes parcels.
3. **Ground-truth labels** — benchmark/reference (sample tiles only).

## Quick start
```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Download the Delineate Anything weights into `models/`:
`https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnythingv2.pt`

**Web demo:**
```bash
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

**Command line — a sample tile:**
```bash
.\.venv\Scripts\python.exe src\pipeline.py --image data\ai4b\images\FR_55608_ortho_1m_512.tif --delineate
```

**Command line — a large drone orthomosaic (memory-safe, tiled):**
```bash
.\.venv\Scripts\python.exe -m src.large_infer --image "path\to\drone.tif" --out outputs\drone --preview 24576
```

## Repo layout
```
src/pipeline.py     core: parcels → polygons → area → export/overlay
src/detect.py       our trained U-Net inference
src/delineate.py    Delineate Anything v2 inference
src/large_infer.py  tiled inference for huge orthomosaics
app/streamlit_app.py  web demo
notebooks/          Kaggle training scripts (our U-Net, multi-task, pseudo-label)
docs/               architecture + dossier pages, jury script
```

## Data
- **AI4Boundaries** (EU JRC) — 1 m aerial, 7 countries, real parcel labels (training + eval).
- Real drone orthomosaics (Tamil Nadu) for the primary demo.

## Tech
PyTorch · segmentation-models-pytorch · Ultralytics (YOLOv11) · rasterio · shapely ·
geopandas · pyproj · scikit-image · Streamlit · Kaggle.
