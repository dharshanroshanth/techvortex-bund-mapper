"""Field-boundary detection with Delineate Anything v2.

A YOLOv11 instance-segmentation foundation model built specifically for
agricultural field boundaries (trained on 73M field instances, 61 countries).
It returns one instance mask per field, so we get parcels directly — no
watershed post-processing needed.
"""
from __future__ import annotations
import os
import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
WEIGHTS = os.path.join(_MODELS_DIR, "DelineateAnythingv2.pt")

_MODEL = None


def available(path: str = WEIGHTS) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1_000_000


def load(path: str = WEIGHTS):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(path)
    return _MODEL


def predict_polygons(rgb: np.ndarray, conf: float = 0.2, imgsz: int = 1024,
                     path: str = WEIGHTS):
    """RGB uint8 (H,W,3) -> list of (K,2) field-boundary polygons in pixel (x,y)."""
    model = load(path)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])          # ultralytics expects BGR
    res = model.predict(source=bgr, imgsz=imgsz, conf=conf,
                        retina_masks=True, verbose=False)
    r = res[0]
    if r.masks is None:
        return []
    return [np.asarray(p, dtype="float64") for p in r.masks.xy]
