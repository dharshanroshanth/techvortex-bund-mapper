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


def predict_polygons(rgb: np.ndarray, conf: float = 0.2, imgsz: int = None,
                     path: str = WEIGHTS, augment: bool = False, iou: float = 0.6):
    """RGB uint8 (H,W,3) -> list of (K,2) field-boundary polygons in pixel (x,y).

    imgsz defaults to the image's own size (rounded to a multiple of 32, clamped
    to [512, 1280]). Upscaling a small tile to a fixed large size blurs the
    fields and makes the model miss them, so we match the native resolution.
    """
    model = load(path)
    if imgsz is None:
        imgsz = max(512, min(1280, int(round(max(rgb.shape[:2]) / 32) * 32)))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])          # ultralytics expects BGR
    res = model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou,
                        retina_masks=True, augment=augment, verbose=False)
    r = res[0]
    if r.masks is None:
        return []
    return [np.asarray(p, dtype="float64") for p in r.masks.xy]
