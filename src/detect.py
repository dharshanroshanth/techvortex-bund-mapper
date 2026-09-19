"""Bund (and field-extent) detection with the trained U-Nets.

Two trained models live in models/:
  - bund_unet_resnet34.pth : single-task, predicts the bund BOUNDARY (1 ch)
  - bund_unet_mt.pth       : multi-task, predicts field EXTENT + BOUNDARY (2 ch)

The multi-task model is preferred when present: its extent channel masks the
parcel extraction to real fields, which gives noticeably cleaner parcels.
"""
from __future__ import annotations
import os
import numpy as np

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ST_WEIGHTS = os.path.join(_MODELS_DIR, "bund_unet_resnet34.pth")   # single-task
MT_WEIGHTS = os.path.join(_MODELS_DIR, "bund_unet_mt.pth")         # multi-task
DEFAULT_WEIGHTS = ST_WEIGHTS                                        # back-compat

_CACHE = {}  # (weights_path, classes) -> model


def weights_available(path: str = ST_WEIGHTS) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1000


def any_model_available() -> bool:
    return weights_available(MT_WEIGHTS) or weights_available(ST_WEIGHTS)


def best_available():
    """Return (kind, weights_path): 'mt' preferred, else 'st', else (None, None)."""
    if weights_available(MT_WEIGHTS):
        return "mt", MT_WEIGHTS
    if weights_available(ST_WEIGHTS):
        return "st", ST_WEIGHTS
    return None, None


def _load(weights_path: str, classes: int, device: str = "cpu"):
    key = (weights_path, classes)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    import segmentation_models_pytorch as smp
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    _CACHE[key] = model
    return model


def _infer(rgb: np.ndarray, weights_path: str, classes: int, device: str = "cpu") -> np.ndarray:
    """RGB uint8 (H,W,3) -> sigmoid probabilities (classes, H, W)."""
    import torch
    model = _load(weights_path, classes, device)
    h, w = rgb.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(img.transpose(2, 0, 1)[None].astype("float32") / 255.0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).cpu().numpy()[0]
    return prob[:, :h, :w]


def predict_boundary(rgb: np.ndarray, weights_path: str = ST_WEIGHTS,
                     device: str = "cpu") -> np.ndarray:
    """Single-task: boundary probability map (H, W)."""
    return _infer(rgb, weights_path, 1, device)[0]


def predict_extent_boundary(rgb: np.ndarray, weights_path: str = MT_WEIGHTS,
                            device: str = "cpu"):
    """Multi-task: (extent_prob, boundary_prob), each (H, W)."""
    p = _infer(rgb, weights_path, 2, device)
    return p[0], p[1]
