# ============================================================================
# TechVortex'26 - SAM 3 on all drone regions -> per-region mask polygons (pixel
# coords) as JSON, for local georeferencing into a "SAM 3" map layer.
# Needs: GPU + internet + HF token with facebook/sam3 access.
# ============================================================================
import os, sys, subprocess, glob, json
def _pip(*p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=False)
_pip("-U", "transformers", "accelerate")

HF_TOKEN = "PLACEHOLDER_HFTOKEN"
if os.environ.get("HF_TOKEN"):
    HF_TOKEN = os.environ["HF_TOKEN"]
assert HF_TOKEN.startswith("hf_"), "HF token not injected"
from huggingface_hub import login
login(HF_TOKEN)

import numpy as np, torch, cv2
from PIL import Image
from transformers import pipeline

print("CUDA:", torch.cuda.is_available())
gen = pipeline("mask-generation", model="facebook/sam3",
               device=0 if torch.cuda.is_available() else -1)

imgs = sorted(glob.glob("/kaggle/input/**/R*.png", recursive=True))
print("regions found:", len(imgs))
out = {}
for ip in imgs:
    name = os.path.splitext(os.path.basename(ip))[0]
    img = Image.open(ip).convert("RGB")
    res = gen(img, points_per_batch=64)
    polys = []
    for m in res["masks"]:
        m = (np.asarray(m) > 0).astype("uint8")
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 80:
                continue
            eps = 0.002 * cv2.arcLength(c, True)
            ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(ap) >= 3:
                polys.append(ap.tolist())
    out[name] = polys
    print(f"{name}: {len(res['masks'])} masks -> {len(polys)} polygons")

json.dump(out, open("/kaggle/working/sam3_pixel_polys.json", "w"))
print("saved sam3_pixel_polys.json ; total polygons:", sum(len(v) for v in out.values()))
