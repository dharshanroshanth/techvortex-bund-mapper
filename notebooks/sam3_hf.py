# ============================================================================
# TechVortex'26 - SAM 3 (facebook/sam3) on a drone region, via HF Transformers.
# Runs SAM 3 automatic mask generation on the drone sample and saves an overlay
# + mask count, for comparison against Delineate Anything v2.
# Needs: GPU + internet + HF token with facebook/sam3 access.
# ============================================================================
import os, sys, subprocess
def _pip(*p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=False)
_pip("-U", "transformers", "accelerate")

# HF token injected at push time (unique placeholder, replaced once).
HF_TOKEN = "PLACEHOLDER_HFTOKEN"
if os.environ.get("HF_TOKEN"):
    HF_TOKEN = os.environ["HF_TOKEN"]
assert HF_TOKEN.startswith("hf_"), "HF token not injected"

from huggingface_hub import login
login(HF_TOKEN)

import numpy as np, torch, random
from PIL import Image
from transformers import pipeline

import glob as _glob
print("CUDA:", torch.cuda.is_available())
print("input tree:", _glob.glob("/kaggle/input/**/*", recursive=True)[:10])
_cand = _glob.glob("/kaggle/input/**/R6.png", recursive=True) or _glob.glob("/kaggle/input/**/*.png", recursive=True)
assert _cand, "sample image not found under /kaggle/input"
IMG = _cand[0]
img = Image.open(IMG).convert("RGB")
print("image:", IMG, img.size)

gen = pipeline("mask-generation", model="facebook/sam3",
               device=0 if torch.cuda.is_available() else -1)
out = gen(img, points_per_batch=64)
masks = out["masks"]
print("SAM3 masks generated:", len(masks))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 11))
ax.imshow(img)
for m in masks:
    m = np.asarray(m)
    ax.contour(m, colors=[[random.random(), random.random(), random.random()]], linewidths=0.7)
ax.set_title(f"SAM 3 (facebook/sam3) - {len(masks)} masks on drone region R6", fontsize=12)
ax.set_axis_off()
fig.savefig("/kaggle/working/sam3_R6_overlay.png", dpi=130, bbox_inches="tight")
print("saved /kaggle/working/sam3_R6_overlay.png ; total masks:", len(masks))
