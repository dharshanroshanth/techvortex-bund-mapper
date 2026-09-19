# ============================================================================
# TechVortex'26 - SAM 3 field-parcel segmentation on Kaggle (GPU + internet)
# Text-prompts SAM 3 with "field" on AI4Boundaries tiles, saves each tile's
# instance labels as a georeferenced GeoTIFF (+ overlay) for our geo pipeline.
# Requires: HF token (env HF_TOKEN or Kaggle secret) + accepted SAM3 license.
# ============================================================================
import os, sys, subprocess, io, csv, urllib.request
import numpy as np

HF_TOKEN = os.environ.get("HF_TOKEN", "")
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = HF_TOKEN or UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
assert HF_TOKEN, "Set HF_TOKEN and accept the SAM3 license on HuggingFace."

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/facebookresearch/sam3.git", "rasterio"], check=False)

from huggingface_hub import login
login(HF_TOKEN)

import torch
import rasterio
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.4)
PROMPT = "field"

BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/orthophoto/"
CSVU = BASE + "ai4boundaries_ftp_urls_orthophoto_split.csv"
HDR = {"User-Agent": "Mozilla/5.0"}
def fetch(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=HDR), timeout=180).read()

rows = list(csv.DictReader(io.StringIO(fetch(CSVU).decode("utf-8", "ignore"))))
WANT = ["FR_55608", "NL_3141", "AT_8502", "ES_1495", "SE_28230", "FR_50570"]
sel = [r for r in rows if r["file_id"] in WANT]

os.makedirs("/kaggle/working/sam3", exist_ok=True)
os.makedirs("/kaggle/temp", exist_ok=True)

for r in sel:
    ip = "/kaggle/temp/" + r["file_id"] + ".tif"
    open(ip, "wb").write(fetch(r["orthophoto_images_file_url"]))
    with rasterio.open(ip) as s:
        prof = s.profile
        rgb = np.transpose(s.read()[:3], (1, 2, 0)).astype("uint8")

    state = processor.set_image(Image.fromarray(rgb))
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=PROMPT)
    masks = state["masks"]
    m = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)

    labels = np.zeros(rgb.shape[:2], dtype="int32")
    n = m.shape[0] if m.ndim == 3 else 0
    for i in range(n):
        labels[m[i] > 0] = i + 1
    prof.update(count=1, dtype="int32")
    with rasterio.open(f"/kaggle/working/sam3/{r['file_id']}_labels.tif", "w", **prof) as dst:
        dst.write(labels, 1)
    print(f"{r['file_id']}: SAM3 found {n} field instances")

print("done -> /kaggle/working/sam3/*_labels.tif")
