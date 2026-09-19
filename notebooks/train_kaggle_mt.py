# ============================================================================
# TechVortex'26 - MULTI-TASK bund + extent segmentation - Kaggle T4
# Predicts field EXTENT and BOUNDARY together (2 channels). Reports extent IoU
# (headline) and boundary IoU. Saves /kaggle/working/bund_unet_mt.pth
# ============================================================================
import subprocess, sys
def _pip(*p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=False)
_pip("segmentation-models-pytorch", "rasterio", "scipy")

import os, io, csv, urllib.request, random, numpy as np, torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
import rasterio
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi

random.seed(0); np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/orthophoto/"
CSVU = BASE + "ai4boundaries_ftp_urls_orthophoto_split.csv"
HDR = {"User-Agent": "Mozilla/5.0"}
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=180).read()

N_TRAIN, N_VAL, EPOCHS = 400, 80, 18
CACHE = "/kaggle/temp/ai4b"
os.makedirs(CACHE + "/images", exist_ok=True); os.makedirs(CACHE + "/masks", exist_ok=True)

rows = [r for r in csv.DictReader(io.StringIO(fetch(CSVU).decode("utf-8", "ignore")))
        if r["orthophoto_images_file_url"].endswith(".tif")
        and r["orthophoto_masks_file_url"].endswith(".tif")]
random.shuffle(rows)
cand = rows[:650]
print(f"{len(rows)} valid rows; downloading up to {len(cand)} pairs ...")

def dl_pair(r):
    ip = CACHE + "/images/" + r["orthophoto_images_file_url"].split("/")[-1]
    mp = CACHE + "/masks/"  + r["orthophoto_masks_file_url"].split("/")[-1]
    for path, url in [(ip, r["orthophoto_images_file_url"]), (mp, r["orthophoto_masks_file_url"])]:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            continue
        for _ in range(3):
            try:
                b = fetch(url)
                if len(b) > 1000:
                    open(path, "wb").write(b); break
            except Exception:
                pass
    if os.path.exists(ip) and os.path.getsize(ip) > 1000 and \
       os.path.exists(mp) and os.path.getsize(mp) > 200:
        return (ip, mp)
    return None

with ThreadPoolExecutor(max_workers=12) as ex:
    good = [x for x in ex.map(dl_pair, cand) if x]
print(f"Usable pairs: {len(good)}")
train_pairs, val_pairs = good[:N_TRAIN], good[N_TRAIN:N_TRAIN + N_VAL]
print(f"train={len(train_pairs)}  val={len(val_pairs)}")

class MTDS(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        with rasterio.open(ip) as s: img = s.read()[:3].astype("float32") / 255.0
        with rasterio.open(mp) as s:
            ext = (s.read(1) > 0.5).astype("float32")               # band 1 = extent
            bnd = (s.read(2) > 0.5).astype("float32")               # band 2 = boundary
        bnd = ndi.binary_dilation(bnd, iterations=1).astype("float32")
        y = np.stack([ext, bnd], 0)                                 # (2,H,W)
        return torch.from_numpy(img), torch.from_numpy(y)

def iou_ch(logits, y, ch):
    p = (torch.sigmoid(logits[:, ch]) > 0.5).float(); t = y[:, ch]
    inter = (p * t).sum((1, 2)); union = ((p + t) > 0).float().sum((1, 2))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()

tl = DataLoader(MTDS(train_pairs), batch_size=8, shuffle=True, num_workers=2, drop_last=True)
vl = DataLoader(MTDS(val_pairs), batch_size=8, shuffle=False, num_workers=2)

model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=2).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
dice = smp.losses.DiceLoss(mode="multilabel"); bce = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

best = 0.0
for ep in range(EPOCHS):
    model.train()
    for x, y in tl:
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x); loss = bce(out, y) + dice(out, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); ex_i, bd_i = [], []
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.cuda.amp.autocast(): out = model(x)
            ex_i.append(iou_ch(out, y, 0)); bd_i.append(iou_ch(out, y, 1))
    me, mb = float(np.mean(ex_i)), float(np.mean(bd_i))
    print(f"epoch {ep+1}/{EPOCHS}  extent IoU = {me:.3f}   boundary IoU = {mb:.3f}")
    if me + mb > best:
        best = me + mb
        torch.save(model.state_dict(), "/kaggle/working/bund_unet_mt.pth")
        print(f"   saved (extent {me:.3f}, boundary {mb:.3f})")
print("Saved weights -> /kaggle/working/bund_unet_mt.pth")
