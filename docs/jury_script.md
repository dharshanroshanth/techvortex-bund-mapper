# TechVortex '26 — 7-Minute Jury Script
**Project:** Automated farm bund & parcel mapping from drone imagery

> Format: 7 minutes. One person drives the demo, one narrates. Times are a guide.
> Golden rule: **show the working thing early.** Don't spend 3 minutes on slides before the demo.

---

## 0:00–0:30 — The hook (say this first, no slides)
> "To know where one farmer's land ends and the next begins — and how big each plot is — someone has to physically walk the field and measure it. It's slow, needs manpower, and it's error-prone. **We replaced that surveyor with a drone photo and an AI model.**"

## 0:30–1:15 — What we built (one sentence per step, point at the diagram)
> "Our system takes a drone image and, fully automatically: **detects the bunds** — the ridges between fields — with a neural network, **groups them into individual parcels**, converts each to a **real GPS polygon**, computes its **area in hectares and acres**, draws it back on the photo to verify, and **exports GIS-ready files**."

*(Show the architecture page: claude.ai artifact — the 8-stage pipeline.)*

## 1:15–3:30 — LIVE DEMO (the heart — spend the most time here)
1. Open the web app. "This is running on this laptop, no internet needed."
2. Pick a real tile → click **Detect parcels**.
3. When the overlay appears: *"Every red boundary is a field our model found. Each is numbered with its area."*
4. Point at the table: *"Real areas — this one's 2.4 hectares — and real GPS coordinates."*
5. Click **Download GeoJSON** → *"This opens directly in QGIS, professional mapping software — these are actual land records, not a picture."* (Have QGIS open with it if possible.)
6. Switch to **ground-truth mode**: *"Here's how close we are to the official parcel data — 21 vs 20 fields."*

## 3:30–5:00 — The tech, and why (credibility)
> "The bund detection is a **U-Net segmentation model** — the published approach for field-boundary delineation. We made it **multi-task**: it predicts field *extent* and *boundary* together, which is best practice for this data and gives cleaner parcels."
> "We trained it on **AI4Boundaries** — real European parcel data — on a **Kaggle GPU**, and run inference right here on the laptop."
> "The clever part is the **post-processing**: bunds are thin and broken, so we use a **watershed** algorithm to close them into whole fields — where naive methods fail."
> *(If SAM 3 is integrated:)* "We also integrated **Meta's SAM 3**, a state-of-the-art foundation model, as a second detector for the sharpest boundaries."

## 5:00–6:00 — Results (the numbers)
- **Field extent IoU ~0.65**, boundary IoU ~0.39 — on a held-out test set (honest metric, not cherry-picked).
- **18 of 20 parcels** recovered on a real tile.
- **Areas are exact** — computed in an equal-area projection, not estimated.
- Runs on **free compute**; the demo is fully **offline**.

## 6:00–7:00 — Impact + roadmap + close
> "This turns a full day of manual surveying into ten seconds, and produces clean digital land records for planning, monitoring, and dispute resolution."
> "Next: more training for tighter internal bunds, and India-specific paddy imagery."
> **Close:** "A drone flies, our AI maps every field and its size, and it's ready for any GIS system. Thank you."

---

## Anticipated Q&A
- **"How accurate?"** → "Extent IoU ~0.65 on held-out data; 18/20 parcels on the tile you saw. Areas are geometrically exact once boundaries are found."
- **"Why is boundary IoU low (0.39)?"** → "Bunds are 1–2 pixel-thin lines, so IoU is naturally harsh — a 1px shift halves it. Parcel-level accuracy is what matters, and that's strong."
- **"Does it work on Indian farms?"** → "The pipeline is data-agnostic; retraining on local paddy imagery is a short step. The georeferencing and area logic are already universal."
- **"What if bunds are broken/faint?"** → "That's exactly what the watershed post-processing handles — it closes gaps to form whole fields."
- **"Real GPS?"** → "Yes — from the drone image's georeferencing. The GeoJSON opens on real coordinates in QGIS." *(show it)*
- **"Manual work needed?"** → "None. Upload to result is automatic; a human just eyeballs the overlay to verify."

## Demo failure backup
If the live app misbehaves: show the pre-rendered overlay PNG (`outputs/`) and the GeoJSON in QGIS. Never debug live — cut to the saved result and keep talking.
