# Technical report: multi-object detection and persistent ID tracking

**Author:** (add your name)  
**Length:** ~2 pages (markdown; export to PDF if the portal requires PDF).

---

## 0. Video source (mandatory citation)

**Public page URL (Pexels):**  
[Dynamic cricket practice at urban outdoor field](https://www.pexels.com/video/dynamic-cricket-practice-at-urban-outdoor-field-36088031/)

The clip was obtained with **Free download** on that page. Processing used the local file (e.g. `Downloads\video.mp4` or `data\video.mp4`); the **submission cites the Pexels page**, not a proprietary dataset.

**Scene characteristics:** outdoor urban cricket practice, **vertical (portrait)** framing, handheld or moving camera, **large scale difference** between players near the camera (batter, wicketkeeper) and smaller figures mid-pitch, **partial visibility** of close subjects, and **frequent proximity** between fielders—typical challenges for detection and association.

---

## 1. Detector

**Model:** Ultralytics **YOLOv8n** (`yolov8n.pt`), COCO-pretrained, via the `ultralytics` package.

**Target class:** COCO **person** (class index `0`). The ball is not tracked unless `--classes 0 32` is added.

**Rationale:** YOLOv8 gives a strong balance of **accuracy**, **speed**, and **maintainability** for a short assignment. The **nano** variant runs comfortably on CPU or modest GPU while still producing usable boxes on this clip. A larger checkpoint (`yolov8m.pt`) would likely improve small / distant players at the cost of latency.

**Frame processing:** `src/track_video.py` runs the detector and tracker on **every frame** of the input (e.g. **629** frames for the downloaded copy used in development) to maximize temporal continuity for IDs.

---

## 2. Tracker

**Primary:** **ByteTrack** (`bytetrack.yaml` in Ultralytics).

**Why ByteTrack:** It associates both **high-** and **low-confidence** detections. When **occlusion** or **motion blur** temporarily lowers scores, secondary detections still participate in matching, which reduces **broken tracks** compared with greedy high-threshold-only association.

**Alternative:** **BoT-SORT** (`botsort.yaml`) adds **appearance (ReID)** features. It can help when players **cross** or wear **similar clothing**, but ReID can degrade under **blur** or **compression**; it is kept as a one-flag swap in the codebase for comparison.

---

## 3. How ID consistency is maintained

1. **Detection:** Each frame yields person boxes and scores from YOLO.  
2. **Track state:** ByteTrack maintains a Kalman-style motion state per track and predicts the next position.  
3. **Association:** Detections are matched to tracks (IoU / cost matrix). High-confidence unmatched boxes create **new IDs**; low-confidence boxes can **recover** existing tracks instead of spawning duplicates.  
4. **Session persistence:** Ultralytics is called with **`persist=True`** so the same tracker session spans the **entire video**, avoiding a reset per batch.

**Stability** is good when motion is continuous and boxes reappear after short gaps. **ID swaps** are more likely when players **overlap for many frames**, **leave the frame** for long intervals, or when the **detector drops** one subject for several frames in a row.

---

## 4. Challenges observed on this cricket clip

| Challenge | Effect |
|-----------|--------|
| **Scale variation** | Near-camera subjects are large and often **truncated**; distant players are small and harder for `n` weights. |
| **Portrait aspect** | Letterboxing to the model input shape (`640×384` in logs) is normal; extreme aspect ratios can marginally affect box alignment. |
| **Crowding** | Several players stand close; boxes and IDs compete in the same image region. |
| **Varying “person count” in logs** | The console count is **detections above threshold this frame**, not a fixed ground-truth headcount—expected when people move, occlude, or fall below `conf`. |

---

## 5. Failure cases (what to look for in `outputs/cricket_tracked.mp4`)

- **ID switch** after two players cross or stand in a line with overlapping boxes.  
- **New ID** when a player was **lost** for many frames and reappears.  
- **Fragmented IDs** if confidence drops on fast motion (mitigated partly by ByteTrack’s second association pass).

For the submission demo, **pause on one good frame** (stable IDs) and **one harder segment** (overlap or motion) and describe what you see.

---

## 6. Possible improvements

1. **`yolov8m.pt` / `yolov8l.pt`** for better small-person recall.  
2. **BoT-SORT** (`--tracker botsort.yaml`) and qualitative comparison on the same file.  
3. **Confidence tuning** (`--conf 0.2`–`0.35`) to trade false positives vs missed detections.  
4. **Domain fine-tuning** on sports/broadcast data if labels were available.  
5. **Optional quantitative proxy:** `scripts/track_stats.py` exports per-frame detection and unique-ID counts to CSV for simple time-series plots (not ground-truth MOT metrics).

