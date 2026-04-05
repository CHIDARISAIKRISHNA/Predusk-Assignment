# Multi-object detection and persistent ID tracking

Pipeline for **detecting people (and optional COCO classes)** in sports or public-event video and **assigning persistent track IDs** across frames, using **Ultralytics YOLO** plus **ByteTrack** (or **BoT-SORT**) multi-object tracking.

## Video used for this project (cite in your report / demo)

**Source (Pexels, public):** [Dynamic cricket practice at urban outdoor field](https://www.pexels.com/video/dynamic-cricket-practice-at-urban-outdoor-field-36088031/)

On that page, use **Free download**, save the file as e.g. `data/video.mp4`, then run:

```bash
python src/track_video.py -i data/video.mp4 -o outputs/video_tracked.mp4
```

**Example (Windows, file in Downloads, output name you used in class):**

```bash
python src/track_video.py -i "C:\Users\chida\Downloads\video.mp4" -o outputs/cricket_tracked.mp4
```

## Other public video options

| Source | Example |
|--------|--------|
| Pexels (cricket) | [Cricket practice (this project)](https://www.pexels.com/video/dynamic-cricket-practice-at-urban-outdoor-field-36088031/) |
| Pexels (football) | [Football players on the field](https://www.pexels.com/video/football-players-playing-on-the-field-10246153/) |
| Pexels search | [Sports videos](https://www.pexels.com/search/videos/sport/) |

**Submission checklist:** include the **exact page URL** you used, not just a filename. Only use content that is clearly public / licensed for your use.

## Installation

Requires **Python 3.10+** (3.11 recommended).

```bash
cd Predesk-Assignment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

## Dependencies
pip install -r requirements.txt
```

On first run, YOLO weights (e.g. `yolov8n.pt`) download automatically into your Ultralytics cache.

## How to run

### 1) (Optional) Quick smoke test clip

`my_video.mp4` in the examples below is **only a placeholder**. Either save your real file under that name or pass whatever path you use (e.g. `C:\Videos\football.mp4`).

A tiny sample file (not sports) to verify the install:

```bash
python scripts/download_sample.py
python src/track_video.py -i data/video.mp4 -o outputs/Output_Video.mp4
```

### 2) Main pipeline — annotated output video

Use the path to **your** downloaded sports clip (any folder/name is fine):

```bash
python src/track_video.py -i data/my_video.mp4 -o outputs/my_video_tracked.mp4
```

**Track only people (default):** `--classes 0` (COCO class `person`).

**Also track sports ball (COCO class 32):**

```bash
python src/track_video.py -i data/my_video.mp4 -o outputs/out.mp4 --classes 0 32
```

**Stronger model (better boxes, slower):**

```bash
python src/track_video.py -i data/my_video.mp4 -m yolov8m.pt
```

**Alternative tracker** (sometimes more stable on fast motion / camera shake; try both on your clip):

```bash
python src/track_video.py -i data/my_video.mp4 --tracker botsort.yaml
```

**GPU:** omit `--device` for auto-selection, or pass e.g. `--device cuda:0` / `--device mps`.

### 3) Screenshots for the report

After you have the tracked video (e.g. `outputs/cricket_tracked.mp4`):

```bash
python scripts/extract_screenshots.py -i outputs/cricket_tracked.mp4 -o outputs/screenshots --n 6
```

### 4) Optional: trajectory + movement heatmap (enhancement)

```bash
python src/trajectory_heatmap.py -i data/video.mp4 -o outputs/cricket_heatmap.png
```

### 5) Optional: detection / ID counts over time (CSV)

```bash
python scripts/track_stats.py -i data/video.mp4 -o outputs/track_stats.csv
```

## Repository layout

| Path | Role |
|------|------|
| `src/track_video.py` | Main detection + tracking + annotated MP4 writer |
| `src/trajectory_heatmap.py` | Optional heatmap / sparse trajectory overlay |
| `scripts/download_sample.py` | Tiny sample download for install checks |
| `scripts/extract_screenshots.py` | Evenly spaced JPEGs from a video |
| `scripts/track_stats.py` | Optional CSV: per-frame person detections & unique track IDs |
| `docs/TECHNICAL_REPORT.md` | 1–2 page write-up for submission |
| `docs/DEMO_SCRIPT.md` | Speaking outline for the required 3–5 min screen recording |
| `docs/MODEL_COMPARISON.md` | Optional: ByteTrack vs BoT-SORT, same detector |
| `SUBMISSION_CHECKLIST.md` | **Rubric-aligned** mandatory + optional checklist |
| `requirements.txt` | Python dependencies |

## Assumptions

- Input is a **standard video file** (e.g. `.mp4`) readable by OpenCV, or a **direct** stream URL supported by Ultralytics (a **web page** URL such as Pexels is not a valid `-i` source until you download the file).
- **“Person”** is the relevant subject class (COCO `0`); the model is **not** fine-tuned for cricket-specific poses or uniforms.
- **Evaluation** is **qualitative** unless you add labeled data: we do not claim MOT benchmark numbers (MOTA/IDF1) without ground truth.
- **Hardware:** CPU is supported; GPU reduces runtime. First run **downloads** `yolov8n.pt` automatically.

## Limitations

- **ID switches** and **fragmentation** can occur under heavy **occlusion**, **similar appearance**, or **missed detections** for several frames.
- **yolov8n** may **miss or jitter** on very small / distant players compared with larger weights.
- Output is encoded with **OpenCV `mp4v`**; some players prefer H.264—if needed, re-encode with FFmpeg for wider compatibility.
- The pipeline does **not** infer **team**, **role**, or **ball trajectory** unless you extend classes and logic.

## Model and tracker choices (summary)

- **Detector:** YOLOv8 / YOLO11 (Ultralytics), COCO-pretrained. Default `yolov8n.pt` balances speed and ease; upgrade to `yolov8s.pt` / `yolov8m.pt` or `yolo11*.pt` for harder scenes.
- **Tracker:** **ByteTrack** (`bytetrack.yaml`) associates high- and low-confidence detections to reduce fragmentations; **BoT-SORT** (`botsort.yaml`) adds appearance (ReID) cues and can help when players look similar or cross often.

Details, experiments on the cricket clip, and citations are in **`docs/TECHNICAL_REPORT.md`**. Optional **two-tracker comparison**: **`docs/MODEL_COMPARISON.md`**.


