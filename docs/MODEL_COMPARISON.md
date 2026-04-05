# Model comparison (optional): two tracking approaches

Same detector (**YOLOv8n**), two association strategies available in this repo.

## A — ByteTrack (default)

```bash
python src/track_video.py -i data/video.mp4 -o outputs/track_bytetrack.mp4 --tracker bytetrack.yaml
```

**Strengths:** Uses high- and low-confidence boxes in a second association pass, which often **recovers tracks** through short occlusions or blur without needing a separate ReID network.

**Weaknesses:** Relies heavily on **motion (IoU)**; when players **cross** or look **very similar**, IDs can still swap.

## B — BoT-SORT

```bash
python src/track_video.py -i data/video.mp4 -o outputs/track_botsort.mp4 --tracker botsort.yaml
```

**Strengths:** Adds **appearance (ReID)** features on top of motion, which can **stabilize IDs** when trajectories intersect or kits look alike.

**Weaknesses:** More compute; ReID features can be **noisy** on **motion blur**, **compression**, or **tiny** distant players—sometimes worse than ByteTrack on the same clip.

## How to compare fairly

Use the **same** input file, **same** `--conf` / `--iou`, then skim both outputs for **ID continuity** in difficult segments (overlap, fast motion). A short side-by-side or “before/after” clip is enough for the assignment; no ground-truth MOT metrics are required unless you add labels.
