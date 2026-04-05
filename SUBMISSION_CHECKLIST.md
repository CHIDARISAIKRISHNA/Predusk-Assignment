# Submission checklist (matches assignment rubric)

Graders care about **correctness, reasoning, pipeline design, and clarity**—not flashy UI.

---

## Mandatory deliverables

| # | Required | Your artifact (typical path) | You |
|---|----------|------------------------------|-----|
| 1 | **GitHub repository or zipped codebase** | Whole `Predesk-Assignment/` folder (no `.venv`) | Zip or push to GitHub |
| 2 | **README.md** | `README.md` (root) | Done in repo |
| 3 | **Annotated output video** | e.g. `outputs/cricket_tracked.mp4` (boxes + numeric IDs) | Generate + include in zip / LFS / link |
| 4 | **Original public video link** | Pexels page URL (also in README + report) | Done in docs; cite in demo |
| 5 | **Short technical report** | `docs/TECHNICAL_REPORT.md` (export PDF only if they ask) | Add your name; review |
| 6 | **Sample screenshots of results** | e.g. `outputs/screenshots/*.jpg` from `extract_screenshots.py` | Generate + include in zip |
| 7 | **Demo video 3–5 min [MUST]** | Your `.mp4` screen recording | Follow `docs/DEMO_SCRIPT.md` |

**Public video URL used in this project:**  
https://www.pexels.com/video/dynamic-cricket-practice-at-urban-outdoor-field-36088031/

**Regenerate core outputs (if needed):**

```bash
python src/track_video.py -i "<path-to-downloaded-pexels-video>.mp4" -o outputs/cricket_tracked.mp4
python scripts/extract_screenshots.py -i outputs/cricket_tracked.mp4 -o outputs/screenshots --n 6
```

---

## Optional enhancements (strong positives)

| Idea | In this repo | Command / doc |
|------|----------------|---------------|
| Trajectory visualization | Sparse footpoints + trails in heatmap script | `python src/trajectory_heatmap.py -i data/video.mp4 -o outputs/heatmap.png` |
| Movement heatmaps | Same script overlays density on last frame | ↑ |
| Top-view / bird’s-eye | Not implemented | Needs **homography** from image plane to pitch (calibration or known field geometry)—out of scope for minimal pipeline |
| Object count over time | CSV export | `python scripts/track_stats.py -i data/video.mp4 -o outputs/track_stats.csv` |
| Team / role clustering | Not implemented | Would need team labels, jersey OCR, or separate classifier |
| Speed / movement stats | Partial | `track_stats.csv` + bbox-derived speed would need extra script (pixel → m/s requires scale) |
| Simple evaluation metrics | Proxy counts in CSV | ↑ (not MOTA/IDF1 without ground truth) |
| Model comparison (two approaches) | ByteTrack vs BoT-SORT | `docs/MODEL_COMPARISON.md` + second run with `--tracker botsort.yaml` |
| Notebook / app / demo script | CLI demo | `src/track_video.py` is the runnable pipeline; add Jupyter only if you want |

---

## Packaging

**Zip contents (example):**

- `README.md`, `requirements.txt`, `.gitignore`, `SUBMISSION_CHECKLIST.md`
- `src/`, `scripts/`, `docs/`
- `outputs/cricket_tracked.mp4`, `outputs/screenshots/*.jpg`, **your demo `.mp4`**
- **Exclude:** `.venv/`, `__pycache__/`, huge `.pt` caches if policy allows (reviewers reinstall deps)

**GitHub:** Install [Git for Windows](https://git-scm.com/download/win) if needed, then `git init`, commit, push. Use **Git LFS** or **Drive link** in README for large videos if GitHub rejects file size.

**Report PDF (only if required):** VS Code / Markdown → PDF → e.g. `docs/TECHNICAL_REPORT.pdf`.
