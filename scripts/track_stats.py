"""
Optional evaluation helper: per-frame person count and number of active track IDs.

Writes CSV to outputs/ for plots or report tables (object count over time).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Export per-frame track stats to CSV.")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Input video path.")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs") / "track_stats.csv",
        help="Output CSV path.",
    )
    ap.add_argument("-m", "--model", type=str, default="yolov8n.pt")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    device = args.device or None

    rows: list[tuple[int, int, int, str]] = []
    for fi, r in enumerate(
        model.track(
            source=str(args.input),
            stream=True,
            persist=True,
            tracker=args.tracker,
            classes=[0],
            conf=args.conf,
            device=device,
            verbose=False,
        )
    ):
        n_det = 0
        n_ids = 0
        id_str = ""
        if r.boxes is not None and len(r.boxes):
            n_det = len(r.boxes)
            if r.boxes.id is not None:
                ids = r.boxes.id.cpu().int().tolist()
                uniq = sorted(set(ids))
                n_ids = len(uniq)
                id_str = ",".join(str(x) for x in uniq)
        rows.append((fi, n_det, n_ids, id_str))

    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "num_person_detections", "num_unique_track_ids", "track_ids"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
