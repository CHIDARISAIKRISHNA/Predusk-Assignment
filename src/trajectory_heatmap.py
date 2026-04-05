"""
Optional: accumulate per-track footpoint (bbox bottom-center) and save a heatmap overlay.

Run after you have a working environment; uses the same YOLO+tracker stack as track_video.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trajectory points + movement heatmap from tracking.")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path. Default: outputs/<stem>_heatmap.png",
    )
    p.add_argument("--model", "-m", type=str, default="yolov8n.pt")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--classes", type=int, nargs="*", default=[0])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    out = args.output
    if out is None:
        out = Path("outputs") / f"{args.input.stem}_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(args.model)
    device = args.device or None

    heat = np.zeros((h, w), dtype=np.float32)
    traj_img = np.zeros((h, w, 3), dtype=np.uint8)
    last_frame = None

    results_iter = model.track(
        source=str(args.input),
        stream=True,
        persist=True,
        tracker=args.tracker,
        classes=args.classes if args.classes else None,
        conf=args.conf,
        device=device,
        verbose=False,
    )

    for r in results_iter:
        img = getattr(r, "orig_img", None)
        last_frame = img if img is not None else r.plot()
        if r.boxes is None or r.boxes.id is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        for box, tid in zip(xyxy, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            heat[cy, cx] += 1.0
            color = (
                int(50 + (tid * 37) % 200),
                int(80 + (tid * 17) % 170),
                int(100 + (tid * 91) % 155),
            )
            cv2.circle(traj_img, (cx, cy), 2, color, -1)

    if last_frame is None:
        print("No frames processed.", file=sys.stderr)
        return 1

    # Smooth heatmap for visualization
    heat_u8 = np.clip(heat / (heat.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    heat_blur = cv2.GaussianBlur(heat_u8, (0, 0), sigmaX=15, sigmaY=15)
    heat_color = cv2.applyColorMap(heat_blur, cv2.COLORMAP_INFERNO)
    base = last_frame.copy() if last_frame.ndim == 3 else cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 0.65, heat_color, 0.35, 0)
    combo = cv2.addWeighted(overlay, 0.85, traj_img, 0.45, 0)

    cv2.imwrite(str(out), combo)
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
