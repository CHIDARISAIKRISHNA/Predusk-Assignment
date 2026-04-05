"""
Multi-object detection and persistent ID tracking for sports / event footage.

Uses Ultralytics YOLO for detection and a configurable tracker (default: ByteTrack)
for cross-frame identity association.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect and track people (and optional classes) in a video."
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input video, or stream URL (e.g. https://...).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output annotated video path. Default: outputs/<input_stem>_tracked.mp4",
    )
    p.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="YOLO weights (e.g. yolov8n.pt, yolo11n.pt, yolov8m.pt) or path to custom .pt.",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config name or path (bytetrack.yaml, botsort.yaml).",
    )
    p.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=[0],
        help="COCO class IDs to keep. Default: 0 (person). Example: --classes 0 32 for person+ball.",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda:0, cpu, or mps. Empty = auto.",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per image (raise for crowded scenes).",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference on CUDA (faster, slightly less stable on some GPUs).",
    )
    p.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width for visualization.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = args.input
    if not str(inp).startswith(("http://", "https://")) and not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        print(
            "  Use a real path to your video, e.g.:\n"
            "    python scripts/download_sample.py\n"
            "    python src/track_video.py -i data/sample_640x360.mp4 -o outputs/out.mp4\n"
            "  Or download sports footage (see README), save as data/yourname.mp4, and pass that path.",
            file=sys.stderr,
        )
        return 1

    out = args.output
    if out is None:
        if str(inp).startswith(("http://", "https://")):
            stem = "video"
        else:
            stem = inp.stem if inp.suffix else "video"
        out = Path("outputs") / f"{stem}_tracked.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    from ultralytics import YOLO

    video_fps = 30.0
    if not str(inp).startswith(("http://", "https://")):
        cap_meta = cv2.VideoCapture(str(inp))
        if cap_meta.isOpened():
            vf = cap_meta.get(cv2.CAP_PROP_FPS)
            if vf and vf > 1.0:
                video_fps = float(vf)
            cap_meta.release()

    model = YOLO(args.model)
    device = args.device or None

    # stream=True yields results per frame; persist=True keeps track IDs across frames
    results_iter = model.track(
        source=str(inp),
        stream=True,
        persist=True,
        tracker=args.tracker,
        classes=args.classes if args.classes else None,
        conf=args.conf,
        iou=args.iou,
        device=device,
        max_det=args.max_det,
        half=args.half,
        line_width=args.line_width,
        verbose=True,
    )

    writer = None
    n_frames = 0

    for r in results_iter:
        frame = r.plot()
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out), fourcc, video_fps, (w, h))
        writer.write(frame)
        n_frames += 1

    if writer is not None:
        writer.release()

    if n_frames == 0:
        print("No frames written — check input path, codec, or network URL.", file=sys.stderr)
        return 1

    print(f"Wrote {n_frames} frames to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
