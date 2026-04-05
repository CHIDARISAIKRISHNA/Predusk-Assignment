"""Extract evenly spaced frames from a video (for report screenshots)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", "-i", type=Path, required=True)
    ap.add_argument("--out-dir", "-o", type=Path, default=Path("outputs") / "screenshots")
    ap.add_argument("--n", type=int, default=6, help="Number of frames to save.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    n = max(1, args.n)
    indices = []
    if total > 0:
        indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    else:
        indices = list(range(n))

    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        path = args.out_dir / f"frame_{i:03d}_idx{frame_idx}.jpg"
        cv2.imwrite(str(path), frame)
        saved += 1
    cap.release()
    print(f"Saved {saved} images under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
