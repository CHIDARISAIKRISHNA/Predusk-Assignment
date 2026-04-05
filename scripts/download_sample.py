"""
Download a short public-domain style sample clip for quick pipeline tests.

For the assignment submission, prefer a sports clip from the README-linked sources
and cite that URL in your report instead of this sample.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Remote file for `urlretrieve` only — not a local path. Your own clip stays in data/;
# run: python src/track_video.py -i data/video.mp4 -o outputs/tracked.mp4
DEFAULT_URL = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data") / "sample_640x360.mp4",
        help="Where to save the file.",
    )
    ap.add_argument("--url", type=str, default=DEFAULT_URL)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url} -> {args.output}")
    urllib.request.urlretrieve(args.url, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
