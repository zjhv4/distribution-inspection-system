from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from edge_inspection.breaker_reference import save_breaker_reference
from edge_inspection.config import load_site_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture site breaker ROI reference images")
    parser.add_argument("--source", required=True, help="Video path or camera index")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--state", choices=["closed", "open"], required=True)
    parser.add_argument("--time", type=float, default=0.0, help="Video timestamp in seconds")
    parser.add_argument("--output-dir", default="runtime/breaker_references")
    args = parser.parse_args()

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    if args.time > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Cannot read reference frame")

    config = load_site_config(args.config)
    output_dir = Path(args.output_dir)
    for roi in config.breaker.rois:
        output = output_dir / f"{roi.name}_{args.state}.jpg"
        save_breaker_reference(frame, roi, output)
        print(output)


if __name__ == "__main__":
    main()
