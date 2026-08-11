from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark edge inference latency")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default=None, help="Video/camera source. Uses synthetic frames when omitted.")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--platform",
        default="generic",
        choices=["generic", "jetson"],
        help="Target platform label. Jetson reports hardware identity for deployable evidence.",
    )
    parser.add_argument("--output", default="runs/latency_report.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    capture = None
    if args.source is not None:
        source: str | int = int(args.source) if str(args.source).isdigit() else args.source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")

    latencies: list[float] = []
    total = args.frames + args.warmup
    for idx in range(total):
        if capture is None:
            frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        else:
            ok, frame = capture.read()
            if not ok:
                break

        start = perf_counter()
        model.predict(frame, imgsz=args.imgsz, device=args.device, conf=args.conf, iou=args.iou, verbose=False)
        elapsed_ms = (perf_counter() - start) * 1000
        if idx >= args.warmup:
            latencies.append(elapsed_ms)

    if capture is not None:
        capture.release()

    if not latencies:
        raise RuntimeError("No frames were benchmarked.")

    p95 = float(np.percentile(latencies, 95))
    avg = float(mean(latencies))
    report = {
        "weights": args.weights,
        "device": args.device,
        "platform": args.platform,
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "frames": len(latencies),
        "avg_latency_ms": round(avg, 2),
        "p95_latency_ms": round(p95, 2),
        "fps": round(1000.0 / avg, 2) if avg else 0,
        "meets_500ms_target": p95 <= 500,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
