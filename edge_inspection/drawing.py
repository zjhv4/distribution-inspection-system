from __future__ import annotations

import cv2
import numpy as np

from .config import SiteConfig
from .events import AlertEvent, Detection


def draw_overlay(frame: np.ndarray, config: SiteConfig, detections: list[Detection], events: list[AlertEvent]) -> np.ndarray:
    output = frame.copy()

    for zone in config.intrusion.zones:
        points = np.array(zone.polygon, dtype=np.int32)
        cv2.polylines(output, [points], isClosed=True, color=(0, 220, 255), thickness=2)
        if len(points):
            cv2.putText(output, zone.name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        color = (0, 255, 0)
        if detection.class_name.upper() == "UNKNOWN":
            color = (0, 200, 255)
        if detection.class_name.upper() in {name.upper() for name in config.breaker.abnormal_classes}:
            color = (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for idx, event in enumerate(events[:3]):
        cv2.putText(
            output,
            event.message,
            (16, 32 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )
    return output
