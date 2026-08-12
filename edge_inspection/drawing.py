from __future__ import annotations

import cv2
import numpy as np

from .config import SiteConfig
from .events import AlertEvent, Detection


def draw_overlay(
    frame: np.ndarray,
    config: SiteConfig,
    detections: list[Detection],
    events: list[AlertEvent],
    *,
    draw_intrusion_zones: bool = True,
) -> np.ndarray:
    output = frame.copy()

    if draw_intrusion_zones:
        for zone in config.intrusion.zones:
            points = np.array(zone.polygon, dtype=np.int32)
            cv2.polylines(output, [points], isClosed=True, color=(0, 220, 255), thickness=2)
            if len(points):
                cv2.putText(output, zone.name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    ordered_detections = sorted(detections, key=lambda item: (item.bbox[1], item.bbox[0]))
    for detection_index, detection in enumerate(ordered_detections):
        x1, y1, x2, y2 = map(int, detection.bbox)
        color = (0, 255, 0)
        device_class = str(detection.metadata.get("device_class", "")).upper()
        class_name = detection.class_name.upper()
        if device_class == "MCB" and class_name == "CLOSED":
            color = (0, 0, 255)
        elif device_class == "MCB" and class_name == "UNKNOWN":
            color = (0, 200, 255)
        elif class_name == "RCD":
            color = (255, 180, 0)
        elif class_name == "ISOLATOR":
            color = (210, 210, 210)
        elif class_name in {name.upper() for name in config.breaker.abnormal_classes}:
            color = (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        if device_class == "MCB":
            state_code = {"OPEN": "O", "CLOSED": "C", "UNKNOWN": "?"}.get(class_name, "?")
            label = f"MCB {state_code}"
        elif class_name == "ISOLATOR":
            label = "ISO"
        else:
            label = detection.class_name.upper()
        font_scale = 0.38
        thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        lane_offset = 4 + (detection_index % 2) * (label_height + 6)
        label_bottom = max(label_height + baseline + 2, y1 - lane_offset)
        label_top = label_bottom - label_height - baseline - 4
        label_right = min(output.shape[1] - 1, x1 + label_width + 6)
        cv2.rectangle(output, (x1, label_top), (label_right, label_bottom), (20, 20, 20), -1)
        cv2.putText(
            output,
            label,
            (x1 + 3, label_bottom - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

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
