from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2

from .alarm import JsonlAlarmSink
from .breaker import BreakerStateDetector
from .breaker_mobile import (
    MobileBreakerTracker,
    MobileMcbStateGate,
    classify_mcb_crops,
    filter_visible_breaker_controls,
    merge_breaker_detections,
    select_coherent_breaker_row,
)
from .config import SiteConfig, TaskName
from .drawing import draw_overlay
from .events import AlertEvent, Detection
from .intrusion import IntrusionDetector
from .model import YoloClassifier, YoloDetector


def run_video(
    *,
    source: str | int,
    config: SiteConfig,
    task: TaskName,
    display: bool = False,
    output: str | Path | None = None,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    writer = None
    if output:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    intrusion_model = None
    breaker_model = None
    breaker_tiled_model = None
    breaker_mobile_tracker = None
    breaker_mobile_state_gate = None
    if task in ("intrusion", "all"):
        intrusion_model = YoloDetector(
            resolve_intrusion_weights(config),
            device=config.runtime.device,
            imgsz=config.runtime.imgsz,
            half=config.runtime.half,
        )
        intrusion = IntrusionDetector(config.intrusion)
    if task in ("breaker", "all"):
        if not config.models.breaker:
            raise RuntimeError("breaker.mode=mobile_detection requires models.breaker.")
        if not config.models.breaker_state_classifier:
            raise RuntimeError(
                "breaker.mode=mobile_detection requires models.breaker_state_classifier."
            )
        breaker_model = YoloDetector(
            config.models.breaker,
            device=config.runtime.device,
            imgsz=config.runtime.imgsz,
            half=config.runtime.half,
        )
        if config.breaker.mobile_tiled_detection:
            breaker_tiled_model = YoloDetector(
                config.models.breaker_far or config.models.breaker,
                device=config.runtime.device,
                imgsz=config.runtime.imgsz,
                half=config.runtime.half,
            )
        breaker_mobile_tracker = MobileBreakerTracker(
            iou_threshold=config.breaker.mobile_tracker_iou_threshold,
            max_missing_frames=config.breaker.mobile_tracker_max_missing_frames,
        )
        breaker_mobile_state_gate = MobileMcbStateGate(
            closed_confirm_seconds=config.breaker.mobile_closed_confirm_seconds,
            max_missing_frames=config.breaker.mobile_tracker_max_missing_frames,
        )
        breaker_mobile_state_classifier = YoloClassifier(
            config.models.breaker_state_classifier,
            device=config.runtime.device,
            imgsz=config.breaker.classifier_imgsz,
            half=config.runtime.half,
        )
        breaker = BreakerStateDetector(config.breaker)

    alarm_sink = JsonlAlarmSink(config.alarm)
    frame_id = 0
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_id += 1
        position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        observed_at_seconds = (
            position_ms / 1000.0 if position_ms > 0 else (frame_id - 1) / source_fps
        )
        start = perf_counter()

        detections: list[Detection] = []
        events: list[AlertEvent] = []

        if intrusion_model is not None:
            intrusion_detections = intrusion_model.predict(
                frame,
                conf=config.intrusion.confidence,
                iou=config.intrusion.iou,
                track=config.intrusion.use_model_tracking,
                tracker=config.intrusion.tracker,
            )
            detections.extend(intrusion_detections)
            events.extend(intrusion.update(intrusion_detections, frame_id=frame_id))

        if breaker_model is not None:
            breaker_detections = breaker_model.predict(
                frame,
                conf=config.breaker.confidence,
                iou=config.breaker.iou,
                track=breaker_mobile_tracker is not None,
                tracker=config.breaker.mobile_tracker,
            )
            breaker_detections = merge_breaker_detections(breaker_detections)
            used_tiled_detection = False
            if (
                breaker_tiled_model is not None
                and len(breaker_detections) < config.breaker.mobile_tiled_trigger_count
            ):
                used_tiled_detection = True
                tiled_detections = breaker_tiled_model.predict_tiled(
                    frame,
                    conf=config.breaker.mobile_tiled_confidence,
                    iou=config.breaker.iou,
                    columns=config.breaker.mobile_tiled_columns,
                    rows=config.breaker.mobile_tiled_rows,
                    overlap=config.breaker.mobile_tiled_overlap,
                )
                tiled_detections = merge_breaker_detections(tiled_detections)
                tiled_detections = select_coherent_breaker_row(
                    tiled_detections,
                    minimum_devices=config.breaker.mobile_tiled_minimum_row_devices,
                )
                breaker_detections = merge_breaker_detections(
                    [*breaker_detections, *tiled_detections]
                )
            breaker_detections = filter_visible_breaker_controls(frame, breaker_detections)
            if (
                used_tiled_detection
                and len(breaker_detections)
                < config.breaker.mobile_tiled_minimum_row_devices
            ):
                breaker_detections = []
            if config.breaker.mobile_class_limits:
                limited = []
                for class_name, limit in config.breaker.mobile_class_limits.items():
                    candidates = [
                        item
                        for item in breaker_detections
                        if item.class_name.upper() == class_name and limit > 0
                    ]
                    limited.extend(
                        sorted(candidates, key=lambda item: item.confidence, reverse=True)[:limit]
                    )
                breaker_detections = limited
            device_detections = breaker_detections
            mcb_detections = [
                item
                for item in device_detections
                if item.class_name.upper()
                in {name.upper() for name in config.breaker.mobile_device_classes}
            ]
            mcb_detections = breaker_mobile_tracker.update(
                mcb_detections,
                frame_id=frame_id,
            )
            breaker_detections = classify_mcb_crops(
                frame,
                mcb_detections,
                breaker_mobile_state_classifier,
                confidence_threshold=config.breaker.mobile_state_confidence,
                unknown_margin=config.breaker.mobile_state_unknown_margin,
                confirm_closed_geometry=config.breaker.mobile_confirm_closed_geometry,
                handle_up_means_closed=config.breaker.handle_up_means_closed,
            )
            breaker_detections = breaker_mobile_state_gate.update(
                breaker_detections,
                frame_id=frame_id,
                observed_at_seconds=observed_at_seconds,
            )
            detections.extend(
                item
                for item in device_detections
                if item.class_name.upper() not in {
                    name.upper() for name in config.breaker.mobile_device_classes
                }
            )
            detections.extend(breaker_detections)
            events.extend(
                breaker.update(
                    breaker_detections,
                    frame_id=frame_id,
                    observed_at_seconds=observed_at_seconds,
                )
            )

        latency_ms = (perf_counter() - start) * 1000
        for event in events:
            event.metadata["latency_ms"] = round(latency_ms, 2)
            alarm_sink.emit(event, frame)

        annotated = draw_overlay(
            frame,
            config,
            detections,
            events,
            draw_intrusion_zones=task in ("intrusion", "all"),
        )
        if writer is not None:
            writer.write(annotated)
        if display:
            cv2.imshow("edge-inspection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    alarm_sink.close(flush=True, timeout_seconds=5.0)
    capture.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()


def resolve_intrusion_weights(config: SiteConfig) -> str:
    """Select an explicitly declared camera-domain model without silent fallback."""
    profile = config.intrusion.model_profile
    if profile is None:
        return config.models.intrusion
    try:
        return config.models.intrusion_profiles[profile]
    except KeyError as exc:
        available = ", ".join(sorted(config.models.intrusion_profiles)) or "<none>"
        raise RuntimeError(
            f"intrusion.model_profile={profile!r} has no matching model; available: {available}"
        ) from exc
