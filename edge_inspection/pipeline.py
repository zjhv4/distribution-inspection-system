from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2

from .alarm import JsonlAlarmSink
from .anomaly import DinoReferenceAnomalyScorer, ReconstructionAnomalyScorer
from .breaker import BreakerStateDetector
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
    breaker_classifier = None
    breaker_anomaly_scorer = None
    if task in ("intrusion", "all"):
        intrusion_model = YoloDetector(
            resolve_intrusion_weights(config),
            device=config.runtime.device,
            imgsz=config.runtime.imgsz,
            half=config.runtime.half,
        )
        intrusion = IntrusionDetector(config.intrusion)
    if task in ("breaker", "all"):
        if config.breaker.mode in {"roi_classification", "roi_hybrid"}:
            if not config.models.breaker_classifier:
                raise RuntimeError(f"breaker.mode={config.breaker.mode} requires models.breaker_classifier.")
            if not config.breaker.rois:
                raise RuntimeError(f"breaker.mode={config.breaker.mode} requires at least one breaker.rois entry.")
            breaker_classifier = YoloClassifier(
                config.models.breaker_classifier,
                device=config.runtime.device,
                imgsz=config.breaker.classifier_imgsz,
                half=config.runtime.half,
            )
            if config.breaker.mode == "roi_hybrid":
                breaker_anomaly_scorer = build_breaker_anomaly_scorer(config)
        elif config.breaker.mode in {"detection", "detection_hybrid"}:
            breaker_model = YoloDetector(
                config.models.breaker,
                device=config.runtime.device,
                imgsz=config.runtime.imgsz,
                half=config.runtime.half,
            )
            if config.breaker.mode == "detection_hybrid":
                breaker_anomaly_scorer = build_breaker_anomaly_scorer(config)
        else:
            raise RuntimeError(f"Unsupported breaker.mode: {config.breaker.mode}")
        breaker = BreakerStateDetector(config.breaker)

    alarm_sink = JsonlAlarmSink(config.alarm)
    frame_id = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_id += 1
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
            breaker_detections = breaker_model.predict(frame, conf=config.breaker.confidence, iou=config.breaker.iou)
            breaker_detections = assign_breaker_assets(breaker_detections, config)
            if breaker_anomaly_scorer is not None:
                breaker_detections = score_breaker_detection_crops(
                    frame,
                    breaker_detections,
                    breaker_anomaly_scorer,
                    config,
                )
            detections.extend(breaker_detections)
            events.extend(breaker.update(breaker_detections, frame_id=frame_id))
        elif breaker_classifier is not None:
            breaker_detections = classify_breaker_rois(
                frame,
                breaker_classifier,
                config,
                anomaly_scorer=breaker_anomaly_scorer,
            )
            detections.extend(breaker_detections)
            events.extend(breaker.update(breaker_detections, frame_id=frame_id))

        latency_ms = (perf_counter() - start) * 1000
        for event in events:
            event.metadata["latency_ms"] = round(latency_ms, 2)
            alarm_sink.emit(event, frame)

        annotated = draw_overlay(frame, config, detections, events)
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


def classify_breaker_rois(
    frame,
    classifier: YoloClassifier,
    config: SiteConfig,
    *,
    anomaly_scorer: ReconstructionAnomalyScorer | None = None,
) -> list[Detection]:
    detections: list[Detection] = []
    height, width = frame.shape[:2]
    for roi in config.breaker.rois:
        x1, y1, x2, y2 = roi.bbox
        left = max(0, min(width, int(round(x1))))
        top = max(0, min(height, int(round(y1))))
        right = max(0, min(width, int(round(x2))))
        bottom = max(0, min(height, int(round(y2))))
        if right <= left or bottom <= top:
            continue

        crop = frame[top:bottom, left:right]
        result = classifier.predict(crop)
        probabilities = {
            str(name).upper(): float(probability)
            for name, probability in result.metadata.get("class_probabilities", {}).items()
        }
        alarm_candidates = []
        for class_name in config.breaker.abnormal_classes:
            normalized_name = class_name.upper()
            probability = probabilities.get(normalized_name)
            threshold = config.breaker.class_thresholds.get(
                normalized_name, config.breaker.classifier_confidence
            )
            if probability is not None and probability >= threshold:
                alarm_candidates.append((probability, normalized_name))

        if alarm_candidates:
            confidence, class_name = max(alarm_candidates)
            class_id = next(
                (
                    index
                    for index, name in enumerate(probabilities)
                    if name == class_name
                ),
                result.class_id,
            )
        else:
            confidence = result.confidence
            class_name = result.class_name.upper()
            class_id = result.class_id
            if confidence < config.breaker.classifier_confidence:
                continue
        metadata = {"roi_name": roi.name, "class_probabilities": probabilities}
        if anomaly_scorer is not None:
            closed_probability = max(
                (probabilities.get(name.upper(), 0.0) for name in config.breaker.closed_classes),
                default=0.0,
            )
            metadata.update(
                anomaly_scorer.score(
                    crop,
                    asset_id=roi.name,
                    allow_calibration=(
                        class_name in {name.upper() for name in config.breaker.closed_classes}
                        and closed_probability >= config.breaker.anomaly_normal_confidence
                    ),
                )
            )
        detections.append(
            Detection(
                bbox=(float(left), float(top), float(right), float(bottom)),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                metadata=metadata,
            )
        )
    return detections


def assign_breaker_assets(detections: list[Detection], config: SiteConfig) -> list[Detection]:
    """Attach a stable configured asset key to state detections in detection mode."""
    if not config.breaker.rois:
        return detections
    assigned: list[Detection] = []
    for detection in detections:
        center_x = (detection.bbox[0] + detection.bbox[2]) / 2
        center_y = (detection.bbox[1] + detection.bbox[3]) / 2
        for roi in config.breaker.rois:
            x1, y1, x2, y2 = roi.bbox
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                assigned.append(
                    Detection(
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        metadata={**detection.metadata, "roi_name": roi.name, "asset_id": roi.name},
                    )
                )
                break
    return assigned


def build_breaker_anomaly_scorer(config: SiteConfig):
    if not config.models.breaker_anomaly:
        raise RuntimeError(f"breaker.mode={config.breaker.mode} requires models.breaker_anomaly.")
    if config.breaker.anomaly_backend == "dinov2_reference":
        return DinoReferenceAnomalyScorer(
            config.models.breaker_anomaly,
            device=config.runtime.device,
            imgsz=config.breaker.classifier_imgsz,
            bootstrap_frames=config.breaker.anomaly_bootstrap_frames,
            bank_size=config.breaker.anomaly_bank_size,
            sample_stride=config.breaker.anomaly_sample_stride,
            neighbors=config.breaker.anomaly_neighbors,
            normal_quantile=config.breaker.anomaly_normal_quantile,
            min_raw_threshold=config.breaker.anomaly_min_raw_threshold,
        )
    if not config.models.breaker_anomaly_calibration:
        raise RuntimeError(
            "reconstruction anomaly backend requires models.breaker_anomaly_calibration."
        )
    return ReconstructionAnomalyScorer(
        config.models.breaker_anomaly,
        config.models.breaker_anomaly_calibration,
        device=config.runtime.device,
        imgsz=config.breaker.classifier_imgsz,
    )


def score_breaker_detection_crops(frame, detections, anomaly_scorer, config: SiteConfig) -> list[Detection]:
    height, width = frame.shape[:2]
    closed_names = {name.upper() for name in config.breaker.closed_classes}
    scored: list[Detection] = []
    for detection in detections:
        left = max(0, min(width, int(round(detection.bbox[0]))))
        top = max(0, min(height, int(round(detection.bbox[1]))))
        right = max(0, min(width, int(round(detection.bbox[2]))))
        bottom = max(0, min(height, int(round(detection.bbox[3]))))
        if right <= left or bottom <= top:
            scored.append(detection)
            continue
        asset_id = str(
            detection.metadata.get("asset_id")
            or detection.metadata.get("roi_name")
            or f"bbox_{left}_{top}_{right}_{bottom}"
        )
        anomaly_metadata = anomaly_scorer.score(
            frame[top:bottom, left:right],
            asset_id=asset_id,
            allow_calibration=(
                detection.class_name.upper() in closed_names
                and detection.confidence >= config.breaker.anomaly_normal_confidence
            ),
        )
        scored.append(
            Detection(
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
                metadata={**detection.metadata, **anomaly_metadata},
            )
        )
    return scored
