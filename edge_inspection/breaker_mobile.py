from __future__ import annotations

from dataclasses import dataclass

from .events import Detection


@dataclass
class BreakerTrack:
    track_id: str
    bbox: tuple[float, float, float, float]
    last_frame_id: int


class MobileBreakerTracker:
    """Attach stable local IDs to breaker detections from a moving camera."""

    def __init__(self, *, iou_threshold: float = 0.25, max_missing_frames: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._tracks: dict[str, BreakerTrack] = {}
        self._next_track_id = 1

    def update(self, detections: list[Detection], *, frame_id: int) -> list[Detection]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_id - track.last_frame_id <= self.max_missing_frames
        }
        candidates = [item for item in detections if item.class_name.upper() == "MCB"]
        available = set(self._tracks)
        assigned_model_ids: set[str] = set()
        output: list[Detection] = []
        for detection in sorted(candidates, key=lambda item: item.bbox[0]):
            best_id = None
            model_track_id = detection.metadata.get("track_id")
            if model_track_id is not None:
                candidate_id = f"MCB-{model_track_id}"
                if candidate_id not in assigned_model_ids:
                    best_id = candidate_id
                    assigned_model_ids.add(candidate_id)
                    available.discard(candidate_id)
            best_iou = self.iou_threshold
            if best_id is None:
                for track_id in available:
                    overlap = bbox_iou(detection.bbox, self._tracks[track_id].bbox)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_id = track_id
            if best_id is None:
                while f"MCB-{self._next_track_id}" in self._tracks:
                    self._next_track_id += 1
                best_id = f"MCB-{self._next_track_id}"
                self._next_track_id += 1
            else:
                available.discard(best_id)
            self._tracks[best_id] = BreakerTrack(best_id, detection.bbox, frame_id)
            output.append(
                Detection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    metadata={**detection.metadata, "track_id": best_id, "asset_id": best_id},
                )
            )
        return output

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1


class MobileMcbStateGate:
    """Require a short, continuous CLOSED observation before exposing it."""

    def __init__(self, *, closed_confirm_seconds: float = 0.5, max_missing_frames: int = 15) -> None:
        self.closed_confirm_seconds = closed_confirm_seconds
        self.max_missing_frames = max_missing_frames
        self._closed_since_seconds: dict[str, float] = {}
        self._last_seen: dict[str, int] = {}

    def update(
        self,
        detections: list[Detection],
        *,
        frame_id: int,
        observed_at_seconds: float | None = None,
    ) -> list[Detection]:
        active_keys = set()
        output = []
        for detection in detections:
            key = str(
                detection.metadata.get("asset_id")
                or detection.metadata.get("track_id")
                or ""
            )
            if not key:
                output.append(detection)
                continue
            active_keys.add(key)
            self._last_seen[key] = frame_id
            if detection.class_name.upper() == "CLOSED":
                since = self._closed_since_seconds.setdefault(
                    key, observed_at_seconds if observed_at_seconds is not None else float(frame_id)
                )
                duration = (
                    observed_at_seconds - since
                    if observed_at_seconds is not None
                    else float(frame_id) - since
                )
                if duration < self.closed_confirm_seconds:
                    output.append(
                        Detection(
                            bbox=detection.bbox,
                            confidence=0.0,
                            class_id=-1,
                            class_name="UNKNOWN",
                            metadata={
                                **detection.metadata,
                                "observation_valid": False,
                                "raw_state": "CLOSED",
                                "closed_confirmation_seconds": max(0.0, duration),
                                "decision_basis": "closed_temporal_confirmation",
                            },
                        )
                    )
                    continue
            else:
                self._closed_since_seconds.pop(key, None)
            output.append(detection)

        for key in self._closed_since_seconds.keys() - active_keys:
            self._closed_since_seconds.pop(key, None)
        expired = [
            key
            for key, last_seen in self._last_seen.items()
            if key not in active_keys and frame_id - last_seen > self.max_missing_frames
        ]
        for key in expired:
            self._last_seen.pop(key, None)
            self._closed_since_seconds.pop(key, None)
        return output

    def reset(self) -> None:
        self._closed_since_seconds.clear()
        self._last_seen.clear()


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def classify_mcb_handle_geometry(
    frame,
    detections: list[Detection],
    *,
    handle_up_means_closed: bool = True,
) -> list[Detection]:
    """Read a vertical MCB handle after the device itself has been detected.

    A lowered handle occupies the lower part of the operating slot and means
    OPEN for the supported IEC-style MCBs. A raised handle means CLOSED.
    Skin-coloured occlusion and weak geometry are returned as UNKNOWN.
    """

    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    output: list[Detection] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        left = max(0, min(width, int(round(x1))))
        top = max(0, min(height, int(round(y1))))
        right = max(0, min(width, int(round(x2))))
        bottom = max(0, min(height, int(round(y2))))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x_start, x_end = int(0.16 * w), int(0.84 * w)
        upper = gray[int(0.32 * h) : int(0.50 * h), x_start:x_end]
        lower = gray[int(0.58 * h) : int(0.80 * h), x_start:x_end]
        shell = gray[int(0.08 * h) : int(0.34 * h), x_start:x_end]
        if min(upper.size, lower.size, shell.size) == 0:
            continue

        shell_brightness = float(np.median(shell))
        upper_dark = float((upper < 105).mean())
        lower_dark = float((lower < 105).mean())
        handle_position = float(
            (0.41 * upper_dark + 0.69 * lower_dark)
            / max(upper_dark + lower_dark, 1e-6)
        )

        visible = shell_brightness >= 115
        down_score = lower_dark - upper_dark
        up_score = upper_dark - lower_dark
        if not visible:
            state = "UNKNOWN"
            confidence = 0.0
        elif lower_dark >= 0.55 and down_score >= 0.18:
            is_up = False
            state = "CLOSED" if is_up == handle_up_means_closed else "OPEN"
            confidence = min(0.99, 0.62 + down_score * 0.45)
        elif upper_dark >= 0.48 and up_score >= 0.16:
            is_up = True
            state = "CLOSED" if is_up == handle_up_means_closed else "OPEN"
            confidence = min(0.99, 0.62 + up_score * 0.45)
        else:
            state = "UNKNOWN"
            confidence = 0.0
        output.append(
            Detection(
                bbox=detection.bbox,
                confidence=float(confidence),
                class_id=0 if state == "CLOSED" else 1 if state == "OPEN" else -1,
                class_name=state,
                metadata={
                    **detection.metadata,
                    "device_class": "MCB",
                    "device_confidence": detection.confidence,
                    "handle_position": handle_position,
                    "handle_upper_dark_ratio": upper_dark,
                    "handle_lower_dark_ratio": lower_dark,
                    "observation_valid": state != "UNKNOWN",
                    "decision_basis": "detected_mcb_handle_geometry",
                },
            )
        )
    return output


def classify_mcb_crops(
    frame,
    detections: list[Detection],
    classifier,
    *,
    confidence_threshold: float,
    unknown_margin: float,
    confirm_closed_geometry: bool = True,
    handle_up_means_closed: bool = True,
) -> list[Detection]:
    """Classify detected MCB crops and reject uncertain state observations."""

    height, width = frame.shape[:2]
    valid = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        left = max(0, min(width, int(round(x1))))
        top = max(0, min(height, int(round(y1))))
        right = max(0, min(width, int(round(x2))))
        bottom = max(0, min(height, int(round(y2))))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        valid.append((detection, crop))
    if not valid:
        return []
    crops = [crop for _, crop in valid]
    results = (
        classifier.predict_many(crops)
        if hasattr(classifier, "predict_many")
        else [classifier.predict(crop) for crop in crops]
    )
    output: list[Detection] = []
    for (detection, _), result in zip(valid, results, strict=True):
        probabilities = {
            str(name).upper(): float(value)
            for name, value in result.metadata.get("class_probabilities", {}).items()
        }
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ranked[0] if ranked else (result.class_name.upper(), result.confidence)
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        accepted = (
            best_name in {"OPEN", "CLOSED"}
            and best_score >= confidence_threshold
            and best_score - second_score >= unknown_margin
        )
        geometry_state = None
        if accepted and best_name == "CLOSED" and confirm_closed_geometry:
            geometry = classify_mcb_handle_geometry(
                frame,
                [detection],
                handle_up_means_closed=handle_up_means_closed,
            )
            geometry_state = geometry[0].class_name if geometry else "UNKNOWN"
            accepted = geometry_state == "CLOSED"
        state = best_name if accepted else "UNKNOWN"
        output.append(
            Detection(
                bbox=detection.bbox,
                confidence=float(best_score if accepted else 0.0),
                class_id=0 if state == "CLOSED" else 1 if state == "OPEN" else -1,
                class_name=state,
                metadata={
                    **detection.metadata,
                    "device_class": "MCB",
                    "device_confidence": detection.confidence,
                    "class_probabilities": probabilities,
                    "closed_geometry_confirmation": geometry_state,
                    "observation_valid": accepted,
                    "decision_basis": "detected_mcb_state_classifier",
                },
            )
        )
    return output
