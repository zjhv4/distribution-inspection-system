from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
import json
from math import hypot
from typing import Any
from zoneinfo import ZoneInfo

from .config import AccessWindowConfig, IntrusionConfig, ZoneConfig
from .events import AlertEvent, Detection
from .geometry import bbox_anchor_point, point_in_polygon


@dataclass
class ZoneTrackState:
    hit_frames: int = 0
    miss_frames: int = 0
    active: bool = False
    event_id: str | None = None
    last_confidence: float = 0.0
    last_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonTrack:
    track_id: str
    bbox: tuple[float, float, float, float]
    last_frame_id: int


@dataclass(frozen=True)
class AccessDecision:
    permitted: bool
    alert_type: str | None
    reason_code: str
    verification_status: str
    identity_id: str | None
    identity_status: str
    within_access_window: bool | None


class IntrusionDetector:
    """Per-person electronic-fence and access-policy event detector.

    A person detector establishes only the spatial relation.  Illegal-entry
    semantics are derived from zone policy, upstream identity/credential
    metadata and local access windows.  Track IDs from an upstream tracker are
    preferred; otherwise a lightweight IoU/centroid tracker keeps event
    lifecycles separate for simultaneous people.
    """

    def __init__(self, config: IntrusionConfig):
        self.config = config
        self._states: dict[tuple[str, str], ZoneTrackState] = {}
        self._tracks: dict[str, PersonTrack] = {}
        self._zone_membership: dict[tuple[str, str], bool] = {}
        self._next_track_id = 1
        self._identity_context: dict[str, Any] = {}
        self._identity_context_mtime_ns: int | None = None

    def update(
        self,
        detections: list[Detection],
        *,
        frame_id: int,
        observed_at: datetime | None = None,
    ) -> list[AlertEvent]:
        observed_at = self._local_time(observed_at)
        person_names = {name.upper() for name in self.config.person_class_names}
        people = [item for item in detections if item.class_name.upper() in person_names]
        tracked = self._assign_tracks(people, frame_id)
        evaluated_keys: set[tuple[str, str]] = set()
        events: list[AlertEvent] = []

        for detection, track_id in tracked:
            anchor = bbox_anchor_point(detection.bbox, self.config.footpoint_ratio)
            for zone in self.config.zones:
                key = (zone.name, track_id)
                inside = point_in_polygon(anchor, zone.polygon)
                previous_inside = self._zone_membership.get(key)
                self._zone_membership[key] = inside
                evaluated_keys.add(key)

                if not inside:
                    self._record_miss(key, frame_id, events)
                    continue

                policy_metadata = {
                    **detection.metadata,
                    **self._identity_metadata(track_id, observed_at),
                }
                decision = evaluate_access(zone, policy_metadata, observed_at)
                if decision.permitted:
                    self._record_miss(key, frame_id, events)
                    continue

                state = self._states.setdefault(key, ZoneTrackState())
                state.hit_frames += 1
                state.miss_frames = 0
                state.last_confidence = detection.confidence
                relation = (
                    "CROSSED_IN"
                    if previous_inside is False
                    else "INITIAL_PRESENCE"
                    if previous_inside is None
                    else "INSIDE"
                )
                metadata = {
                    "zone": zone.name,
                    "track_id": track_id,
                    "bbox": detection.bbox,
                    "anchor": anchor,
                    "spatial_relation": relation,
                    "access_policy": zone.access_policy,
                    "reason_code": decision.reason_code,
                    "verification_status": decision.verification_status,
                    "identity_id": decision.identity_id,
                    "identity_status": decision.identity_status,
                    "within_access_window": decision.within_access_window,
                    "observed_at": observed_at.isoformat(),
                    "confirmation_frames": state.hit_frames,
                    "alert_type": decision.alert_type or "PERSON_INTRUSION",
                    "authorization_source": policy_metadata.get(
                        "authorization_source", "detection_metadata"
                    ),
                }
                state.last_metadata = metadata
                if state.active or state.hit_frames < self.config.min_consecutive_frames:
                    continue

                event = AlertEvent.create(
                    task="intrusion",
                    alert_type=decision.alert_type or "PERSON_INTRUSION",
                    phase="START",
                    message=_start_message(zone.name, decision),
                    confidence=detection.confidence,
                    frame_id=frame_id,
                    metadata=metadata,
                )
                state.active = True
                state.event_id = event.event_id
                events.append(event)

        # A missed detection also advances recovery, but track identity survives
        # brief detector gaps so a reappearing person does not create a new alarm.
        for key in list(self._states):
            if key not in evaluated_keys:
                self._record_miss(key, frame_id, events)
        self._expire_tracks(frame_id)
        return events

    def _identity_metadata(self, track_id: str, observed_at: datetime) -> dict[str, Any]:
        path = self.config.identity_context_path
        if path is None:
            return {}
        try:
            stat = path.stat()
            if self._identity_context_mtime_ns != stat.st_mtime_ns:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._identity_context = payload if isinstance(payload, dict) else {}
                self._identity_context_mtime_ns = stat.st_mtime_ns
        except (OSError, json.JSONDecodeError):
            return {}

        updated_at = self._identity_context.get("updated_at")
        if updated_at:
            try:
                context_time = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
                if context_time.tzinfo is None:
                    context_time = context_time.replace(tzinfo=ZoneInfo(self.config.timezone))
                age = abs((observed_at - context_time.astimezone(observed_at.tzinfo)).total_seconds())
                if age > self.config.identity_context_ttl_seconds:
                    return {}
            except ValueError:
                return {}
        tracks = self._identity_context.get("tracks") or {}
        metadata = tracks.get(str(track_id)) if isinstance(tracks, dict) else None
        if not isinstance(metadata, dict):
            return {}
        return {**metadata, "authorization_source": "identity_context"}

    def _record_miss(
        self,
        key: tuple[str, str],
        frame_id: int,
        events: list[AlertEvent],
    ) -> None:
        state = self._states.get(key)
        if state is None:
            return
        state.hit_frames = 0
        if not state.active:
            self._states.pop(key, None)
            return
        state.miss_frames += 1
        if state.miss_frames < self.config.recovery_consecutive_frames:
            return
        zone_name, track_id = key
        metadata = {
            **state.last_metadata,
            "zone": zone_name,
            "track_id": track_id,
            "recovery_frames": state.miss_frames,
        }
        events.append(
            AlertEvent.create(
                event_id=state.event_id,
                task="intrusion",
                alert_type=str(state.last_metadata.get("alert_type", "PERSON_INTRUSION")),
                phase="RECOVERED",
                message=f"电子围栏事件已恢复：{zone_name}",
                confidence=state.last_confidence,
                frame_id=frame_id,
                metadata=metadata,
            )
        )
        self._states.pop(key, None)

    def _assign_tracks(
        self, detections: list[Detection], frame_id: int
    ) -> list[tuple[Detection, str]]:
        assignments: list[tuple[Detection, str]] = []
        claimed: set[str] = set()
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            supplied_id = detection.metadata.get("track_id")
            if supplied_id is not None:
                track_id = str(supplied_id)
            else:
                track_id = self._match_track(detection.bbox, frame_id, claimed)
                if track_id is None:
                    track_id = f"auto-{self._next_track_id}"
                    self._next_track_id += 1
            self._tracks[track_id] = PersonTrack(track_id, detection.bbox, frame_id)
            claimed.add(track_id)
            assignments.append((detection, track_id))
        return assignments

    def _match_track(
        self,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        claimed: set[str],
    ) -> str | None:
        best: tuple[float, str] | None = None
        for track_id, track in self._tracks.items():
            if track_id in claimed:
                continue
            if frame_id - track.last_frame_id > self.config.tracker_max_missing_frames:
                continue
            overlap = _bbox_iou(bbox, track.bbox)
            distance = _normalized_center_distance(bbox, track.bbox)
            if overlap < self.config.tracker_iou_threshold and distance > 0.75:
                continue
            score = overlap - 0.05 * distance
            if best is None or score > best[0]:
                best = (score, track_id)
        return None if best is None else best[1]

    def _expire_tracks(self, frame_id: int) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_id - track.last_frame_id > self.config.tracker_max_missing_frames
        ]
        for track_id in expired:
            self._tracks.pop(track_id, None)
            for key in [item for item in self._zone_membership if item[1] == track_id]:
                self._zone_membership.pop(key, None)

    def _local_time(self, value: datetime | None) -> datetime:
        zone = ZoneInfo(self.config.timezone)
        if value is None:
            return datetime.now(timezone.utc).astimezone(zone)
        if value.tzinfo is None:
            return value.replace(tzinfo=zone)
        return value.astimezone(zone)


def evaluate_access(
    zone: ZoneConfig,
    metadata: dict[str, Any],
    observed_at: datetime,
) -> AccessDecision:
    identity_id = next(
        (
            str(metadata[key])
            for key in ("identity_id", "person_id", "credential_id")
            if metadata.get(key) not in (None, "")
        ),
        None,
    )
    if zone.access_policy == "deny_all":
        return AccessDecision(
            False,
            "PERSON_INTRUSION",
            "RESTRICTED_ZONE",
            "CONFIRMED_POLICY_VIOLATION",
            identity_id,
            "KNOWN" if identity_id else "UNKNOWN",
            None,
        )

    explicit = metadata.get("authorized")
    raw_authorized_zones = metadata.get("authorized_zones", [])
    if isinstance(raw_authorized_zones, str):
        raw_authorized_zones = [raw_authorized_zones]
    authorized_zones = {str(item) for item in raw_authorized_zones}
    if zone.allowed_identity_ids:
        authorized: bool | None = (
            None if identity_id is None else identity_id in set(zone.allowed_identity_ids)
        )
    elif explicit is not None:
        authorized = bool(explicit)
    elif authorized_zones:
        authorized = zone.name in authorized_zones
    else:
        authorized = None

    if authorized is None:
        if zone.unknown_identity_action == "allow":
            return AccessDecision(True, None, "UNKNOWN_IDENTITY_ALLOWED", "NOT_APPLICABLE", None, "UNKNOWN", None)
        alert_type = "PERSON_ACCESS_REVIEW" if zone.unknown_identity_action == "review" else "PERSON_INTRUSION"
        return AccessDecision(
            False,
            alert_type,
            "UNKNOWN_IDENTITY",
            "REVIEW_REQUIRED" if zone.unknown_identity_action == "review" else "SUSPECTED_IDENTITY_UNKNOWN",
            None,
            "UNKNOWN",
            None,
        )
    if not authorized:
        return AccessDecision(
            False,
            "PERSON_INTRUSION",
            "UNAUTHORIZED_IDENTITY",
            "CONFIRMED_POLICY_VIOLATION",
            identity_id,
            "UNAUTHORIZED",
            None,
        )

    within_window = _within_access_windows(zone.access_windows, observed_at)
    if zone.access_policy == "scheduled_authorized" and not within_window:
        return AccessDecision(
            False,
            "PERSON_INTRUSION",
            "OUTSIDE_ACCESS_WINDOW",
            "CONFIRMED_POLICY_VIOLATION",
            identity_id,
            "AUTHORIZED",
            False,
        )
    return AccessDecision(
        True,
        None,
        "ACCESS_PERMITTED",
        "CONFIRMED_AUTHORIZED",
        identity_id,
        "AUTHORIZED",
        within_window if zone.access_policy == "scheduled_authorized" else None,
    )


def _within_access_windows(windows: list[AccessWindowConfig], observed_at: datetime) -> bool:
    if not windows:
        return True
    current = observed_at.timetz().replace(tzinfo=None)
    weekday = observed_at.weekday()
    for window in windows:
        start = _parse_time(window.start)
        end = _parse_time(window.end)
        if start <= end:
            if weekday in window.days and start <= current <= end:
                return True
        else:
            previous_day = (weekday - 1) % 7
            if (weekday in window.days and current >= start) or (
                previous_day in window.days and current <= end
            ):
                return True
    return False


def _parse_time(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":"))
    return time(hour, minute)


def _start_message(zone_name: str, decision: AccessDecision) -> str:
    descriptions = {
        "RESTRICTED_ZONE": "人员进入全禁区域",
        "UNAUTHORIZED_IDENTITY": "无权限人员进入监测区域",
        "OUTSIDE_ACCESS_WINDOW": "人员在非授权时段进入监测区域",
        "UNKNOWN_IDENTITY": "未知身份人员进入监测区域，需复核",
    }
    return f"{descriptions.get(decision.reason_code, '电子围栏策略触发')}：{zone_name}"


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    scale = max(
        1.0,
        left[2] - left[0],
        left[3] - left[1],
        right[2] - right[0],
        right[3] - right[1],
    )
    return hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]) / scale
