from __future__ import annotations

from dataclasses import dataclass

from .config import BreakerConfig
from .events import AlertEvent, Detection


@dataclass
class BreakerEventState:
    class_name: str
    hit_frames: int = 0
    miss_frames: int = 0
    active: bool = False
    event_id: str | None = None
    last_confidence: float = 0.0


@dataclass
class TemporalBreakerState:
    open_frames: int = 0
    closed_frames: int = 0
    trip_active: bool = False
    trip_event_id: str | None = None
    last_open_confidence: float = 0.0
    first_open_frame: int | None = None
    missing_frames: int = 0
    suppressed_by_command: bool = False
    confirmed: bool = False
    confirmation_emitted: bool = False
    evidence_sources: set[str] | None = None
    armed: bool = False
    closed_since_seconds: float | None = None
    open_since_seconds: float | None = None
    last_open_seconds: float | None = None
    recovery_since_seconds: float | None = None
    last_observation_seconds: float | None = None


class BreakerStateDetector:
    """Turns frame-level breaker states into de-duplicated alarm events."""

    def __init__(self, config: BreakerConfig):
        self.config = config
        self._states: dict[str, BreakerEventState] = {}
        self._temporal_states: dict[str, TemporalBreakerState] = {}

    def update(
        self,
        detections: list[Detection],
        *,
        frame_id: int,
        observed_at_seconds: float | None = None,
    ) -> list[AlertEvent]:
        if self.config.decision_mode in {"temporal_open", "temporal_evidence"}:
            return self._update_temporal_evidence(
                detections,
                frame_id=frame_id,
                observed_at_seconds=observed_at_seconds,
                open_only=self.config.decision_mode == "temporal_open",
            )
        if self.config.decision_mode != "direct_classes":
            raise ValueError(f"Unsupported breaker.decision_mode: {self.config.decision_mode}")

        abnormal_names = {name.upper() for name in self.config.abnormal_classes}
        best_by_asset: dict[str, Detection] = {}
        for detection in detections:
            class_name = detection.class_name.upper()
            if class_name not in abnormal_names:
                continue
            normalized = Detection(
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=class_name,
                metadata=detection.metadata,
            )
            key = self._asset_key(normalized)
            current = best_by_asset.get(key)
            if current is None or normalized.confidence > current.confidence:
                best_by_asset[key] = normalized

        events: list[AlertEvent] = []
        for key in sorted(set(self._states) | set(best_by_asset)):
            detection = best_by_asset.get(key)
            state = self._states.get(key)

            if detection is None:
                if state is None or not state.active:
                    self._states.pop(key, None)
                    continue
                state.hit_frames = 0
                state.miss_frames += 1
                if state.miss_frames >= self.config.recovery_consecutive_frames:
                    events.append(self._recovered_event(key, state, frame_id))
                    self._states.pop(key, None)
                continue

            if state is not None and state.class_name != detection.class_name:
                if state.active:
                    events.append(self._recovered_event(key, state, frame_id))
                state = None

            if state is None:
                state = BreakerEventState(class_name=detection.class_name)
                self._states[key] = state
            state.hit_frames += 1
            state.miss_frames = 0
            state.last_confidence = detection.confidence
            if state.active or state.hit_frames < self.config.min_consecutive_frames:
                continue

            metadata = {
                **detection.metadata,
                "bbox": detection.bbox,
                "class_id": detection.class_id,
                "confirmation_frames": state.hit_frames,
            }
            event = AlertEvent.create(
                task="breaker",
                alert_type=detection.class_name,
                phase="START",
                message=f"断路器异常状态：{detection.class_name}",
                confidence=detection.confidence,
                frame_id=frame_id,
                metadata=metadata,
            )
            state.active = True
            state.event_id = event.event_id
            events.append(event)

        return events

    def _update_temporal_evidence(
        self,
        detections: list[Detection],
        *,
        frame_id: int,
        observed_at_seconds: float | None,
        open_only: bool,
    ) -> list[AlertEvent]:
        open_names = {name.upper() for name in self.config.open_classes}
        closed_names = {name.upper() for name in self.config.closed_classes}
        deviation_names = open_names if open_only else {
            name.upper() for name in self.config.deviation_classes
        }
        best_by_asset: dict[str, Detection] = {}
        for detection in detections:
            class_name = detection.class_name.upper()
            anomaly_score = self._anomaly_score(detection)
            score_is_anomalous = (
                anomaly_score is not None
                and anomaly_score >= self.config.anomaly_score_threshold
            )
            if class_name not in deviation_names | closed_names and not score_is_anomalous:
                continue
            key = self._asset_key(detection)
            current = best_by_asset.get(key)
            if current is None or detection.confidence > current.confidence:
                best_by_asset[key] = detection

        events: list[AlertEvent] = []
        for key in sorted(set(best_by_asset) | set(self._temporal_states)):
            detection = best_by_asset.get(key)
            if key not in self._temporal_states:
                self._temporal_states[key] = self._new_temporal_state()
            state = self._temporal_states[key]
            if self._time_gap_exceeded(state, observed_at_seconds):
                state = self._new_temporal_state()
                self._temporal_states[key] = state
            if observed_at_seconds is not None:
                state.last_observation_seconds = observed_at_seconds
            if detection is None:
                state.missing_frames += 1
                state.closed_frames = 0
                if not state.trip_active and state.missing_frames > self.config.max_missing_frames:
                    self._temporal_states[key] = self._new_temporal_state()
                continue

            state.missing_frames = 0
            class_name = detection.class_name.upper()
            anomaly_score = self._anomaly_score(detection)
            score_is_anomalous = (
                anomaly_score is not None
                and anomaly_score >= self.config.anomaly_score_threshold
            )
            is_deviation = class_name in deviation_names or score_is_anomalous
            if not self._observation_is_reliable(
                detection,
                class_name=class_name,
                is_deviation=is_deviation,
                score_is_anomalous=score_is_anomalous,
            ):
                self._reset_unconfirmed_transition(state)
                continue
            commanded = self._metadata_flag(detection, self.config.command_metadata_keys)
            confirmed = self._metadata_flag(detection, self.config.trip_confirmation_keys)

            if is_deviation:
                if state.open_frames == 0:
                    state.first_open_frame = frame_id
                    state.evidence_sources = set()
                state.open_frames += 1
                state.closed_frames = 0
                state.closed_since_seconds = None
                state.recovery_since_seconds = None
                if observed_at_seconds is not None:
                    if state.open_since_seconds is None:
                        state.open_since_seconds = observed_at_seconds
                    state.last_open_seconds = observed_at_seconds
                state.last_open_confidence = detection.confidence
                state.suppressed_by_command = state.suppressed_by_command or commanded
                state.confirmed = state.confirmed or confirmed
                if state.evidence_sources is None:
                    state.evidence_sources = set()
                state.evidence_sources.add(f"visual:{class_name}")
                if score_is_anomalous:
                    state.evidence_sources.add("visual:anomaly_score")
                if confirmed:
                    state.evidence_sources.add("control:trip_confirmation")

                if state.trip_active and state.confirmed and not state.confirmation_emitted:
                    state.confirmation_emitted = True
                    events.append(
                        AlertEvent.create(
                            event_id=state.trip_event_id,
                            task="breaker",
                            alert_type="TRIP",
                            phase="CONFIRMED",
                            message="断路器跳闸已由辅助证据确认",
                            confidence=detection.confidence,
                            frame_id=frame_id,
                            metadata=self._temporal_metadata(
                                key,
                                state,
                                detection,
                                observed_class=class_name,
                                anomaly_score=anomaly_score,
                            ),
                        )
                    )
                    continue

                if not state.armed:
                    self._reset_open_transition(state)
                    continue

                if not state.trip_active and self._trip_confirmed(state, observed_at_seconds):
                    if state.suppressed_by_command:
                        continue
                    event = AlertEvent.create(
                        task="breaker",
                        alert_type="TRIP",
                        phase="START",
                        message=(
                            "断路器跳闸（多源证据确认）"
                            if state.confirmed
                            else "断路器疑似跳闸（视觉状态持续偏离）"
                        ),
                        confidence=detection.confidence,
                        frame_id=frame_id,
                        metadata={
                            **self._temporal_metadata(
                                key,
                                state,
                                detection,
                                observed_class=class_name,
                                anomaly_score=anomaly_score,
                            ),
                            "confirmation_frames": state.open_frames,
                            "open_duration_seconds": self._open_duration_seconds(
                                state, observed_at_seconds
                            ),
                            "verification_status": self._verification_status(state),
                        },
                    )
                    state.trip_active = True
                    state.trip_event_id = event.event_id
                    state.confirmation_emitted = state.confirmed
                    events.append(event)
                continue

            state.closed_frames += 1
            if observed_at_seconds is not None and state.closed_since_seconds is None:
                state.closed_since_seconds = observed_at_seconds
            if (
                observed_at_seconds is not None
                and state.open_frames > 0
                and state.recovery_since_seconds is None
            ):
                state.recovery_since_seconds = observed_at_seconds
            if not state.armed and self._closed_arm_ready(state, observed_at_seconds):
                state.armed = True
            if state.trip_active:
                if observed_at_seconds is not None and state.recovery_since_seconds is None:
                    state.recovery_since_seconds = observed_at_seconds
                if not self._recovery_confirmed(state, observed_at_seconds):
                    continue
                events.append(
                    AlertEvent.create(
                        event_id=state.trip_event_id,
                        task="breaker",
                        alert_type="TRIP",
                        phase="RECOVERED",
                        message="断路器跳闸状态已恢复",
                        confidence=state.last_open_confidence,
                        frame_id=frame_id,
                        metadata={
                            **detection.metadata,
                            "asset_key": key,
                            "recovery_frames": state.closed_frames,
                            "recovery_seconds": self._recovery_duration_seconds(
                                state, observed_at_seconds
                            ),
                            "verification_status": self._verification_status(state),
                        },
                    )
                )
                self._temporal_states[key] = self._post_event_state(observed_at_seconds)
                continue

            if state.open_frames == 0 or not self._recovery_confirmed(state, observed_at_seconds):
                continue
            if self._is_micro_trip(state, observed_at_seconds) and not state.suppressed_by_command:
                verification_status = self._verification_status(state)
                event = AlertEvent.create(
                    task="breaker",
                    alert_type="MICRO_TRIP",
                    phase="START",
                    message=(
                        "断路器微跳（多源证据确认）"
                        if state.confirmed
                        else "断路器疑似微跳（短时视觉偏离后恢复）"
                    ),
                    confidence=state.last_open_confidence,
                    frame_id=frame_id,
                    metadata={
                        **self._temporal_metadata(
                            key,
                            state,
                            detection,
                            observed_class=class_name,
                            anomaly_score=anomaly_score,
                        ),
                        "open_duration_frames": state.open_frames,
                        "open_duration_seconds": self._open_duration_seconds(
                            state, state.last_open_seconds
                        ),
                        "recovery_frames": state.closed_frames,
                        "verification_status": verification_status,
                    },
                )
                events.extend(
                    [
                        event,
                        AlertEvent.create(
                            event_id=event.event_id,
                            task="breaker",
                            alert_type="MICRO_TRIP",
                            phase="RECOVERED",
                            message="断路器微跳状态已恢复",
                            confidence=state.last_open_confidence,
                            frame_id=frame_id,
                            metadata={
                                **detection.metadata,
                                "asset_key": key,
                                "open_duration_frames": state.open_frames,
                                "open_duration_seconds": self._open_duration_seconds(
                                    state, state.last_open_seconds
                                ),
                                "verification_status": verification_status,
                            },
                        ),
                    ]
                )
            self._temporal_states[key] = self._post_event_state(observed_at_seconds)

        return events

    def _observation_is_reliable(
        self,
        detection: Detection,
        *,
        class_name: str,
        is_deviation: bool,
        score_is_anomalous: bool,
    ) -> bool:
        if detection.metadata.get("observation_valid") is False:
            return False
        if detection.metadata.get("decision_basis") == "site_reference_geometry":
            return True
        threshold = self.config.observation_confidence
        if threshold <= 0:
            return True
        probabilities = detection.metadata.get("class_probabilities", {})
        class_probability = probabilities.get(class_name)
        if isinstance(class_probability, (int, float)):
            visual_confidence = float(class_probability)
        else:
            visual_confidence = float(detection.confidence)
        if score_is_anomalous:
            return bool(detection.metadata.get("anomaly_calibration_ready", False))
        if not is_deviation and class_name not in {
            name.upper() for name in self.config.closed_classes
        }:
            return False
        return visual_confidence >= threshold

    def _time_gap_exceeded(
        self, state: TemporalBreakerState, observed_at_seconds: float | None
    ) -> bool:
        return bool(
            self.config.temporal_seconds_enabled
            and observed_at_seconds is not None
            and state.last_observation_seconds is not None
            and observed_at_seconds - state.last_observation_seconds
            > self.config.max_observation_gap_seconds
        )

    def _closed_arm_ready(
        self, state: TemporalBreakerState, observed_at_seconds: float | None
    ) -> bool:
        if not self.config.temporal_seconds_enabled or observed_at_seconds is None:
            return state.closed_frames >= 1
        return bool(
            state.closed_since_seconds is not None
            and observed_at_seconds - state.closed_since_seconds
            >= self.config.arm_closed_seconds
        )

    def _trip_confirmed(
        self, state: TemporalBreakerState, observed_at_seconds: float | None
    ) -> bool:
        if not self.config.temporal_seconds_enabled or observed_at_seconds is None:
            return state.open_frames >= self.config.trip_confirm_frames
        return bool(
            state.open_since_seconds is not None
            and observed_at_seconds - state.open_since_seconds
            >= self.config.trip_confirm_seconds
        )

    def _recovery_confirmed(
        self, state: TemporalBreakerState, observed_at_seconds: float | None
    ) -> bool:
        if not self.config.temporal_seconds_enabled or observed_at_seconds is None:
            return state.closed_frames >= self.config.recovery_consecutive_frames
        return bool(
            state.recovery_since_seconds is not None
            and observed_at_seconds - state.recovery_since_seconds
            >= self.config.recovery_seconds
        )

    def _is_micro_trip(
        self, state: TemporalBreakerState, observed_at_seconds: float | None
    ) -> bool:
        if not self.config.temporal_seconds_enabled or observed_at_seconds is None:
            return (
                self.config.micro_trip_min_frames
                <= state.open_frames
                <= self.config.micro_trip_max_frames
            )
        duration = self._open_duration_seconds(state, state.last_open_seconds)
        return bool(
            duration is not None
            and self.config.micro_trip_min_seconds
            <= duration
            <= self.config.micro_trip_max_seconds
        )

    @staticmethod
    def _open_duration_seconds(
        state: TemporalBreakerState, end_seconds: float | None
    ) -> float | None:
        if state.open_since_seconds is None or end_seconds is None:
            return None
        return max(0.0, end_seconds - state.open_since_seconds)

    @staticmethod
    def _recovery_duration_seconds(
        state: TemporalBreakerState, end_seconds: float | None
    ) -> float | None:
        if state.recovery_since_seconds is None or end_seconds is None:
            return None
        return max(0.0, end_seconds - state.recovery_since_seconds)

    def _post_event_state(self, observed_at_seconds: float | None) -> TemporalBreakerState:
        state = self._new_temporal_state()
        state.last_observation_seconds = observed_at_seconds
        if not self.config.rearm_after_event:
            state.armed = True
            state.closed_frames = 1
            state.closed_since_seconds = observed_at_seconds
        return state

    def _new_temporal_state(self) -> TemporalBreakerState:
        return TemporalBreakerState(armed=not self.config.temporal_seconds_enabled)

    @staticmethod
    def _reset_open_transition(state: TemporalBreakerState) -> None:
        state.open_frames = 0
        state.open_since_seconds = None
        state.last_open_seconds = None
        state.first_open_frame = None
        state.suppressed_by_command = False
        state.confirmed = False
        state.evidence_sources = None

    def _reset_unconfirmed_transition(self, state: TemporalBreakerState) -> None:
        state.closed_frames = 0
        state.closed_since_seconds = None
        state.recovery_since_seconds = None
        if not state.trip_active:
            self._reset_open_transition(state)

    @staticmethod
    def _anomaly_score(detection: Detection) -> float | None:
        value = detection.metadata.get("anomaly_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _metadata_flag(detection: Detection, keys: list[str]) -> bool:
        return any(bool(detection.metadata.get(key)) for key in keys)

    @staticmethod
    def _verification_status(state: TemporalBreakerState) -> str:
        return "CONFIRMED" if state.confirmed else "SUSPECTED_VISUAL_ONLY"

    @staticmethod
    def _temporal_metadata(
        key: str,
        state: TemporalBreakerState,
        detection: Detection,
        *,
        observed_class: str,
        anomaly_score: float | None,
    ) -> dict:
        metadata = {
            **detection.metadata,
            "asset_key": key,
            "first_open_frame": state.first_open_frame,
            "observed_class": observed_class,
            "decision_basis": "temporal_visual_deviation",
            "command_suppressed": state.suppressed_by_command,
            "evidence_sources": sorted(state.evidence_sources or set()),
        }
        if anomaly_score is not None:
            metadata["anomaly_score"] = anomaly_score
        return metadata

    @staticmethod
    def _asset_key(detection: Detection) -> str:
        return str(detection.metadata.get("asset_id") or detection.metadata.get("roi_name") or detection.class_name)

    @staticmethod
    def _recovered_event(key: str, state: BreakerEventState, frame_id: int) -> AlertEvent:
        return AlertEvent.create(
            event_id=state.event_id,
            task="breaker",
            alert_type=state.class_name,
            phase="RECOVERED",
            message=f"断路器异常状态已恢复：{state.class_name}",
            confidence=state.last_confidence,
            frame_id=frame_id,
            metadata={"asset_key": key, "recovery_frames": state.miss_frames},
        )
