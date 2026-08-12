from edge_inspection.breaker import BreakerStateDetector
from datetime import datetime
from zoneinfo import ZoneInfo

from edge_inspection.config import AccessWindowConfig, BreakerConfig, BreakerRoiConfig, IntrusionConfig, ModelsConfig, RuntimeConfig, SiteConfig, ZoneConfig
from edge_inspection.events import Detection
from edge_inspection.intrusion import IntrusionDetector
from edge_inspection.pipeline import (
    assign_breaker_assets,
    classify_breaker_rois,
    score_breaker_detection_crops,
)

import numpy as np

from edge_inspection.anomaly import normalize_torch_device


def test_numeric_runtime_device_is_normalized_for_torch() -> None:
    assert normalize_torch_device("0") == "cuda:0"
    assert normalize_torch_device("1") == "cuda:1"
    assert normalize_torch_device("cpu") == "cpu"
    assert normalize_torch_device("cuda:0") == "cuda:0"


def test_intrusion_requires_consecutive_frames() -> None:
    detector = IntrusionDetector(
        IntrusionConfig(
            min_consecutive_frames=2,
            zones=[ZoneConfig(name="z1", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])],
        )
    )
    detection = Detection(bbox=(10, 10, 30, 60), confidence=0.9, class_id=0, class_name="person")

    assert detector.update([detection], frame_id=1) == []
    events = detector.update([detection], frame_id=2)
    assert len(events) == 1
    assert events[0].alert_type == "PERSON_INTRUSION"
    assert events[0].phase == "START"
    assert detector.update([detection], frame_id=3) == []

    assert detector.update([], frame_id=4) == []
    assert detector.update([], frame_id=5) == []
    recovered = detector.update([], frame_id=6)
    assert recovered[0].phase == "RECOVERED"
    assert recovered[0].event_id == events[0].event_id


def test_intrusion_allows_authorized_identity() -> None:
    detector = IntrusionDetector(
        IntrusionConfig(
            min_consecutive_frames=1,
            zones=[
                ZoneConfig(
                    name="z1",
                    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                    access_policy="authorized_only",
                    allowed_identity_ids=["worker-7"],
                )
            ],
        )
    )
    person = Detection(
        (10, 10, 30, 60),
        0.9,
        0,
        "person",
        {"track_id": "p1", "identity_id": "worker-7"},
    )
    assert detector.update([person], frame_id=1) == []


def test_intrusion_marks_unknown_identity_for_review() -> None:
    detector = IntrusionDetector(
        IntrusionConfig(
            min_consecutive_frames=1,
            zones=[
                ZoneConfig(
                    name="z1",
                    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                    unknown_identity_action="review",
                )
            ],
        )
    )
    person = Detection((10, 10, 30, 60), 0.9, 0, "person", {"track_id": "p1"})
    event = detector.update([person], frame_id=1)[0]
    assert event.alert_type == "PERSON_ACCESS_REVIEW"
    assert event.metadata["reason_code"] == "UNKNOWN_IDENTITY"
    assert event.metadata["verification_status"] == "REVIEW_REQUIRED"


def test_intrusion_rejects_authorized_identity_outside_schedule() -> None:
    detector = IntrusionDetector(
        IntrusionConfig(
            min_consecutive_frames=1,
            zones=[
                ZoneConfig(
                    name="z1",
                    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                    access_policy="scheduled_authorized",
                    allowed_identity_ids=["worker-7"],
                    access_windows=[AccessWindowConfig(days=[0], start="08:00", end="18:00")],
                )
            ],
        )
    )
    person = Detection(
        (10, 10, 30, 60),
        0.9,
        0,
        "person",
        {"track_id": "p1", "identity_id": "worker-7"},
    )
    observed_at = datetime(2026, 8, 3, 20, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    event = detector.update([person], frame_id=1, observed_at=observed_at)[0]
    assert event.metadata["reason_code"] == "OUTSIDE_ACCESS_WINDOW"
    assert event.metadata["identity_status"] == "AUTHORIZED"


def test_intrusion_tracks_two_people_as_separate_fence_events() -> None:
    detector = IntrusionDetector(
        IntrusionConfig(
            min_consecutive_frames=1,
            recovery_consecutive_frames=1,
            zones=[
                ZoneConfig(
                    name="z1",
                    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                    access_policy="deny_all",
                )
            ],
        )
    )
    p1 = Detection((10, 10, 30, 60), 0.9, 0, "person", {"track_id": "p1"})
    p2 = Detection((60, 10, 80, 60), 0.8, 0, "person", {"track_id": "p2"})
    started = detector.update([p1, p2], frame_id=1)
    assert len(started) == 2
    assert {event.metadata["track_id"] for event in started} == {"p1", "p2"}
    recovered = detector.update([], frame_id=2)
    assert len(recovered) == 2
    assert {event.event_id for event in recovered} == {event.event_id for event in started}


def test_intrusion_reads_fresh_identity_context(tmp_path) -> None:
    context = tmp_path / "identity.json"
    context.write_text(
        '{"updated_at":"2026-08-03T10:00:00+08:00","tracks":{"p1":{"identity_id":"worker-7","authorized":true}}}',
        encoding="utf-8",
    )
    detector = IntrusionDetector(
        IntrusionConfig(
            min_consecutive_frames=1,
            identity_context_path=context,
            identity_context_ttl_seconds=10,
            zones=[
                ZoneConfig(
                    name="z1",
                    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                    access_policy="authorized_only",
                )
            ],
        )
    )
    person = Detection((10, 10, 30, 60), 0.9, 0, "person", {"track_id": "p1"})
    observed_at = datetime(2026, 8, 3, 10, 0, 5, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    assert detector.update([person], frame_id=1, observed_at=observed_at) == []


def test_breaker_alerts_only_abnormal_classes() -> None:
    detector = BreakerStateDetector(BreakerConfig(min_consecutive_frames=1))
    normal = Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="closed")
    abnormal = Detection(bbox=(0, 0, 10, 10), confidence=0.95, class_id=2, class_name="trip")

    assert detector.update([normal], frame_id=1) == []
    events = detector.update([abnormal], frame_id=2)
    assert len(events) == 1
    assert events[0].alert_type == "TRIP"
    assert detector.update([abnormal], frame_id=3) == []
    assert detector.update([], frame_id=4) == []
    recovered = detector.update([], frame_id=5)
    assert recovered[0].phase == "RECOVERED"
    assert recovered[0].event_id == events[0].event_id


def test_breaker_roi_classifier_detection_metadata() -> None:
    class FakeClassifier:
        def predict(self, frame):
            assert frame.shape[:2] == (20, 20)
            return Detection(
                bbox=(0, 0, 20, 20),
                confidence=0.70,
                class_id=1,
                class_name="OPEN",
                metadata={"class_probabilities": {"OPEN": 0.70, "TRIP": 0.25, "MICRO_TRIP": 0.05}},
            )

    config = SiteConfig(
        models=ModelsConfig(),
        runtime=RuntimeConfig(),
        intrusion=IntrusionConfig(zones=[]),
        breaker=BreakerConfig(
            mode="roi_classification",
            classifier_confidence=0.5,
            abnormal_classes=["TRIP"],
            class_thresholds={"TRIP": 0.2},
            min_consecutive_frames=1,
            rois=[BreakerRoiConfig(name="QF1", bbox=(10, 10, 30, 30))],
        ),
    )
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    detections = classify_breaker_rois(frame, FakeClassifier(), config)
    assert detections == [
        Detection(
            bbox=(10.0, 10.0, 30.0, 30.0),
            confidence=0.25,
            class_id=1,
            class_name="TRIP",
            metadata={
                "roi_name": "QF1",
                "class_probabilities": {"OPEN": 0.70, "TRIP": 0.25, "MICRO_TRIP": 0.05},
            },
        )
    ]

    events = BreakerStateDetector(config.breaker).update(detections, frame_id=1)
    assert events[0].metadata["roi_name"] == "QF1"


def test_temporal_open_short_pulse_becomes_micro_trip() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            trip_confirm_frames=4,
            micro_trip_max_frames=3,
            recovery_consecutive_frames=2,
        )
    )
    opened = Detection((0, 0, 10, 10), 0.9, 1, "OPEN", {"roi_name": "QF1"})
    closed = Detection((0, 0, 10, 10), 0.9, 0, "CLOSE", {"roi_name": "QF1"})

    assert detector.update([opened], frame_id=1) == []
    assert detector.update([opened], frame_id=2) == []
    assert detector.update([closed], frame_id=3) == []
    events = detector.update([closed], frame_id=4)
    assert [event.alert_type for event in events] == ["MICRO_TRIP", "MICRO_TRIP"]
    assert [event.phase for event in events] == ["START", "RECOVERED"]
    assert events[0].event_id == events[1].event_id
    assert events[0].metadata["open_duration_frames"] == 2


def test_temporal_open_stable_state_becomes_trip() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            trip_confirm_frames=3,
            micro_trip_max_frames=2,
            recovery_consecutive_frames=2,
        )
    )
    opened = Detection((0, 0, 10, 10), 0.9, 1, "OPEN", {"roi_name": "QF1"})
    closed = Detection((0, 0, 10, 10), 0.9, 0, "CLOSE", {"roi_name": "QF1"})

    assert detector.update([opened], frame_id=1) == []
    assert detector.update([opened], frame_id=2) == []
    started = detector.update([opened], frame_id=3)
    assert len(started) == 1 and started[0].alert_type == "TRIP"
    assert detector.update([opened], frame_id=4) == []
    assert detector.update([closed], frame_id=5) == []
    recovered = detector.update([closed], frame_id=6)
    assert len(recovered) == 1 and recovered[0].phase == "RECOVERED"
    assert recovered[0].event_id == started[0].event_id


def test_temporal_open_does_not_accumulate_across_long_detection_gap() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            trip_confirm_frames=3,
            micro_trip_max_frames=2,
            max_missing_frames=1,
        )
    )
    opened = Detection((0, 0, 10, 10), 0.9, 1, "OPEN", {"roi_name": "QF1"})
    assert detector.update([opened], frame_id=1) == []
    assert detector.update([], frame_id=2) == []
    assert detector.update([], frame_id=3) == []
    assert detector.update([opened], frame_id=4) == []
    assert detector.update([opened], frame_id=5) == []
    started = detector.update([opened], frame_id=6)
    assert len(started) == 1 and started[0].alert_type == "TRIP"


def test_temporal_evidence_uses_anomaly_score_for_suspected_micro_trip() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_evidence",
            trip_confirm_frames=4,
            micro_trip_min_frames=2,
            micro_trip_max_frames=3,
            recovery_consecutive_frames=2,
        )
    )
    deviation = Detection(
        (0, 0, 10, 10),
        0.8,
        0,
        "CLOSE",
        {"roi_name": "QF1", "anomaly_score": 0.9},
    )
    closed = Detection((0, 0, 10, 10), 0.95, 0, "CLOSE", {"roi_name": "QF1"})

    assert detector.update([deviation], frame_id=1) == []
    assert detector.update([deviation], frame_id=2) == []
    assert detector.update([closed], frame_id=3) == []
    events = detector.update([closed], frame_id=4)
    assert [event.phase for event in events] == ["START", "RECOVERED"]
    assert events[0].alert_type == "MICRO_TRIP"
    assert events[0].metadata["verification_status"] == "SUSPECTED_VISUAL_ONLY"
    assert "visual:anomaly_score" in events[0].metadata["evidence_sources"]


def test_temporal_evidence_ignores_single_frame_visual_noise() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_evidence",
            trip_confirm_frames=4,
            micro_trip_min_frames=2,
            micro_trip_max_frames=3,
            recovery_consecutive_frames=2,
        )
    )
    anomaly = Detection((0, 0, 10, 10), 0.8, 2, "ANOMALY", {"roi_name": "QF1"})
    closed = Detection((0, 0, 10, 10), 0.95, 0, "CLOSE", {"roi_name": "QF1"})

    assert detector.update([anomaly], frame_id=1) == []
    assert detector.update([closed], frame_id=2) == []
    assert detector.update([closed], frame_id=3) == []


def test_temporal_evidence_suppresses_commanded_open() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_evidence",
            trip_confirm_frames=3,
            micro_trip_min_frames=2,
            micro_trip_max_frames=2,
            recovery_consecutive_frames=2,
        )
    )
    opened = Detection(
        (0, 0, 10, 10),
        0.9,
        1,
        "OPEN",
        {"roi_name": "QF1", "commanded_open": True},
    )
    closed = Detection((0, 0, 10, 10), 0.9, 0, "CLOSE", {"roi_name": "QF1"})

    assert detector.update([opened], frame_id=1) == []
    assert detector.update([opened], frame_id=2) == []
    assert detector.update([opened], frame_id=3) == []
    assert detector.update([closed], frame_id=4) == []
    assert detector.update([closed], frame_id=5) == []


def test_temporal_evidence_marks_protection_confirmed_trip() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_evidence",
            trip_confirm_frames=3,
            micro_trip_min_frames=2,
            micro_trip_max_frames=2,
        )
    )
    opened = Detection(
        (0, 0, 10, 10),
        0.9,
        1,
        "OPEN",
        {"roi_name": "QF1", "protection_trip": True},
    )

    assert detector.update([opened], frame_id=1) == []
    assert detector.update([opened], frame_id=2) == []
    events = detector.update([opened], frame_id=3)
    assert len(events) == 1
    assert events[0].alert_type == "TRIP"
    assert events[0].metadata["verification_status"] == "CONFIRMED"
    assert "control:trip_confirmation" in events[0].metadata["evidence_sources"]


def test_temporal_seconds_requires_stable_closed_arming() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            temporal_seconds_enabled=True,
            observation_confidence=0.9,
            arm_closed_seconds=3.0,
            micro_trip_min_seconds=0.5,
            micro_trip_max_seconds=1.0,
            trip_confirm_seconds=2.0,
            recovery_seconds=0.5,
            max_observation_gap_seconds=0.6,
            rearm_after_event=True,
        )
    )
    opened = Detection(
        (0, 0, 10, 10),
        0.99,
        1,
        "OPEN",
        {"roi_name": "QF1", "class_probabilities": {"OPEN": 0.99, "CLOSED": 0.01}},
    )
    closed = Detection(
        (0, 0, 10, 10),
        0.99,
        0,
        "CLOSED",
        {"roi_name": "QF1", "class_probabilities": {"OPEN": 0.01, "CLOSED": 0.99}},
    )

    assert detector.update([opened], frame_id=1, observed_at_seconds=0.0) == []
    assert detector.update([opened], frame_id=2, observed_at_seconds=0.5) == []
    assert detector.update([opened], frame_id=3, observed_at_seconds=1.0) == []
    for frame_id, seconds in enumerate((2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0), start=4):
        assert detector.update([closed], frame_id=frame_id, observed_at_seconds=seconds) == []

    assert detector.update([opened], frame_id=11, observed_at_seconds=5.2) == []
    assert detector.update([opened], frame_id=12, observed_at_seconds=5.8) == []
    assert detector.update([closed], frame_id=13, observed_at_seconds=6.0) == []
    events = detector.update([closed], frame_id=14, observed_at_seconds=6.6)
    assert [event.alert_type for event in events] == ["MICRO_TRIP", "MICRO_TRIP"]
    assert 0.5 <= events[0].metadata["open_duration_seconds"] <= 1.0


def test_temporal_seconds_rejects_low_confidence_state_flip() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            temporal_seconds_enabled=True,
            observation_confidence=0.98,
            arm_closed_seconds=1.0,
            micro_trip_min_seconds=0.4,
            micro_trip_max_seconds=1.0,
            trip_confirm_seconds=2.0,
            recovery_seconds=0.4,
            max_observation_gap_seconds=0.6,
            rearm_after_event=True,
        )
    )
    closed = Detection(
        (0, 0, 10, 10),
        0.99,
        0,
        "CLOSED",
        {"roi_name": "QF1", "class_probabilities": {"OPEN": 0.01, "CLOSED": 0.99}},
    )
    uncertain_open = Detection(
        (0, 0, 10, 10),
        0.60,
        1,
        "OPEN",
        {"roi_name": "QF1", "class_probabilities": {"OPEN": 0.60, "CLOSED": 0.40}},
    )

    for frame_id, seconds in enumerate((0.0, 0.5, 1.0), start=1):
        assert detector.update([closed], frame_id=frame_id, observed_at_seconds=seconds) == []
    assert detector.update([uncertain_open], frame_id=4, observed_at_seconds=1.2) == []
    assert detector.update([uncertain_open], frame_id=5, observed_at_seconds=1.8) == []
    assert detector.update([closed], frame_id=6, observed_at_seconds=2.0) == []
    assert detector.update([closed], frame_id=7, observed_at_seconds=2.5) == []


def test_temporal_seconds_emits_trip_and_recovery() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            temporal_seconds_enabled=True,
            observation_confidence=0.9,
            arm_closed_seconds=1.0,
            micro_trip_min_seconds=0.4,
            micro_trip_max_seconds=1.0,
            trip_confirm_seconds=2.0,
            recovery_seconds=0.5,
            max_observation_gap_seconds=0.6,
            rearm_after_event=True,
        )
    )
    closed = Detection((0, 0, 10, 10), 0.99, 0, "CLOSED", {"roi_name": "QF1"})
    opened = Detection((0, 0, 10, 10), 0.99, 1, "OPEN", {"roi_name": "QF1"})

    for frame_id, seconds in enumerate((0.0, 0.5, 1.0), start=1):
        assert detector.update([closed], frame_id=frame_id, observed_at_seconds=seconds) == []
    assert detector.update([opened], frame_id=4, observed_at_seconds=1.2) == []
    assert detector.update([opened], frame_id=5, observed_at_seconds=1.7) == []
    assert detector.update([opened], frame_id=6, observed_at_seconds=2.2) == []
    assert detector.update([opened], frame_id=7, observed_at_seconds=2.7) == []
    started = detector.update([opened], frame_id=8, observed_at_seconds=3.2)
    assert len(started) == 1 and started[0].alert_type == "TRIP"
    assert started[0].metadata["open_duration_seconds"] == 2.0

    assert detector.update([closed], frame_id=9, observed_at_seconds=3.4) == []
    recovered = detector.update([closed], frame_id=10, observed_at_seconds=3.9)
    assert len(recovered) == 1 and recovered[0].phase == "RECOVERED"
    assert recovered[0].event_id == started[0].event_id


def test_temporal_seconds_resets_after_timestamp_gap() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            temporal_seconds_enabled=True,
            observation_confidence=0.9,
            arm_closed_seconds=1.0,
            micro_trip_min_seconds=0.4,
            micro_trip_max_seconds=1.0,
            trip_confirm_seconds=2.0,
            recovery_seconds=0.4,
            max_observation_gap_seconds=0.6,
            rearm_after_event=True,
        )
    )
    closed = Detection((0, 0, 10, 10), 0.99, 0, "CLOSED", {"roi_name": "QF1"})
    opened = Detection((0, 0, 10, 10), 0.99, 1, "OPEN", {"roi_name": "QF1"})

    for frame_id, seconds in enumerate((0.0, 0.5, 1.0), start=1):
        assert detector.update([closed], frame_id=frame_id, observed_at_seconds=seconds) == []
    assert detector.update([opened], frame_id=4, observed_at_seconds=1.2) == []
    assert detector.update([opened], frame_id=5, observed_at_seconds=2.2) == []
    assert detector.update([closed], frame_id=6, observed_at_seconds=2.4) == []
    assert detector.update([closed], frame_id=7, observed_at_seconds=2.9) == []


def test_assign_breaker_assets_uses_configured_roi() -> None:
    config = SiteConfig(
        models=ModelsConfig(),
        runtime=RuntimeConfig(),
        intrusion=IntrusionConfig(zones=[]),
        breaker=BreakerConfig(rois=[BreakerRoiConfig(name="QF1", bbox=(0, 0, 50, 50))]),
    )
    inside = Detection((10, 10, 20, 20), 0.9, 1, "OPEN")
    outside = Detection((60, 60, 70, 70), 0.9, 1, "OPEN")
    assigned = assign_breaker_assets([inside, outside], config)
    assert len(assigned) == 1
    assert assigned[0].metadata == {"roi_name": "QF1", "asset_id": "QF1"}


def test_detection_hybrid_scores_detected_crop_and_gates_calibration() -> None:
    class FakeAnomalyScorer:
        def score(self, crop, *, asset_id, allow_calibration):
            assert crop.shape[:2] == (20, 20)
            assert asset_id == "QF1"
            assert allow_calibration is True
            return {"anomaly_score": 0.2, "anomaly_calibration_ready": True}

    config = SiteConfig(
        models=ModelsConfig(),
        runtime=RuntimeConfig(),
        intrusion=IntrusionConfig(zones=[]),
        breaker=BreakerConfig(
            mode="detection_hybrid",
            closed_classes=["CLOSE", "CLOSED"],
            anomaly_normal_confidence=0.9,
        ),
    )
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    detection = Detection(
        (10, 10, 30, 30),
        0.95,
        0,
        "CLOSE",
        {"asset_id": "QF1"},
    )

    scored = score_breaker_detection_crops(
        frame,
        [detection],
        FakeAnomalyScorer(),
        config,
    )
    assert scored[0].metadata["anomaly_score"] == 0.2
    assert scored[0].metadata["anomaly_calibration_ready"] is True
