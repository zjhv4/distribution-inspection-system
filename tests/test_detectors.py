from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from edge_inspection.breaker import BreakerStateDetector
from edge_inspection.breaker_mobile import (
    MobileBreakerTracker,
    MobileMcbStateGate,
    classify_mcb_crops,
    classify_mcb_handle_geometry,
    filter_visible_breaker_controls,
    merge_breaker_detections,
    select_coherent_breaker_row,
)
from edge_inspection.config import AccessWindowConfig, BreakerConfig, IntrusionConfig, ZoneConfig
from edge_inspection.events import Detection
from edge_inspection.intrusion import IntrusionDetector


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


def test_mobile_breaker_tracker_ignores_non_mcb_and_keeps_identity() -> None:
    tracker = MobileBreakerTracker(iou_threshold=0.2, max_missing_frames=2)
    mcb = Detection((10, 10, 30, 60), 0.9, 0, "MCB")
    isolator = Detection((40, 10, 80, 60), 0.9, 2, "ISOLATOR")
    first = tracker.update([mcb, isolator], frame_id=1)
    second = tracker.update([Detection((11, 10, 31, 60), 0.9, 0, "MCB")], frame_id=2)
    assert len(first) == 1
    assert first[0].metadata["track_id"] == second[0].metadata["track_id"]


def test_mobile_tiled_detections_are_deduplicated() -> None:
    detections = [
        Detection((10, 20, 50, 100), 0.80, 0, "MCB"),
        Detection((12, 19, 51, 101), 0.55, 0, "MCB"),
        Detection((70, 20, 110, 100), 0.70, 0, "MCB"),
    ]
    merged = merge_breaker_detections(detections)
    assert len(merged) == 2
    assert merged[0].confidence == 0.80


def test_mobile_tiled_detections_remove_cross_class_duplicates() -> None:
    detections = [
        Detection((10, 20, 60, 100), 0.75, 0, "MCB"),
        Detection((11, 20, 61, 100), 0.60, 1, "RCD"),
    ]
    assert merge_breaker_detections(detections) == [detections[0]]


def test_mobile_tiled_detection_requires_coherent_device_row() -> None:
    row = [
        Detection((10, 100, 50, 200), 0.4, 0, "MCB"),
        Detection((60, 103, 100, 202), 0.5, 0, "MCB"),
        Detection((110, 98, 150, 199), 0.6, 1, "RCD"),
        Detection((200, 10, 240, 80), 0.7, 0, "MCB"),
    ]
    selected = select_coherent_breaker_row(row, minimum_devices=3)
    assert [item.bbox[0] for item in selected] == [10, 60, 110]
    assert select_coherent_breaker_row(row[:2], minimum_devices=3) == []


def test_mobile_detection_waits_until_operating_area_is_visible() -> None:
    import cv2

    hidden = np.full((140, 80, 3), 220, dtype=np.uint8)
    visible = hidden.copy()
    cv2.rectangle(visible, (25, 70), (55, 125), (25, 25, 25), -1)
    detection = Detection((10, 10, 70, 135), 0.8, 0, "MCB")
    assert filter_visible_breaker_controls(hidden, [detection]) == []
    result = filter_visible_breaker_controls(visible, [detection])
    assert len(result) == 1
    assert result[0].metadata["control_visible"] is True


def test_mobile_handle_geometry_marks_down_handle_open() -> None:
    import cv2

    frame = np.full((140, 80, 3), 230, dtype=np.uint8)
    cv2.rectangle(frame, (28, 72), (52, 128), (20, 20, 20), -1)
    cv2.rectangle(frame, (18, 110), (62, 130), (35, 35, 35), -1)
    detection = Detection((10, 10, 70, 135), 0.95, 0, "MCB", {"track_id": "MCB-1"})
    result = classify_mcb_handle_geometry(frame, [detection], handle_up_means_closed=True)
    assert result[0].class_name == "OPEN"
    assert result[0].metadata["device_class"] == "MCB"
    assert result[0].metadata["track_id"] == "MCB-1"


def test_mobile_handle_geometry_marks_up_handle_closed() -> None:
    import cv2

    frame = np.full((140, 80, 3), 230, dtype=np.uint8)
    cv2.rectangle(frame, (18, 48), (62, 68), (25, 25, 25), -1)
    cv2.rectangle(frame, (28, 48), (52, 92), (20, 20, 20), -1)
    detection = Detection((10, 10, 70, 135), 0.95, 0, "MCB", {"track_id": "MCB-1"})
    result = classify_mcb_handle_geometry(frame, [detection], handle_up_means_closed=True)
    assert result[0].class_name == "CLOSED"


def test_mobile_handle_geometry_returns_unknown_when_hand_occludes_device() -> None:
    frame = np.full((140, 80, 3), (105, 145, 195), dtype=np.uint8)
    detection = Detection((10, 10, 70, 135), 0.95, 0, "MCB", {"track_id": "MCB-1"})
    result = classify_mcb_handle_geometry(frame, [detection])
    assert result[0].class_name == "UNKNOWN"
    assert result[0].metadata["observation_valid"] is False


def test_mobile_tracker_replaces_identity_after_long_absence() -> None:
    tracker = MobileBreakerTracker(iou_threshold=0.2, max_missing_frames=2)
    first = tracker.update([Detection((10, 10, 30, 60), 0.9, 0, "MCB")], frame_id=1)
    tracker.update([], frame_id=2)
    tracker.update([], frame_id=4)
    returned = tracker.update([Detection((10, 10, 30, 60), 0.9, 0, "MCB")], frame_id=5)
    assert first[0].metadata["track_id"] != returned[0].metadata["track_id"]


def test_mobile_tracker_uses_model_track_id_when_available() -> None:
    tracker = MobileBreakerTracker(iou_threshold=0.2, max_missing_frames=2)
    detection = Detection((10, 10, 30, 60), 0.9, 0, "MCB", {"track_id": 17})
    result = tracker.update([detection], frame_id=1)
    assert result[0].metadata["track_id"] == "MCB-17"
    assert result[0].metadata["asset_id"] == "MCB-17"


def test_mobile_tracker_does_not_replace_model_identity_with_iou_match() -> None:
    tracker = MobileBreakerTracker(iou_threshold=0.2, max_missing_frames=2)
    tracker.update([Detection((10, 10, 30, 60), 0.9, 0, "MCB", {"track_id": 17})], frame_id=1)
    moved = tracker.update(
        [Detection((11, 10, 31, 60), 0.9, 0, "MCB", {"track_id": 23})],
        frame_id=2,
    )
    assert moved[0].metadata["track_id"] == "MCB-23"


def test_mobile_state_classifier_rejects_uncertain_prediction() -> None:
    class FakeClassifier:
        def predict(self, crop):
            return Detection(
                (0, 0, 10, 10),
                0.55,
                0,
                "OPEN",
                {"class_probabilities": {"OPEN": 0.55, "CLOSED": 0.45}},
            )

    frame = np.full((100, 60, 3), 220, dtype=np.uint8)
    device = Detection((0, 0, 60, 100), 0.9, 0, "MCB", {"track_id": "MCB-1"})
    result = classify_mcb_crops(
        frame,
        [device],
        FakeClassifier(),
        confidence_threshold=0.75,
        unknown_margin=0.15,
    )
    assert result[0].class_name == "UNKNOWN"
    assert result[0].metadata["observation_valid"] is False


def test_mobile_state_classifier_rejects_closed_without_geometry_confirmation() -> None:
    import cv2

    class FakeClassifier:
        def predict(self, crop):
            return Detection(
                (0, 0, 10, 10),
                0.99,
                0,
                "CLOSED",
                {"class_probabilities": {"OPEN": 0.01, "CLOSED": 0.99}},
            )

    frame = np.full((140, 80, 3), 230, dtype=np.uint8)
    cv2.rectangle(frame, (28, 72), (52, 128), (20, 20, 20), -1)
    cv2.rectangle(frame, (18, 110), (62, 130), (35, 35, 35), -1)
    device = Detection((10, 10, 70, 135), 0.9, 0, "MCB", {"track_id": "MCB-1"})
    result = classify_mcb_crops(
        frame,
        [device],
        FakeClassifier(),
        confidence_threshold=0.75,
        unknown_margin=0.15,
    )
    assert result[0].class_name == "UNKNOWN"
    assert result[0].metadata["closed_geometry_confirmation"] == "OPEN"


def test_mobile_state_gate_rejects_short_closed_flicker() -> None:
    gate = MobileMcbStateGate(closed_confirm_seconds=0.5, max_missing_frames=2)
    closed = Detection(
        (0, 0, 20, 80),
        0.95,
        0,
        "CLOSED",
        {"asset_id": "MCB-7", "observation_valid": True},
    )
    assert gate.update([closed], frame_id=1, observed_at_seconds=1.0)[0].class_name == "UNKNOWN"
    assert gate.update([closed], frame_id=2, observed_at_seconds=1.3)[0].class_name == "UNKNOWN"
    accepted = gate.update([closed], frame_id=3, observed_at_seconds=1.5)[0]
    assert accepted.class_name == "CLOSED"
    opened = Detection((0, 0, 20, 80), 0.95, 1, "OPEN", {"asset_id": "MCB-7"})
    assert gate.update([opened], frame_id=4, observed_at_seconds=1.6)[0].class_name == "OPEN"
    assert gate.update([closed], frame_id=5, observed_at_seconds=1.7)[0].class_name == "UNKNOWN"


def test_mobile_handle_geometry_returns_unknown_when_crop_has_no_handle() -> None:
    frame = np.full((120, 60, 3), 230, dtype=np.uint8)
    detection = Detection((0, 0, 60, 120), 0.95, 0, "MCB", {"track_id": "MCB-1"})
    result = classify_mcb_handle_geometry(frame, [detection])
    assert result[0].class_name == "UNKNOWN"
    assert result[0].metadata["observation_valid"] is False


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


def test_temporal_evidence_ignores_unknown_state() -> None:
    detector = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_evidence",
            trip_confirm_frames=4,
            micro_trip_min_frames=2,
            micro_trip_max_frames=3,
            recovery_consecutive_frames=2,
        )
    )
    unknown = Detection((0, 0, 10, 10), 0.0, -1, "UNKNOWN", {"roi_name": "QF1"})
    closed = Detection((0, 0, 10, 10), 0.95, 0, "CLOSE", {"roi_name": "QF1"})

    assert detector.update([unknown], frame_id=1) == []
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
