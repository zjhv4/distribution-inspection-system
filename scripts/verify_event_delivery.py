from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_inspection.alarm import JsonlAlarmSink
from edge_inspection.breaker import BreakerStateDetector
from edge_inspection.config import AlarmConfig, BreakerConfig, IntrusionConfig, ZoneConfig
from edge_inspection.events import Detection
from edge_inspection.intrusion import IntrusionDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify event lifecycle and webhook delivery")
    parser.add_argument("--webhook", default="http://127.0.0.1:18088/alerts")
    parser.add_argument("--output", default="output/event_delivery.json")
    args = parser.parse_args()

    output = Path(args.output)
    local_events = output.with_suffix(".events.jsonl")
    if local_events.exists():
        local_events.unlink()
    sink = JsonlAlarmSink(
        AlarmConfig(
            jsonl_path=local_events,
            save_snapshots=False,
            webhook_url=args.webhook,
            webhook_timeout_seconds=3,
            webhook_retries=2,
            outbox_db_path=output.with_suffix(".outbox.sqlite3"),
            background_delivery=False,
            webhook_max_attempts=3,
        )
    )

    intrusion = IntrusionDetector(
        IntrusionConfig(
            zones=[ZoneConfig("restricted_area", [(0, 0), (100, 0), (100, 100), (0, 100)])],
            min_consecutive_frames=2,
            recovery_consecutive_frames=2,
        )
    )
    person = Detection((20, 10, 40, 80), 0.97, 0, "person")
    emitted = []
    for frame_id, detections in enumerate(([person], [person], [person], [], []), start=1):
        emitted.extend(intrusion.update(detections, frame_id=frame_id))

    breaker = BreakerStateDetector(
        BreakerConfig(
            decision_mode="temporal_open",
            trip_confirm_frames=3,
            micro_trip_max_frames=2,
            recovery_consecutive_frames=2,
        )
    )
    opened = Detection((0, 0, 20, 20), 0.96, 1, "OPEN", {"roi_name": "QF1"})
    closed = Detection((0, 0, 20, 20), 0.96, 0, "CLOSE", {"roi_name": "QF1"})
    for offset, detection in enumerate((opened, opened, opened, closed, closed)):
        emitted.extend(breaker.update([detection], frame_id=10 + offset))
    for offset, detection in enumerate((opened, opened, closed, closed)):
        emitted.extend(breaker.update([detection], frame_id=20 + offset))

    for event in emitted:
        sink.emit(event)
    delivery_complete = sink.flush(timeout_seconds=5, force=True)
    delivery_stats = sink.stats()
    sink.close(flush=False)

    rows = [json.loads(line) for line in local_events.read_text(encoding="utf-8").splitlines()]
    result = {
        "ok": len(rows) == 6,
        "webhook": args.webhook,
        "events": [
            {
                "event_id": row["event_id"],
                "task": row["task"],
                "alert_type": row["alert_type"],
                "phase": row["phase"],
            }
            for row in rows
        ],
        "checks": {
            "intrusion_start_recovered": count_pair(rows, "PERSON_INTRUSION"),
            "trip_start_recovered": count_pair(rows, "TRIP"),
            "micro_trip_start_recovered": count_pair(rows, "MICRO_TRIP"),
            "no_per_frame_duplicates": len(rows) == 6,
            "backend_acknowledged_all": delivery_complete and delivery_stats["ACKED"] >= 6,
        },
        "delivery_stats": delivery_stats,
    }
    result["ok"] = result["ok"] and all(result["checks"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


def count_pair(rows: list[dict], alert_type: str) -> bool:
    selected = [row for row in rows if row["alert_type"] == alert_type]
    return (
        len(selected) == 2
        and {row["phase"] for row in selected} == {"START", "RECOVERED"}
        and len({row["event_id"] for row in selected}) == 1
    )


if __name__ == "__main__":
    main()
