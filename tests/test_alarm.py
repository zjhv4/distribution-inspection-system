import json

from edge_inspection.alarm import JsonlAlarmSink, delivery_id_for
from edge_inspection.config import AlarmConfig
from edge_inspection.events import AlertEvent


def test_alarm_sink_writes_jsonl(tmp_path) -> None:
    sink = JsonlAlarmSink(AlarmConfig(jsonl_path=tmp_path / "alerts.jsonl", save_snapshots=False))
    event = AlertEvent.create(
        task="intrusion",
        alert_type="person_intrusion",
        message="人员进入监测区域",
        confidence=0.98,
        frame_id=1,
    )

    sink.emit(event)

    payload = json.loads((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip())
    assert payload["task"] == "intrusion"
    assert payload["alert_type"] == "person_intrusion"
    assert payload["delivery_id"] == delivery_id_for(event.event_id, event.phase)
    assert sink.stats()["LOCAL_ONLY"] == 1


def test_alarm_sink_records_failed_webhook(tmp_path, monkeypatch) -> None:
    def fail_post(*args, **kwargs):
        import requests

        raise requests.Timeout("timeout")

    monkeypatch.setattr("edge_inspection.alarm.requests.post", fail_post)
    sink = JsonlAlarmSink(
        AlarmConfig(
            jsonl_path=tmp_path / "alerts.jsonl",
            save_snapshots=False,
            webhook_url="http://127.0.0.1:9/alerts",
            webhook_retries=1,
            webhook_max_attempts=1,
            background_delivery=False,
            outbox_db_path=tmp_path / "outbox.sqlite3",
        )
    )

    event = AlertEvent.create(
        task="breaker",
        alert_type="trip",
        message="断路器异常状态：trip",
        confidence=0.96,
        frame_id=3,
    )
    sink.emit(event)

    failed = tmp_path / "alerts.webhook_failed.jsonl"
    assert failed.exists()
    payload = json.loads(failed.read_text(encoding="utf-8").strip())
    assert payload["event"]["alert_type"] == "trip"
    assert sink.stats()["DEAD"] == 1


def test_alarm_sink_is_idempotent_by_event_and_phase(tmp_path) -> None:
    sink = JsonlAlarmSink(
        AlarmConfig(
            jsonl_path=tmp_path / "alerts.jsonl",
            save_snapshots=False,
            outbox_db_path=tmp_path / "outbox.sqlite3",
        )
    )
    event = AlertEvent.create(
        task="breaker",
        alert_type="TRIP",
        message="trip",
        confidence=0.9,
        frame_id=1,
    )
    sink.emit(event)
    sink.emit(event)
    assert len((tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_alarm_sink_accepts_backend_ack(tmp_path, monkeypatch) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"acknowledged": True, "delivery_id": delivery_id}

    monkeypatch.setattr("edge_inspection.alarm.requests.post", lambda *args, **kwargs: Response())
    sink = JsonlAlarmSink(
        AlarmConfig(
            jsonl_path=tmp_path / "alerts.jsonl",
            save_snapshots=False,
            webhook_url="http://backend/alerts",
            background_delivery=False,
            outbox_db_path=tmp_path / "outbox.sqlite3",
        )
    )
    event = AlertEvent.create(
        task="intrusion",
        alert_type="PERSON_INTRUSION",
        message="intrusion",
        confidence=0.9,
        frame_id=1,
    )
    delivery_id = delivery_id_for(event.event_id, event.phase)
    sink.emit(event)
    assert sink.delivery_status(delivery_id) == "ACKED"


def test_alarm_outbox_retries_after_process_restart(tmp_path, monkeypatch) -> None:
    import requests

    config = AlarmConfig(
        jsonl_path=tmp_path / "alerts.jsonl",
        save_snapshots=False,
        webhook_url="http://backend/alerts",
        webhook_retries=1,
        webhook_max_attempts=3,
        retry_base_seconds=0,
        background_delivery=False,
        outbox_db_path=tmp_path / "outbox.sqlite3",
    )
    monkeypatch.setattr(
        "edge_inspection.alarm.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("offline")),
    )
    event = AlertEvent.create(
        task="breaker", alert_type="TRIP", message="trip", confidence=0.9, frame_id=1
    )
    delivery_id = delivery_id_for(event.event_id, event.phase)
    first_process = JsonlAlarmSink(config)
    first_process.emit(event)
    assert first_process.delivery_status(delivery_id) == "RETRY"
    first_process.close(flush=False)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"acknowledged": True, "delivery_id": delivery_id}

    monkeypatch.setattr("edge_inspection.alarm.requests.post", lambda *args, **kwargs: Response())
    restarted_process = JsonlAlarmSink(config)
    assert restarted_process.flush(timeout_seconds=1, force=True) is True
    assert restarted_process.delivery_status(delivery_id) == "ACKED"
