from pathlib import Path

from scripts.alarm_server import AlarmStore, render_page


def test_alarm_store_roundtrip(tmp_path: Path) -> None:
    store = AlarmStore(tmp_path / "alerts.jsonl")
    store.append({"task": "breaker", "alert_type": "trip"})
    assert store.latest()[0]["alert_type"] == "trip"


def test_alarm_store_deduplicates_delivery_id(tmp_path: Path) -> None:
    store = AlarmStore(tmp_path / "alerts.jsonl")
    first = store.append({"event_id": "e1", "phase": "START", "task": "intrusion"})
    second = store.append({"event_id": "e1", "phase": "START", "task": "intrusion"})
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert store.count() == 1
    assert store.get(first["delivery_id"])["acknowledged"] is True


def test_render_page_escapes_content() -> None:
    page = render_page([{"message": "<script>alert(1)</script>"}])
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "阶段" in page
