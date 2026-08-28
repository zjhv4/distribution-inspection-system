from pathlib import Path

import pytest

from scripts.alarm_server import (
    AlarmStore,
    _has_valid_token,
    _is_loopback_host,
    _parse_content_length,
    render_page,
)


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


def test_loopback_host_detection() -> None:
    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("::1") is True
    assert _is_loopback_host("0.0.0.0") is False


def test_alarm_server_token_validation() -> None:
    assert _has_valid_token("", None) is True
    assert _has_valid_token("Bearer site-secret", "site-secret") is True
    assert _has_valid_token("Bearer wrong", "site-secret") is False


def test_alarm_server_limits_request_body() -> None:
    assert _parse_content_length("128", 128) == 128
    with pytest.raises(ValueError):
        _parse_content_length("invalid", 128)
    with pytest.raises(ValueError):
        _parse_content_length("0", 128)
    with pytest.raises(OverflowError):
        _parse_content_length("129", 128)
