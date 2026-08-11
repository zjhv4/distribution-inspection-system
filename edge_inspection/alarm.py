from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import requests

from .config import AlarmConfig
from .events import AlertEvent


DEFAULT_OUTBOX_PATH = Path("alarms/alarm_outbox.sqlite3")


class JsonlAlarmSink:
    """Local audit log plus durable, idempotent webhook outbox.

    ``emit`` commits the delivery to SQLite before returning.  Network I/O is
    performed by a background worker by default, so a slow or unavailable
    backend cannot stall video inference.  START and RECOVERED phases receive
    distinct deterministic delivery IDs and are safe to retry.
    """

    def __init__(self, config: AlarmConfig):
        self.config = config
        self.config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.save_snapshots:
            self.config.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_path = (
            self.config.jsonl_path.with_suffix(".outbox.sqlite3")
            if self.config.outbox_db_path == DEFAULT_OUTBOX_PATH
            else self.config.outbox_db_path
        )
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._initialize_database()
        if self.config.webhook_url and self.config.background_delivery:
            self._worker = threading.Thread(
                target=self._delivery_loop,
                name="alarm-outbox-delivery",
                daemon=True,
            )
            self._worker.start()

    def emit(self, event: AlertEvent, frame: Any | None = None) -> Path | None:
        payload = event.to_dict()
        delivery_id = delivery_id_for(event.event_id, event.phase)
        payload["delivery_id"] = delivery_id
        snapshot_path: Path | None = None

        if frame is not None and self.config.save_snapshots:
            snapshot_name = (
                f"{event.task}_{event.alert_type}_{event.event_id}_{event.phase.lower()}.jpg"
            )
            snapshot_path = self.config.snapshot_dir / snapshot_name
            if not snapshot_path.exists():
                cv2.imwrite(str(snapshot_path), frame)
            payload["snapshot_path"] = str(snapshot_path)

        inserted = self._enqueue(delivery_id, payload)
        if inserted:
            with self._lock, self.config.jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if self.config.webhook_url:
            if self.config.background_delivery:
                self._wake.set()
            else:
                for _ in range(max(1, self.config.webhook_retries)):
                    if not self.deliver_pending_once(force=True):
                        break
                    if self.delivery_status(delivery_id) == "ACKED":
                        break
        return snapshot_path

    def deliver_pending_once(self, *, force: bool = False) -> bool:
        row = self._next_delivery(force=force)
        if row is None:
            return False
        delivery_id = str(row["delivery_id"])
        payload = json.loads(str(row["payload_json"]))
        try:
            response = requests.post(
                str(self.config.webhook_url),
                json=payload,
                headers={"Idempotency-Key": delivery_id},
                timeout=self.config.webhook_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            acknowledged = bool(body.get("acknowledged", body.get("ok", False)))
            returned_id = body.get("delivery_id")
            if not acknowledged:
                raise requests.RequestException("backend response did not acknowledge the delivery")
            if returned_id is not None and str(returned_id) != delivery_id:
                raise requests.RequestException("backend acknowledged a different delivery_id")
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            self._mark_failed(row, exc)
            return True

        with self._connect() as database:
            database.execute(
                """
                UPDATE deliveries
                SET status='ACKED', acked_at=?, last_error=NULL
                WHERE delivery_id=?
                """,
                (_utc_iso(), delivery_id),
            )
        return True

    def flush(self, timeout_seconds: float = 5.0, *, force: bool = False) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() <= deadline:
            if self.pending_count() == 0:
                return True
            if self.config.webhook_url:
                self.deliver_pending_once(force=force)
            else:
                return True
            time.sleep(0.01)
        return self.pending_count() == 0

    def close(self, *, flush: bool = True, timeout_seconds: float = 5.0) -> bool:
        completed = self.flush(timeout_seconds, force=False) if flush else self.pending_count() == 0
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
        return completed

    def pending_count(self) -> int:
        with self._connect() as database:
            row = database.execute(
                "SELECT COUNT(*) FROM deliveries WHERE status IN ('PENDING', 'RETRY')"
            ).fetchone()
        return int(row[0])

    def delivery_status(self, delivery_id: str) -> str | None:
        with self._connect() as database:
            row = database.execute(
                "SELECT status FROM deliveries WHERE delivery_id=?", (delivery_id,)
            ).fetchone()
        return None if row is None else str(row[0])

    def stats(self) -> dict[str, int]:
        with self._connect() as database:
            rows = database.execute(
                "SELECT status, COUNT(*) AS count FROM deliveries GROUP BY status"
            ).fetchall()
        result = {str(row[0]): int(row[1]) for row in rows}
        for status in ("PENDING", "RETRY", "ACKED", "DEAD", "LOCAL_ONLY"):
            result.setdefault(status, 0)
        return result

    def _enqueue(self, delivery_id: str, payload: dict[str, Any]) -> bool:
        status = "PENDING" if self.config.webhook_url else "LOCAL_ONLY"
        with self._connect() as database:
            cursor = database.execute(
                """
                INSERT OR IGNORE INTO deliveries(
                    delivery_id, event_id, phase, task, payload_json, status,
                    attempts, next_attempt_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    delivery_id,
                    str(payload.get("event_id", "")),
                    str(payload.get("phase", "")),
                    str(payload.get("task", "")),
                    json.dumps(payload, ensure_ascii=False),
                    status,
                    time.time(),
                    _utc_iso(),
                ),
            )
        return cursor.rowcount == 1

    def _next_delivery(self, *, force: bool) -> sqlite3.Row | None:
        condition = "1=1" if force else "next_attempt_at <= ?"
        parameters: tuple[Any, ...] = () if force else (time.time(),)
        with self._connect() as database:
            return database.execute(
                f"""
                SELECT * FROM deliveries
                WHERE status IN ('PENDING', 'RETRY') AND {condition}
                ORDER BY created_at, delivery_id
                LIMIT 1
                """,
                parameters,
            ).fetchone()

    def _mark_failed(self, row: sqlite3.Row, error: Exception) -> None:
        attempts = int(row["attempts"]) + 1
        terminal = attempts >= max(1, self.config.webhook_max_attempts)
        status = "DEAD" if terminal else "RETRY"
        delay = min(
            self.config.retry_max_seconds,
            self.config.retry_base_seconds * (2 ** max(0, attempts - 1)),
        )
        with self._connect() as database:
            database.execute(
                """
                UPDATE deliveries
                SET status=?, attempts=?, next_attempt_at=?, last_error=?
                WHERE delivery_id=?
                """,
                (status, attempts, time.time() + delay, str(error), row["delivery_id"]),
            )
        failed_path = self.config.jsonl_path.with_suffix(".webhook_failed.jsonl")
        failure = {
            "delivery_id": row["delivery_id"],
            "webhook_url": self.config.webhook_url,
            "attempt": attempts,
            "terminal": terminal,
            "error": str(error),
            "event": json.loads(str(row["payload_json"])),
            "failed_at": _utc_iso(),
        }
        with self._lock, failed_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(failure, ensure_ascii=False) + "\n")

    def _delivery_loop(self) -> None:
        while not self._stop.is_set():
            delivered = self.deliver_pending_once(force=False)
            if delivered:
                continue
            self._wake.wait(timeout=0.5)
            self._wake.clear()

    def _initialize_database(self) -> None:
        with self._connect() as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS deliveries(
                    delivery_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    task TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    next_attempt_at REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    acked_at TEXT,
                    last_error TEXT
                )
                """
            )
            database.execute(
                "CREATE INDEX IF NOT EXISTS idx_deliveries_pending ON deliveries(status, next_attempt_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.outbox_path, timeout=5.0)
        database.row_factory = sqlite3.Row
        return database


def delivery_id_for(event_id: str, phase: str) -> str:
    source = f"{event_id}:{phase.upper()}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
