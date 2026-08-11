from __future__ import annotations

import argparse
import hashlib
import html
import json
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class AlarmStore:
    """Idempotent backend store with a JSONL audit copy and SQLite index."""

    def __init__(self, path: Path, database_path: Path | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path or path.with_suffix(".sqlite3")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._archive_lock = threading.Lock()
        with self._connect() as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts(
                    delivery_id TEXT PRIMARY KEY,
                    event_id TEXT,
                    phase TEXT,
                    task TEXT,
                    payload_json TEXT NOT NULL,
                    received_at TEXT NOT NULL
                )
                """
            )

    def append(self, payload: dict, *, delivery_id: str | None = None) -> dict:
        resolved_id = str(delivery_id or payload.get("delivery_id") or _payload_delivery_id(payload))
        received_at = datetime.now(timezone.utc).isoformat()
        stored_payload = {**payload, "delivery_id": resolved_id}
        with self._connect() as database:
            cursor = database.execute(
                """
                INSERT OR IGNORE INTO alerts(
                    delivery_id, event_id, phase, task, payload_json, received_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    payload.get("event_id"),
                    payload.get("phase"),
                    payload.get("task"),
                    json.dumps(stored_payload, ensure_ascii=False),
                    received_at,
                ),
            )
            duplicate = cursor.rowcount == 0
        if not duplicate:
            with self._archive_lock, self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(stored_payload, ensure_ascii=False) + "\n")
        return {
            "acknowledged": True,
            "delivery_id": resolved_id,
            "duplicate": duplicate,
            "received_at": received_at,
        }

    def latest(self, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        with self._connect() as database:
            rows = database.execute(
                "SELECT payload_json, received_at FROM alerts ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            payload.setdefault("received_at", row["received_at"])
            result.append(payload)
        return result

    def get(self, delivery_id: str) -> dict | None:
        with self._connect() as database:
            row = database.execute(
                "SELECT payload_json, received_at FROM alerts WHERE delivery_id=?", (delivery_id,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return {
            "acknowledged": True,
            "delivery_id": delivery_id,
            "received_at": row["received_at"],
            "event": payload,
        }

    def count(self) -> int:
        with self._connect() as database:
            return int(database.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.database_path, timeout=5.0)
        database.row_factory = sqlite3.Row
        return database


def build_handler(store: AlarmStore):
    class AlarmHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/alerts"):
                limit = int(parse_qs(parsed.query).get("limit", ["100"])[0])
                body = render_page(store.latest(limit)).encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
                return
            if path == "/api/alerts":
                limit = int(parse_qs(parsed.query).get("limit", ["100"])[0])
                self._send_json(200, {"alerts": store.latest(limit), "count": store.count()})
                return
            if path == "/health":
                self._send_json(200, {"ok": True, "stored_alerts": store.count()})
                return
            if path.startswith("/acks/"):
                delivery_id = path.removeprefix("/acks/")
                result = store.get(delivery_id)
                if result is None:
                    self._send_json(404, {"acknowledged": False, "delivery_id": delivery_id})
                else:
                    self._send_json(200, result)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/alerts":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_error(400, "Invalid JSON")
                return
            if not isinstance(payload, dict):
                self.send_error(400, "JSON object required")
                return
            result = store.append(
                payload,
                delivery_id=self.headers.get("Idempotency-Key"),
            )
            self._send_json(200, result)

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(
                status,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    return AlarmHandler


def render_page(rows: list[dict]) -> str:
    items = []
    for row in rows:
        metadata = row.get("metadata") or {}
        reason = metadata.get("reason_code") or metadata.get("verification_status") or ""
        items.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(row.get('task', '')))}</td>"
            f"<td>{html.escape(str(row.get('alert_type', '')))}</td>"
            f"<td>{html.escape(str(row.get('phase', '')))}</td>"
            f"<td>{html.escape(str(reason))}</td>"
            f"<td>{html.escape(str(row.get('confidence', '')))}</td>"
            f"<td>{html.escape(str(row.get('message', '')))}</td>"
            "</tr>"
        )
    table = "\n".join(items) or "<tr><td colspan='7'>暂无告警</td></tr>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="5">
  <title>配电巡检告警</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 10px 8px; text-align: left; }}
    th {{ background: #f0f4f8; }}
  </style>
</head>
<body>
  <h1>配电巡检告警</h1>
  <table>
    <thead><tr><th>时间</th><th>任务</th><th>类型</th><th>阶段</th><th>原因/状态</th><th>置信度</th><th>信息</th></tr></thead>
    <tbody>{table}</tbody>
  </table>
</body>
</html>"""


def _payload_delivery_id(payload: dict) -> str:
    event_id = payload.get("event_id")
    phase = str(payload.get("phase", "START")).upper()
    if event_id:
        source = f"{event_id}:{phase}"
    else:
        source = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive, acknowledge and display edge inspection alarms")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--store", default="alarms/server_alerts.jsonl")
    parser.add_argument("--database", default=None)
    args = parser.parse_args()

    store = AlarmStore(Path(args.store), Path(args.database) if args.database else None)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(store))
    print(f"Alarm server listening on http://{args.host}:{args.port}")
    print(f"Webhook endpoint: http://{args.host}:{args.port}/alerts")
    print(f"Health endpoint: http://{args.host}:{args.port}/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
