from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlertEvent:
    event_id: str
    schema_version: str
    phase: str
    task: str
    alert_type: str
    message: str
    confidence: float
    frame_id: int
    timestamp: str
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        task: str,
        alert_type: str,
        message: str,
        confidence: float,
        frame_id: int,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        phase: str = "START",
    ) -> "AlertEvent":
        return cls(
            event_id=event_id or str(uuid4()),
            schema_version="1.0",
            phase=phase.upper(),
            task=task,
            alert_type=alert_type,
            message=message,
            confidence=confidence,
            frame_id=frame_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
