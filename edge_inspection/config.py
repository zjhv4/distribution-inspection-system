from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


Point = tuple[float, float]


@dataclass
class ModelsConfig:
    intrusion: str = "yolo11n.pt"
    intrusion_profiles: dict[str, str] = field(default_factory=dict)
    breaker: str = "runs/detect/train/weights/best.pt"
    breaker_classifier: str | None = None
    breaker_anomaly: str | None = None
    breaker_anomaly_calibration: str | None = None


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    imgsz: int = 640
    half: bool = False


@dataclass
class AccessWindowConfig:
    """Weekly local-time access window. Monday is day 0 and Sunday is day 6."""

    days: list[int] = field(default_factory=lambda: list(range(7)))
    start: str = "00:00"
    end: str = "23:59"


@dataclass
class ZoneConfig:
    name: str
    polygon: list[Point]
    access_policy: str = "authorized_only"
    allowed_identity_ids: list[str] = field(default_factory=list)
    access_windows: list[AccessWindowConfig] = field(default_factory=list)
    unknown_identity_action: str = "alert"


@dataclass
class IntrusionConfig:
    zones: list[ZoneConfig]
    model_profile: str | None = None
    person_class_names: list[str] = field(default_factory=lambda: ["person"])
    confidence: float = 0.45
    iou: float = 0.45
    min_consecutive_frames: int = 3
    recovery_consecutive_frames: int = 3
    footpoint_ratio: float = 0.92
    timezone: str = "Asia/Hong_Kong"
    tracker_iou_threshold: float = 0.15
    tracker_max_missing_frames: int = 15
    use_model_tracking: bool = True
    tracker: str = "bytetrack.yaml"
    identity_context_path: Path | None = None
    identity_context_ttl_seconds: float = 5.0


@dataclass
class BreakerRoiConfig:
    name: str
    bbox: tuple[float, float, float, float]


@dataclass
class BreakerConfig:
    mode: str = "detection"
    decision_mode: str = "direct_classes"
    confidence: float = 0.50
    classifier_confidence: float = 0.50
    classifier_imgsz: int = 224
    iou: float = 0.45
    abnormal_classes: list[str] = field(default_factory=lambda: ["TRIP", "MICRO_TRIP"])
    normal_classes: list[str] = field(default_factory=lambda: ["CLOSED", "OPEN"])
    class_thresholds: dict[str, float] = field(
        default_factory=lambda: {"TRIP": 0.50, "MICRO_TRIP": 0.50}
    )
    min_consecutive_frames: int = 2
    recovery_consecutive_frames: int = 2
    open_classes: list[str] = field(default_factory=lambda: ["OPEN"])
    closed_classes: list[str] = field(default_factory=lambda: ["CLOSE", "CLOSED"])
    deviation_classes: list[str] = field(
        default_factory=lambda: ["OPEN", "ANOMALY", "INTERMEDIATE", "VISUAL_DEVIATION"]
    )
    anomaly_score_threshold: float = 0.50
    anomaly_backend: str = "reconstruction"
    anomaly_bootstrap_frames: int = 100
    anomaly_bank_size: int = 300
    anomaly_sample_stride: int = 15
    anomaly_neighbors: int = 5
    anomaly_normal_confidence: float = 0.90
    anomaly_normal_quantile: float = 0.995
    anomaly_min_raw_threshold: float = 0.02
    micro_trip_min_frames: int = 2
    trip_confirm_frames: int = 5
    micro_trip_max_frames: int = 4
    max_missing_frames: int = 1
    command_metadata_keys: list[str] = field(
        default_factory=lambda: ["commanded_open", "expected_open", "maintenance_mode"]
    )
    trip_confirmation_keys: list[str] = field(
        default_factory=lambda: ["protection_trip", "trip_coil_active", "auxiliary_trip"]
    )
    rois: list[BreakerRoiConfig] = field(default_factory=list)


@dataclass
class AlarmConfig:
    jsonl_path: Path = Path("alarms/edge_alerts.jsonl")
    save_snapshots: bool = True
    snapshot_dir: Path = Path("alarms/snapshots")
    webhook_url: str | None = None
    webhook_timeout_seconds: float = 3.0
    webhook_retries: int = 2
    outbox_db_path: Path = Path("alarms/alarm_outbox.sqlite3")
    background_delivery: bool = True
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 60.0
    webhook_max_attempts: int = 20


@dataclass
class SiteConfig:
    models: ModelsConfig
    intrusion: IntrusionConfig
    breaker: BreakerConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)


TaskName = Literal["intrusion", "breaker", "all"]


def load_site_config(path: str | Path) -> SiteConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return _site_config_from_dict(raw)


def _site_config_from_dict(raw: dict) -> SiteConfig:
    zones = []
    for item in raw["intrusion"]["zones"]:
        zone_raw = dict(item)
        zone_raw["polygon"] = [(float(x), float(y)) for x, y in zone_raw["polygon"]]
        zone_raw["access_windows"] = [
            AccessWindowConfig(
                days=[int(day) for day in window.get("days", range(7))],
                start=str(window.get("start", "00:00")),
                end=str(window.get("end", "23:59")),
            )
            for window in zone_raw.get("access_windows", [])
        ]
        zone = ZoneConfig(**zone_raw)
        if zone.access_policy not in {"deny_all", "authorized_only", "scheduled_authorized"}:
            raise ValueError(
                f"intrusion zone {zone.name!r} access_policy must be deny_all, "
                "authorized_only, or scheduled_authorized"
            )
        if zone.unknown_identity_action not in {"alert", "review", "allow"}:
            raise ValueError(
                f"intrusion zone {zone.name!r} unknown_identity_action must be alert, review, or allow"
            )
        for window in zone.access_windows:
            _validate_access_window(zone.name, window)
        zones.append(zone)
    intrusion_raw = {**raw["intrusion"], "zones": zones}
    if intrusion_raw.get("identity_context_path"):
        intrusion_raw["identity_context_path"] = Path(intrusion_raw["identity_context_path"])
    alarm_raw = raw.get("alarm", {})
    if "jsonl_path" in alarm_raw:
        alarm_raw["jsonl_path"] = Path(alarm_raw["jsonl_path"])
    if "snapshot_dir" in alarm_raw:
        alarm_raw["snapshot_dir"] = Path(alarm_raw["snapshot_dir"])
    if "outbox_db_path" in alarm_raw:
        alarm_raw["outbox_db_path"] = Path(alarm_raw["outbox_db_path"])

    breaker_raw = dict(raw.get("breaker", {}))
    for key in (
        "abnormal_classes",
        "normal_classes",
        "open_classes",
        "closed_classes",
        "deviation_classes",
    ):
        if key in breaker_raw:
            breaker_raw[key] = [str(value).upper() for value in breaker_raw[key]]
    if "class_thresholds" in breaker_raw:
        breaker_raw["class_thresholds"] = {
            str(key).upper(): float(value) for key, value in breaker_raw["class_thresholds"].items()
        }
    decision_mode = breaker_raw.get("decision_mode", "direct_classes")
    if decision_mode in {"temporal_open", "temporal_evidence"}:
        trip_frames = int(breaker_raw.get("trip_confirm_frames", 5))
        micro_min_frames = int(breaker_raw.get("micro_trip_min_frames", 2))
        micro_frames = int(breaker_raw.get("micro_trip_max_frames", 4))
        if trip_frames < 2:
            raise ValueError("breaker.trip_confirm_frames must be at least 2")
        if micro_min_frames < 1 or micro_min_frames > micro_frames:
            raise ValueError("breaker.micro_trip_min_frames must be in [1, micro_trip_max_frames]")
        if micro_frames >= trip_frames:
            raise ValueError("breaker.micro_trip_max_frames must be less than trip_confirm_frames")
        anomaly_threshold = float(breaker_raw.get("anomaly_score_threshold", 0.50))
        if not 0 <= anomaly_threshold <= 1:
            raise ValueError("breaker.anomaly_score_threshold must be in [0, 1]")
        anomaly_backend = str(breaker_raw.get("anomaly_backend", "reconstruction"))
        if anomaly_backend not in {"reconstruction", "dinov2_reference"}:
            raise ValueError("breaker.anomaly_backend must be reconstruction or dinov2_reference")
        if int(breaker_raw.get("anomaly_bootstrap_frames", 100)) < 2:
            raise ValueError("breaker.anomaly_bootstrap_frames must be at least 2")
        if int(breaker_raw.get("anomaly_neighbors", 5)) < 1:
            raise ValueError("breaker.anomaly_neighbors must be positive")
    breaker_rois = [
        BreakerRoiConfig(name=item["name"], bbox=tuple(float(value) for value in item["bbox"]))
        for item in breaker_raw.pop("rois", [])
    ]

    return SiteConfig(
        models=ModelsConfig(**raw.get("models", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        intrusion=IntrusionConfig(**intrusion_raw),
        breaker=BreakerConfig(**breaker_raw, rois=breaker_rois),
        alarm=AlarmConfig(**alarm_raw),
    )


def _validate_access_window(zone_name: str, window: AccessWindowConfig) -> None:
    for day in window.days:
        if day < 0 or day > 6:
            raise ValueError(f"intrusion zone {zone_name!r} access window days must be in [0, 6]")
    for label, value in (("start", window.start), ("end", window.end)):
        try:
            hour, minute = (int(part) for part in value.split(":"))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"intrusion zone {zone_name!r} access window {label} must use HH:MM"
            ) from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(
                f"intrusion zone {zone_name!r} access window {label} must use valid HH:MM"
            )
