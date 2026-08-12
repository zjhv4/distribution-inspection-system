import pytest

from edge_inspection.config import load_site_config
from edge_inspection.pipeline import resolve_intrusion_weights


def test_load_site_config() -> None:
    config = load_site_config("configs/default.yaml")
    assert config.models.intrusion
    assert config.intrusion.zones[0].name == "restricted_area"
    assert "TRIP" in config.breaker.abnormal_classes


def test_default_config_selects_power_visible_profile() -> None:
    config = load_site_config("configs/default.yaml")
    assert config.intrusion.model_profile == "power_visible"
    assert resolve_intrusion_weights(config) == "models/intrusion_power_visible_yolo11l.pt"


def test_unknown_intrusion_profile_fails_closed() -> None:
    config = load_site_config("configs/default.yaml")
    config.intrusion.model_profile = "not_declared"
    with pytest.raises(RuntimeError, match="no matching model"):
        resolve_intrusion_weights(config)


def test_default_breaker_uses_time_based_debounce() -> None:
    config = load_site_config("configs/default.yaml")
    assert config.breaker.temporal_seconds_enabled is True
    assert config.breaker.observation_confidence == pytest.approx(0.60)
    assert config.breaker.arm_closed_seconds == pytest.approx(30.0)
    assert config.breaker.micro_trip_max_seconds < config.breaker.trip_confirm_seconds


def test_default_breaker_uses_mobile_device_detection() -> None:
    config = load_site_config("configs/default.yaml")
    assert config.breaker.mode == "mobile_detection"
    assert config.models.breaker == "models/breaker_mobile_types_yolo11s.pt"
    assert config.models.breaker_state_classifier == "models/breaker_mcb_state_yolo11s_cls.pt"
    assert config.breaker.mobile_device_classes == ["MCB"]
    assert config.breaker.mobile_class_limits == {"MCB": 10, "RCD": 2, "ISOLATOR": 1}
