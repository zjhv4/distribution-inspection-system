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
