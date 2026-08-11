from pathlib import Path

import pytest

from scripts.evaluate import evaluate_frame_false_positive_rate
from scripts.sweep_alarm_class_thresholds import choose_best_row


class _FakeModel:
    model = type("Model", (), {"names": {0: "trip"}})()

    def predict(self, *args, **kwargs):
        return []


def test_evaluate_fpr_without_negative_dir_uses_proxy_mode() -> None:
    report = evaluate_frame_false_positive_rate(
        model=_FakeModel(),
        negative_dir=None,
        alarm_classes=None,
        imgsz=640,
        device="cpu",
        conf=0.25,
        iou=0.6,
    )
    assert report["mode"] == "precision_proxy"


def test_evaluate_fpr_requires_images(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        evaluate_frame_false_positive_rate(
            model=_FakeModel(),
            negative_dir=tmp_path,
            alarm_classes=None,
            imgsz=640,
            device="cpu",
            conf=0.25,
            iou=0.6,
        )


def test_choose_best_row_prefers_accepted_high_recall() -> None:
    rows = [
        {
            "threshold": 0.25,
            "precision": 0.99,
            "recall": 0.94,
            "false_positive_rate": 0.0,
            "f1": 0.96,
            "accepted": False,
        },
        {
            "threshold": 0.35,
            "precision": 0.96,
            "recall": 0.95,
            "false_positive_rate": 0.02,
            "f1": 0.955,
            "accepted": True,
        },
        {
            "threshold": 0.40,
            "precision": 0.98,
            "recall": 0.951,
            "false_positive_rate": 0.01,
            "f1": 0.965,
            "accepted": True,
        },
    ]

    assert choose_best_row(rows, max_fpr=0.03)["threshold"] == 0.40
