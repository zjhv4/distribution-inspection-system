from pathlib import Path

import numpy as np
import pytest

from edge_inspection.config import load_site_config
from edge_inspection import pipeline


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def get(self, property_id):
        if property_id == pipeline.cv2.CAP_PROP_FRAME_WIDTH:
            return 64
        if property_id == pipeline.cv2.CAP_PROP_FRAME_HEIGHT:
            return 48
        return 25

    def release(self):
        self.released = True


def test_pipeline_releases_resources_after_inference_error(monkeypatch, tmp_path: Path) -> None:
    capture = FakeCapture([np.zeros((48, 64, 3), dtype=np.uint8)])

    class FailingModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            raise RuntimeError("inference failed")

    class AlarmSink:
        closed = False

        def __init__(self, config):
            pass

        def close(self, **kwargs):
            type(self).closed = True

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(pipeline, "YoloDetector", FailingModel)
    monkeypatch.setattr(pipeline, "JsonlAlarmSink", AlarmSink)
    config = load_site_config("configs/site.yaml")
    config.alarm.jsonl_path = tmp_path / "alerts.jsonl"
    config.alarm.outbox_db_path = tmp_path / "outbox.sqlite3"

    with pytest.raises(RuntimeError, match="inference failed"):
        pipeline.run_video(source="video.mp4", config=config, task="intrusion")

    assert capture.released is True
    assert AlarmSink.closed is True


def test_pipeline_rejects_unopened_output_writer(monkeypatch, tmp_path: Path) -> None:
    capture = FakeCapture([])

    class Writer:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    writer = Writer()
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter", lambda *args: writer)

    with pytest.raises(RuntimeError, match="Cannot open output video"):
        pipeline.run_video(
            source="video.mp4",
            config=load_site_config("configs/site.yaml"),
            task="intrusion",
            output=tmp_path / "missing" / "result.mp4",
        )

    assert capture.released is True
    assert writer.released is True
