from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import Detection


class YoloDetector:
    """Small adapter around Ultralytics YOLO so the rest of the code stays testable."""

    def __init__(self, weights: str | Path, *, device: str = "cpu", imgsz: int = 640, half: bool = False):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Please install dependencies with `pip install -r requirements.txt`.") from exc

        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.half = half

    def predict(
        self,
        frame: Any,
        *,
        conf: float,
        iou: float,
        track: bool = False,
        tracker: str = "bytetrack.yaml",
    ) -> list[Detection]:
        predict_args = {
            "conf": conf,
            "iou": iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            predict_args["half"] = True
        results = (
            self.model.track(frame, persist=True, tracker=tracker, **predict_args)
            if track
            else self.model.predict(frame, **predict_args)
        )
        if not results:
            return []
        return self._result_to_detections(results[0])

    def predict_tiled(
        self,
        frame: Any,
        *,
        conf: float,
        iou: float,
        columns: int,
        rows: int,
        overlap: float,
    ) -> list[Detection]:
        """Detect small objects by scanning overlapping tiles across the whole frame."""
        if columns < 1 or rows < 1:
            raise ValueError("Tile columns and rows must be positive.")
        if not 0 <= overlap < 1:
            raise ValueError("Tile overlap must be in [0, 1).")

        height, width = frame.shape[:2]
        tile_width = min(width, max(1, int(width / (columns - (columns - 1) * overlap))))
        tile_height = min(height, max(1, int(height / (rows - (rows - 1) * overlap))))
        step_x = max(1, int(tile_width * (1 - overlap)))
        step_y = max(1, int(tile_height * (1 - overlap)))
        x_offsets = [min(index * step_x, width - tile_width) for index in range(columns)]
        y_offsets = [min(index * step_y, height - tile_height) for index in range(rows)]
        offsets = [(x, y) for y in y_offsets for x in x_offsets]
        tiles = [frame[y : y + tile_height, x : x + tile_width] for x, y in offsets]

        predict_args = {
            "conf": conf,
            "iou": iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            predict_args["half"] = True
        results = self.model.predict(tiles, **predict_args)
        detections: list[Detection] = []
        for (offset_x, offset_y), result in zip(offsets, results, strict=True):
            for detection in self._result_to_detections(result):
                x1, y1, x2, y2 = detection.bbox
                detections.append(
                    Detection(
                        bbox=(
                            x1 + offset_x,
                            y1 + offset_y,
                            x2 + offset_x,
                            y2 + offset_y,
                        ),
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        metadata={**detection.metadata, "detection_source": "tiled"},
                    )
                )
        return detections

    @staticmethod
    def _result_to_detections(result: Any) -> list[Detection]:
        names = result.names or {}
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().tolist()
            class_id = int(box.cls[0].detach().cpu().item())
            confidence = float(box.conf[0].detach().cpu().item())
            metadata = {}
            if getattr(box, "id", None) is not None:
                metadata["track_id"] = int(box.id[0].detach().cpu().item())
            detections.append(
                Detection(
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                    metadata=metadata,
                )
            )
        return detections


class YoloClassifier:
    """Adapter for Ultralytics classification models, including OpenVINO exports."""

    def __init__(self, weights: str | Path, *, device: str = "cpu", imgsz: int = 224, half: bool = False):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Please install dependencies with `pip install -r requirements.txt`.") from exc

        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.half = half

    def predict(self, frame: Any) -> Detection:
        return self.predict_many([frame])[0]

    def predict_many(self, frames: list[Any]) -> list[Detection]:
        if not frames:
            return []
        predict_args = {"imgsz": self.imgsz, "device": self.device, "verbose": False}
        if self.half:
            predict_args["half"] = True
        results = self.model.predict(frames, **predict_args)
        if not results:
            raise RuntimeError("Classifier returned no results.")
        detections: list[Detection] = []
        for frame, result in zip(frames, results, strict=True):
            if result.probs is None:
                raise RuntimeError("Classifier result has no probabilities.")
            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf.detach().cpu().item())
            names = result.names or {}
            probability_values = result.probs.data.detach().cpu().tolist()
            class_probabilities = {
                str(names.get(index, index)).upper(): float(probability)
                for index, probability in enumerate(probability_values)
            }
            height, width = frame.shape[:2]
            detections.append(
                Detection(
                    bbox=(0.0, 0.0, float(width), float(height)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)).upper(),
                    metadata={"class_probabilities": class_probabilities},
                )
            )
        return detections
