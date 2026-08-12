from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import BreakerConfig, BreakerRoiConfig
from .events import Detection


class BreakerReferenceClassifier:
    """Classify a fixed breaker ROI against site-specific handle references."""

    def __init__(self, config: BreakerConfig) -> None:
        self.config = config
        self._references: dict[str, dict[str, np.ndarray]] = {}
        for roi in config.rois:
            references = {}
            for state, path in (
                ("CLOSED", roi.closed_reference),
                ("OPEN", roi.open_reference),
            ):
                if not path:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"Cannot read {state} reference for {roi.name}: {path}")
                references[state] = self._prepare(image)
            if not references:
                raise RuntimeError(
                    f"breaker ROI {roi.name!r} requires closed_reference or open_reference"
                )
            self._references[roi.name] = references

    def predict(self, crop: np.ndarray, *, asset_id: str) -> Detection:
        prepared = self._prepare(crop)
        similarities = {
            state: self._similarity(reference, prepared)
            for state, reference in self._references[asset_id].items()
        }
        ranked = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        state, similarity = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = similarity - runner_up if len(ranked) > 1 else similarity
        valid = similarity >= self.config.reference_similarity and (
            len(ranked) == 1 or margin >= self.config.reference_margin
        )
        class_name = state if valid else "UNKNOWN"
        return Detection(
            bbox=(0.0, 0.0, float(crop.shape[1]), float(crop.shape[0])),
            confidence=float(max(0.0, min(1.0, similarity))),
            class_id={"CLOSED": 0, "OPEN": 1}.get(class_name, -1),
            class_name=class_name,
            metadata={
                "reference_similarities": similarities,
                "reference_margin": float(margin),
                "observation_valid": valid,
                "decision_basis": "site_reference_geometry",
            },
        )

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape[:2]
        x1 = int(round(width * self.config.reference_x1_ratio))
        x2 = int(round(width * self.config.reference_x2_ratio))
        y1 = int(round(height * self.config.reference_y1_ratio))
        y2 = int(round(height * self.config.reference_y2_ratio))
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            raise ValueError("breaker reference geometry region is empty")
        region = cv2.resize(
            region,
            (self.config.reference_width, self.config.reference_height),
            interpolation=cv2.INTER_AREA,
        )
        region = cv2.GaussianBlur(region, (3, 3), 0)
        return cv2.equalizeHist(region)

    def _similarity(self, reference: np.ndarray, observed: np.ndarray) -> float:
        search = max(0, int(self.config.reference_search_pixels))
        if search == 0:
            return float(cv2.matchTemplate(observed, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
        padded = cv2.copyMakeBorder(
            observed,
            search,
            search,
            search,
            search,
            cv2.BORDER_REFLECT,
        )
        result = cv2.matchTemplate(padded, reference, cv2.TM_CCOEFF_NORMED)
        return float(result.max())


def save_breaker_reference(
    frame: np.ndarray,
    roi: BreakerRoiConfig,
    output: str | Path,
) -> Path:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi.bbox
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid breaker ROI for {roi.name}: {roi.bbox}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame[top:bottom, left:right]):
        raise RuntimeError(f"Cannot write breaker reference: {destination}")
    return destination
